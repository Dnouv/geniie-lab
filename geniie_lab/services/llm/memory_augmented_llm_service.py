import ast
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Callable, Literal, Type

from pydantic import BaseModel, Field

from geniie_lab.memory import ConversationHistory
from geniie_lab.response import Clicks, NextAction, Query, RelevanceJudgement
from geniie_lab.services.llm.llm_service_protocol import LLMServiceProtocol
from geniie_lab.services.memory_store.sqlite_memory_store import SQLiteMemoryStore


class MemoryToolDecision(BaseModel):
    action: Literal["read_sql", "write_memory", "delete_memory", "noop"] = Field(
        ...,
        description="Choose one DB action at a time. Use noop when no DB operation is useful for this step.",
    )
    content: str | None = Field(
        default=None,
        description="Arbitrary memory text to store for this topic.",
    )
    sql_query: str | None = Field(
        default=None,
        description="A single SELECT query against the scoped SQL views exposed for this topic.",
    )
    memory_id: int | None = Field(default=None)
    match_text: str | None = Field(default=None)
    max_delete: int = Field(default=5, ge=1, le=50)
    reason: str | None = Field(default=None)


@dataclass
class MemoryStagePermissions:
    can_query: bool
    can_write: bool
    can_delete: bool

    def has_any(self) -> bool:
        return self.can_query or self.can_write or self.can_delete


class MemoryAugmentedLLMService(LLMServiceProtocol):
    def __init__(
        self,
        *,
        base_service: LLMServiceProtocol,
        memory_store: SQLiteMemoryStore,
        mode: str,
        run_id: str,
        topic_id: str,
        max_db_ops_per_stage: int,
        default_query_top_k: int,
    ):
        self.base_service = base_service
        self.memory_store = memory_store
        self.mode = mode
        self.run_id = run_id
        self.topic_id = topic_id
        self.max_db_ops_per_stage = max(1, max_db_ops_per_stage)
        self.default_query_top_k = max(1, min(default_query_top_k, 50))

    def create_query(
        self,
        model: str,
        temperature: float,
        memory: ConversationHistory,
        instruction,
        reasoning_mode: str | None = None,
    ) -> Query:
        return self._run_stage_with_memory_tools(
            stage="query",
            model=model,
            temperature=temperature,
            memory=memory,
            instruction_text=instruction.generate(),
            response_model=Query,
            reasoning_mode=reasoning_mode,
        )

    def recreate_query(
        self,
        model: str,
        temperature: float,
        memory: ConversationHistory,
        instruction,
        reasoning_mode: str | None = None,
    ) -> Query:
        return self._run_stage_with_memory_tools(
            stage="reformulate",
            model=model,
            temperature=temperature,
            memory=memory,
            instruction_text=instruction.generate(),
            response_model=Query,
            reasoning_mode=reasoning_mode,
        )

    def create_clicks(
        self,
        model: str,
        temperature: float,
        memory: ConversationHistory,
        instruction,
        reasoning_mode: str | None = None,
    ) -> Clicks:
        return self._run_stage_with_memory_tools(
            stage="click",
            model=model,
            temperature=temperature,
            memory=memory,
            instruction_text=instruction.generate(),
            response_model=Clicks,
            reasoning_mode=reasoning_mode,
        )

    def calc_relevance_judgement(
        self,
        model: str,
        temperature: float,
        memory: ConversationHistory,
        instruction,
        reasoning_mode: str | None = None,
    ) -> RelevanceJudgement:
        return self._run_stage_with_memory_tools(
            stage="relevance",
            model=model,
            temperature=temperature,
            memory=memory,
            instruction_text=instruction.generate(),
            response_model=RelevanceJudgement,
            reasoning_mode=reasoning_mode,
        )

    def decide_next_action(
        self,
        model: str,
        temperature: float,
        memory: ConversationHistory,
        instruction,
        reasoning_mode: str | None = None,
    ) -> NextAction:
        return self._run_stage_with_memory_tools(
            stage="action",
            model=model,
            temperature=temperature,
            memory=memory,
            instruction_text=instruction.generate(),
            response_model=NextAction,
            reasoning_mode=reasoning_mode,
        )

    def generate_structured(
        self,
        model: str,
        temperature: float,
        memory: ConversationHistory,
        prompt: str,
        response_model: Type[BaseModel],
        reasoning_mode: str | None = None,
        stage: str | None = None,
    ) -> BaseModel:
        return self.base_service.generate_structured(
            model=model,
            temperature=temperature,
            memory=memory,
            prompt=prompt,
            response_model=response_model,
            reasoning_mode=reasoning_mode,
            stage=stage,
        )

    def get_tokenizer(self, model_name: str) -> Callable[[str], int]:
        return self.base_service.get_tokenizer(model_name)

    def get_max_tokens(self, model_name: str) -> int:
        return self.base_service.get_max_tokens(model_name)

    def _run_stage_with_memory_tools(
        self,
        *,
        stage: str,
        model: str,
        temperature: float,
        memory: ConversationHistory,
        instruction_text: str,
        response_model: Type[BaseModel],
        reasoning_mode: str | None,
    ) -> BaseModel:
        permissions = self._permissions_for_stage(stage)
        active_memory = self._select_active_memory(memory)
        stage_stats = self._new_stage_stats()

        if permissions.has_any():
            tool_feedback: list[str] = []
            for step_idx in range(1, self.max_db_ops_per_stage + 1):
                prompt = self._build_tool_prompt(
                    stage=stage,
                    instruction_text=instruction_text,
                    permissions=permissions,
                    tool_feedback=tool_feedback,
                    step_idx=step_idx,
                )
                try:
                    decision_obj = self.base_service.generate_structured(
                        model=model,
                        temperature=temperature,
                        memory=active_memory,
                        prompt=prompt,
                        response_model=MemoryToolDecision,
                        reasoning_mode=reasoning_mode,
                        stage=stage,
                    )
                except Exception as exc:
                    failed_generation = self._extract_failed_generation(exc)
                    recovered_output = self._recover_direct_stage_output(
                        failed_generation=failed_generation,
                        response_model=response_model,
                    )
                    if recovered_output is not None:
                        stage_stats["implicit_stage_output_recoveries"] += 1
                        stage_stats["steps_completed"] = step_idx
                        stage_stats["stopped_early"] = True
                        self._emit_memory_event(
                            stage=stage,
                            op="implicit_stage_output",
                            payload={
                                "step_num": step_idx,
                                "failed_generation": failed_generation,
                                "recovered_output": recovered_output.model_dump(exclude_none=True),
                            },
                        )
                        self._emit_stage_summary(
                            stage=stage,
                            stage_stats=stage_stats,
                            fallback_used=False,
                        )
                        return recovered_output
                    self._emit_memory_event(
                        stage=stage,
                        op="decision_generation_failed",
                        payload={
                            "step_num": step_idx,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "failed_generation": failed_generation,
                        },
                    )
                    if self._is_schema_validation_error(exc):
                        stage_stats["decision_generation_failures"] += 1
                        stage_stats["memory_phase_skipped"] = True
                        stage_stats["stopped_early"] = True
                        break
                    self._emit_stage_summary(
                        stage=stage,
                        stage_stats=stage_stats,
                        fallback_used=False,
                    )
                    raise
                decision = MemoryToolDecision.model_validate(decision_obj.model_dump())
                action = decision.action
                stage_stats["action_attempts"][action] += 1
                self._emit_memory_event(
                    stage=stage,
                    op="decision_attempt",
                    payload={
                        "step_num": step_idx,
                        "decision": decision.model_dump(exclude_none=True),
                    },
                )
                status, feedback, outcome = self._execute_decision(
                    stage=stage,
                    decision=decision,
                    step_idx=step_idx,
                )
                if outcome["success"]:
                    stage_stats["action_successes"][action] += 1
                stage_stats["steps_completed"] = step_idx
                if feedback:
                    tool_feedback.append(feedback)
                if status == "stop":
                    stage_stats["stopped_early"] = True
                    break

        self._emit_stage_summary(
            stage=stage,
            stage_stats=stage_stats,
            fallback_used=stage_stats["memory_phase_skipped"],
        )
        return self.base_service.generate_structured(
            model=model,
            temperature=temperature,
            memory=active_memory,
            prompt=instruction_text,
            response_model=response_model,
            reasoning_mode=reasoning_mode,
            stage=stage,
        )

    def _select_active_memory(self, memory: ConversationHistory) -> ConversationHistory:
        if self.mode != "db_only_reformulate_read":
            return memory
        messages = memory.get_all_messages()
        system = messages[0]
        return ConversationHistory(
            system_role=system.get("role"),
            system_prompt=system.get("content", ""),
            memory_policy="full",
        )

    def _permissions_for_stage(self, stage: str) -> MemoryStagePermissions:
        if self.mode == "context_plus_db":
            if stage == "reformulate":
                return MemoryStagePermissions(can_query=True, can_write=False, can_delete=False)
            if stage in {"query", "click", "relevance", "action"}:
                return MemoryStagePermissions(can_query=False, can_write=True, can_delete=True)
            return MemoryStagePermissions(can_query=False, can_write=False, can_delete=False)

        if self.mode == "db_only_reformulate_read":
            if stage == "reformulate":
                return MemoryStagePermissions(can_query=True, can_write=True, can_delete=True)
            if stage in {"query", "click", "relevance", "action"}:
                return MemoryStagePermissions(can_query=False, can_write=True, can_delete=True)
            return MemoryStagePermissions(can_query=False, can_write=False, can_delete=False)

        return MemoryStagePermissions(can_query=False, can_write=False, can_delete=False)

    def _build_tool_prompt(
        self,
        *,
        stage: str,
        instruction_text: str,
        permissions: MemoryStagePermissions,
        tool_feedback: list[str],
        step_idx: int,
    ) -> str:
        allowed_ops = []
        if permissions.can_query:
            allowed_ops.append("read_sql")
        if permissions.can_write:
            allowed_ops.append("write_memory")
        if permissions.can_delete:
            allowed_ops.append("delete_memory")
        allowed_ops.append("noop")
        allowed_ops_block = "\n".join(f"- {item}" for item in allowed_ops)
        feedback_block = "\n".join(f"- {item}" for item in tool_feedback[-8:]) if tool_feedback else "- none"
        ops_remaining_after = self.max_db_ops_per_stage - step_idx
        return (
            f"Stage: {stage}\n"
            f"DB memory mode: {self.mode}\n"
            f"DB operation step: {step_idx}/{self.max_db_ops_per_stage}\n"
            f"DB operations remaining after this step: {ops_remaining_after}\n\n"
            "You are in the memory-interaction phase.\n"
            "Choose exactly one DB action for this step.\n"
            "After these DB-operation steps are over, the runtime will make a separate call to produce the actual task answer.\n\n"
            "Task instruction:\n"
            f"{instruction_text}\n\n"
            "For memory reads, use any single SQL SELECT query you think is useful over the scoped views below.\n"
            "You decide which filters, ordering, joins, grouping, aggregations, and schema-inspection queries are useful.\n"
            "Scoped view schemas:\n"
            "- topic_memories(id, content, memory_timestamp, stage, created_at)\n"
            "- topic_memory_sql_reads(id, stage, step_num, raw_sql_query, executed_sql_query, row_count, created_at)\n"
            "- db_relations(object_name, object_type, description)\n"
            "- db_columns(object_name, column_name, data_type, description)\n"
            "Use a single SELECT statement. WITH ... SELECT is allowed.\n\n"
            "For memory writes, provide only:\n"
            "- content\n"
            "You may write any free-form text that is useful for later retrieval.\n"
            "The runtime will add timestamps and stage metadata automatically.\n\n"
            "Allowed actions this step:\n"
            f"{allowed_ops_block}\n\n"
            "Recent memory operation results:\n"
            f"{feedback_block}\n\n"
            "Action rules:\n"
            "- read_sql: set sql_query and leave other operation fields empty.\n"
            "- write_memory: set content and leave other operation fields empty.\n"
            "- delete_memory: set memory_id or match_text.\n"
            "- noop: do not use any operation fields.\n"
        )

    def _execute_decision(
        self,
        *,
        stage: str,
        decision: MemoryToolDecision,
        step_idx: int,
    ) -> tuple[str, str, dict]:
        action = decision.action
        if action == "read_sql":
            if not self._permissions_for_stage(stage).can_query:
                return "continue", "read_sql rejected: querying is not allowed in this stage.", {"success": False}
            raw_sql_query = (decision.sql_query or "").strip()
            try:
                rows, executed_sql = self.memory_store.execute_read_sql(
                    run_id=self.run_id,
                    topic_id=self.topic_id,
                    sql_query=raw_sql_query,
                )
            except Exception as exc:
                return "continue", f"read_sql failed: {exc}", {"success": False}
            self.memory_store.log_sql_read(
                run_id=self.run_id,
                topic_id=self.topic_id,
                stage=stage,
                step_num=step_idx,
                raw_sql_query=raw_sql_query,
                executed_sql_query=executed_sql,
                row_count=len(rows),
            )
            self._emit_memory_event(
                stage=stage,
                op="read_sql",
                payload={
                    "step_num": step_idx,
                    "raw_sql_query": raw_sql_query,
                    "executed_sql_query": executed_sql,
                    "row_count": len(rows),
                },
            )
            return "continue", self._format_sql_feedback(rows), {"success": True}

        if action == "write_memory":
            if not self._permissions_for_stage(stage).can_write:
                return "continue", "write_memory rejected: writing is not allowed in this stage.", {"success": False}
            content = (decision.content or "").strip()
            if not content:
                return "continue", "write_memory rejected: content is required.", {"success": False}
            memory_id = self.memory_store.write_memory(
                run_id=self.run_id,
                topic_id=self.topic_id,
                stage=stage,
                memory_note=content,
            )
            self._emit_memory_event(
                stage=stage,
                op="write",
                payload={
                    "memory_id": memory_id,
                    "content": content,
                },
            )
            return "continue", f"write_memory ok: id={memory_id}", {"success": True}

        if action == "delete_memory":
            if not self._permissions_for_stage(stage).can_delete:
                return "continue", "delete_memory rejected: deletion is not allowed in this stage.", {"success": False}
            if decision.memory_id is not None:
                deleted = self.memory_store.delete_memory_by_id(
                    run_id=self.run_id,
                    topic_id=self.topic_id,
                    memory_id=decision.memory_id,
                )
                self._emit_memory_event(
                    stage=stage,
                    op="delete_by_id",
                    payload={"memory_id": decision.memory_id, "deleted": deleted},
                )
                return "continue", f"delete_memory by id result: {deleted}", {"success": deleted > 0}
            match_text = (decision.match_text or "").strip()
            if not match_text:
                return "continue", "delete_memory rejected: provide memory_id or match_text.", {"success": False}
            deleted_count = self.memory_store.delete_memories_by_text(
                run_id=self.run_id,
                topic_id=self.topic_id,
                match_text=match_text,
                max_delete=decision.max_delete,
            )
            self._emit_memory_event(
                stage=stage,
                op="delete_by_text",
                payload={"match_text": match_text, "deleted_count": deleted_count},
            )
            return "continue", f"delete_memory by text result: deleted {deleted_count}", {"success": deleted_count > 0}

        if action == "noop":
            return "stop", "noop selected", {"success": True}

        return "continue", f"Unknown action: {action}", {"success": False}

    @staticmethod
    def _format_sql_feedback(rows: list[dict]) -> str:
        if not rows:
            return "read_sql result: no matching rows."
        return "read_sql result: " + json.dumps(rows[:10], ensure_ascii=False)

    @staticmethod
    def _validate_model_output(raw: str | None, response_model: Type[BaseModel]) -> BaseModel | None:
        if raw is None:
            return None
        raw = raw.strip()
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        try:
            return response_model.model_validate(parsed)
        except Exception:
            return None

    @staticmethod
    def _extract_error_payload(exc: Exception) -> dict | None:
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            return body
        text = str(exc)
        marker = " - "
        if marker not in text:
            return None
        payload_text = text.split(marker, 1)[1].strip()
        try:
            parsed = ast.literal_eval(payload_text)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

    @classmethod
    def _extract_failed_generation(cls, exc: Exception) -> str | None:
        payload = cls._extract_error_payload(exc)
        if not payload:
            return None
        error = payload.get("error")
        if not isinstance(error, dict):
            return None
        failed_generation = error.get("failed_generation")
        if not isinstance(failed_generation, str):
            return None
        return failed_generation

    @classmethod
    def _is_schema_validation_error(cls, exc: Exception) -> bool:
        payload = cls._extract_error_payload(exc)
        if payload:
            error = payload.get("error")
            if isinstance(error, dict) and error.get("code") == "json_validate_failed":
                return True
        return "json_validate_failed" in str(exc)

    @classmethod
    def _recover_direct_stage_output(
        cls,
        *,
        failed_generation: str | None,
        response_model: Type[BaseModel],
    ) -> BaseModel | None:
        return cls._validate_model_output(failed_generation, response_model)

    @staticmethod
    def _new_stage_stats() -> dict:
        return {
            "action_attempts": defaultdict(int),
            "action_successes": defaultdict(int),
            "steps_completed": 0,
            "stopped_early": False,
            "memory_phase_skipped": False,
            "decision_generation_failures": 0,
            "implicit_stage_output_recoveries": 0,
        }

    def _emit_stage_summary(self, *, stage: str, stage_stats: dict, fallback_used: bool) -> None:
        self._emit_memory_event(
            stage=stage,
            op="stage_summary",
            payload={
                "action_attempts": dict(stage_stats["action_attempts"]),
                "action_successes": dict(stage_stats["action_successes"]),
                "steps_completed": stage_stats["steps_completed"],
                "max_db_ops_per_stage": self.max_db_ops_per_stage,
                "stopped_early": stage_stats["stopped_early"],
                "memory_phase_skipped": stage_stats["memory_phase_skipped"],
                "decision_generation_failures": stage_stats["decision_generation_failures"],
                "implicit_stage_output_recoveries": stage_stats["implicit_stage_output_recoveries"],
                "fallback_used": fallback_used,
            },
        )

    def _emit_memory_event(self, *, stage: str, op: str, payload: dict) -> None:
        record = {
            "memory_event": op,
            "stage": stage,
            "run_id": self.run_id,
            "topic_id": self.topic_id,
            "created_at": datetime.now(UTC).isoformat(),
            **payload,
        }
        print(json.dumps(record, ensure_ascii=False), file=sys.stderr)

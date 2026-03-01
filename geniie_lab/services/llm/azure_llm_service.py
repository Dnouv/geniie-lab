# Standard library
import ast
import json
import logging
import sys
import os
from typing import Callable, Protocol, Type, TypeVar

# Third-party libraries
from dotenv import load_dotenv
from openai import AzureOpenAI, BadRequestError
from openai.types.chat import ChatCompletionUserMessageParam
from pydantic import BaseModel
import tiktoken

# Local application imports
from geniie_lab.dataclasses.instruction import (
    ClickInstruction,
    NextActionInstruction,
    QueryFormulationInstruction,
    QueryReFormulationInstruction,
    RelevanceJudgementInstruction,
)
from geniie_lab.memory import ConversationHistory
from geniie_lab.response import Clicks, NextAction, Query, RelevanceJudgement
from geniie_lab.services.llm.llm_utils import instruction_stage, extract_message_reasoning

T = TypeVar("T", bound=BaseModel)

class InstructionWithGenerate(Protocol):
    def generate(self) -> str:
        ...

class AzureOpenAILLMService:
    _MAX_TOKEN_LIMITS = {
        "gpt-4o":             131072,  # GPT-4o large context
        "gpt-4o-mini":         8192,
        "gpt-4-turbo":         8192,
        "gpt-4.1-mini-2025-04-14": 1000000,
        "gpt-4.1-mini":        8192,
        "gpt-4":               8192,
        "gpt-4-32k":          32768,

        "gpt-3.5-turbo-16k":  16384,
        "gpt-3.5-turbo":       4096,

        "text-embedding-ada-002": 8191,
        "text-embedding-3-small": 8191,
        "text-embedding-3-large": 8191,
    }
    _MAX_JSON_RETRIES = 3

    def __init__(self, log_llm_io: bool = False):
        load_dotenv()
        self.client = AzureOpenAI(
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            api_key=os.getenv("AZURE_API_KEY")
        )
        self.log_llm_io = log_llm_io

    def _emit_llm_io(self, stage: str | None, model: str, direction: str, payload: dict) -> None:
        if not self.log_llm_io:
            return
        record = {
            "llm_io": direction,
            "stage": stage,
            "model": model,
            **payload,
        }
        print(json.dumps(record, ensure_ascii=False), file=sys.stderr)

    def _emit_llm_error(
        self,
        stage: str | None,
        model: str,
        response_model_name: str,
        attempt: int,
        exc: Exception,
    ) -> None:
        payload: dict = {
            "attempt": attempt,
            "response_model": response_model_name,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if isinstance(exc, BadRequestError):
            error_body = self._extract_error_body(exc)
            if error_body is not None:
                payload["error_body"] = error_body
                err = error_body.get("error")
                if isinstance(err, dict) and "failed_generation" in err:
                    payload["failed_generation"] = err.get("failed_generation")
        self._emit_llm_io(stage, model, "error", payload)

    def _call_llm_with_pydantic_response(
        self,
        model: str,
        temperature: float,
        memory: ConversationHistory,
        instruction: InstructionWithGenerate,
        response_model: Type[T],
        reasoning_mode: str | None = None,
    ) -> T:

        stage = instruction_stage(instruction)
        memory.add_user_message(instruction.generate(), stage=stage)
        messages_dicts: list[dict[str, str]] = memory.get_messages(tokenizer=self.get_tokenizer(model), max_tokens=self.get_max_tokens(model))
        self._emit_llm_io(stage, model, "input", {"messages": messages_dicts})
        messages: list[ChatCompletionUserMessageParam] = [
            ChatCompletionUserMessageParam(role="user", content=msg["content"]) for msg in messages_dicts
        ]

        last_error: Exception | None = None
        for attempt in range(1, self._MAX_JSON_RETRIES + 1):
            try:
                request_kwargs = {
                    "model": model,
                    "messages": messages,
                    "response_format": response_model,
                    "temperature": temperature,
                }
                if reasoning_mode is not None:
                    request_kwargs["reasoning_effort"] = reasoning_mode
                completion = self.client.beta.chat.completions.parse(**request_kwargs)
                message = completion.choices[0].message
                parsed_response = message.parsed
                if parsed_response is None:
                    raise ValueError(f"LLM returned empty parsed object for {response_model.__name__}.")
                reasoning = extract_message_reasoning(message) if stage in {"query", "reformulate"} else None
                self._emit_llm_io(stage, model, "output", {"message": message.to_json()})
                memory.add_assistant_response(message.to_json(), stage=stage, reasoning=reasoning)
                return parsed_response
            except BadRequestError as exc:
                last_error = exc
                self._emit_llm_error(stage, model, response_model.__name__, attempt, exc)
                salvaged = self._try_salvage_failed_generation(exc, response_model, stage, model, memory)
                if salvaged is not None:
                    return salvaged
                if self._should_retry_rate_limit(exc) and attempt < self._MAX_JSON_RETRIES:
                    logging.warning(
                        "Retrying %s due to rate-limit/timeout (attempt %s/%s).",
                        response_model.__name__,
                        attempt + 1,
                        self._MAX_JSON_RETRIES,
                    )
                    continue
                raise
            except Exception as exc:
                self._emit_llm_error(stage, model, response_model.__name__, attempt, exc)
                raise
        if last_error:
            raise last_error
        raise RuntimeError("LLM call failed without exception.")

    def _try_salvage_failed_generation(
        self,
        exc: BadRequestError,
        response_model: Type[T],
        stage: str | None,
        model: str,
        memory: ConversationHistory,
    ) -> T | None:
        raw = self._extract_failed_generation(exc)
        if raw is None:
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        unwrapped = self._unwrap_response_payload(payload, response_model)
        if unwrapped is None:
            return None
        try:
            parsed = response_model.model_validate(unwrapped)
        except Exception:
            return None
        canonical = json.dumps(parsed.model_dump(), ensure_ascii=False)
        self._emit_llm_io(stage, model, "output", {"message": raw, "source": "failed_generation", "memory_message": canonical})
        memory.add_assistant_response(canonical, stage=stage, reasoning=None)
        return parsed

    @staticmethod
    def _extract_failed_generation(exc: BadRequestError) -> str | None:
        candidate = AzureOpenAILLMService._extract_error_body(exc)

        if isinstance(candidate, dict):
            err = candidate.get("error")
            if isinstance(err, dict):
                failed = err.get("failed_generation")
                if isinstance(failed, str):
                    return failed
        return None

    @staticmethod
    def _extract_error_body(exc: BadRequestError) -> dict | None:
        def _coerce_dict(raw: str) -> dict | None:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(raw)
                except Exception:
                    return None
            return parsed if isinstance(parsed, dict) else None

        body = exc.body
        if isinstance(body, dict):
            return body
        if isinstance(body, str):
            return _coerce_dict(body)
        message = str(exc)
        if " - " in message:
            return _coerce_dict(message.split(" - ", 1)[1].strip())
        return None

    @staticmethod
    def _unwrap_response_payload(payload: object, response_model: Type[T]) -> dict | None:
        required = set(response_model.model_json_schema().get("required") or [])

        def is_match(obj: object) -> bool:
            return isinstance(obj, dict) and required.issubset(obj.keys())

        if is_match(payload):
            return payload  # type: ignore[return-value]

        if isinstance(payload, dict):
            for key in ("parsed", "content", "output", "data"):
                value = payload.get(key)
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        value = None
                if value is not None and is_match(value):
                    return value  # type: ignore[return-value]

        if isinstance(payload, str):
            try:
                inner = json.loads(payload)
            except json.JSONDecodeError:
                return None
            if is_match(inner):
                return inner  # type: ignore[return-value]

        return None

    @staticmethod
    def _should_retry_rate_limit(exc: BadRequestError) -> bool:
        message = str(exc).lower()
        return "rate limit" in message or "timeout" in message or "overloaded" in message

    def get_tokenizer(self, model_name: str) -> Callable[[str], int]:

        try:
            enc = tiktoken.encoding_for_model(model_name)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return lambda text: len(enc.encode(text))

    def get_max_tokens(self, model_name: str) -> int:

        name = model_name.lower()
        for prefix, limit in self._MAX_TOKEN_LIMITS.items():
            if name.startswith(prefix):
                return limit
        return 4096

    def create_query(self, model: str, temperature: float, memory: ConversationHistory, instruction: QueryFormulationInstruction, reasoning_mode: str | None = None) -> Query:

        query = self._call_llm_with_pydantic_response(model, temperature, memory, instruction, Query, reasoning_mode)
        return query

    def recreate_query(self, model: str, temperature: float, memory: ConversationHistory, instruction: QueryReFormulationInstruction, reasoning_mode: str | None = None) -> Query:

        query = self._call_llm_with_pydantic_response(model, temperature, memory, instruction, Query, reasoning_mode)
        return query

    def create_clicks(self, model: str, temperature: float, memory: ConversationHistory, instruction: ClickInstruction, reasoning_mode: str | None = None) -> Clicks:

        return self._call_llm_with_pydantic_response(model, temperature, memory, instruction, Clicks, reasoning_mode)

    def calc_relevance_judgement(self, model: str, temperature: float, memory: ConversationHistory, instruction: RelevanceJudgementInstruction, reasoning_mode: str | None = None) -> RelevanceJudgement:

        return self._call_llm_with_pydantic_response(model, temperature, memory, instruction, RelevanceJudgement, reasoning_mode)

    def decide_next_action(self, model: str, temperature: float, memory: ConversationHistory, instruction: NextActionInstruction, reasoning_mode: str | None = None) -> NextAction:
        return self._call_llm_with_pydantic_response(model, temperature, memory, instruction, NextAction, reasoning_mode)

import sys
import json
import re
from typing import List, Dict, Callable, Optional, Any

class ConversationHistory:
    def __init__(self, system_role: str | None, system_prompt: str, memory_policy: Optional[str] = None):
        if system_role is None:
            system_role = "system"
        self._system_prompt = {"role": system_role, "content": system_prompt}
        self._history: List[Dict] = []
        self._memory_policy = memory_policy or "full"

    def add_user_message(self, content: str, stage: Optional[str] = None):
        self._history.append({"role": "user", "content": content, "stage": stage})

    def add_assistant_response(self, response_content: str, stage: Optional[str] = None, reasoning: Optional[str] = None):
        message = {"role": "assistant", "content": response_content, "stage": stage}
        if reasoning and self._memory_policy == "forget_queries_keep_reasoning":
            message["reasoning"] = reasoning
        self._history.append(message)

    def remove_last_message(self):
        if self._history:
            self._history.pop()

    def set_memory_policy(self, policy: Optional[str]) -> None:
        self._memory_policy = policy or "full"
            
    def get_messages(self, tokenizer: Callable[[str], int], max_tokens: int) -> List[Dict[str, str]]:
        messages = [self._system_prompt]

        current_tokens = tokenizer(self._system_prompt["content"])

        for message in reversed(self._apply_memory_policy()):
            message_tokens = tokenizer(message["content"])
            if current_tokens + message_tokens > max_tokens:
                print("Context window limit reached. Pruning older messages.", file=sys.stderr)
                break

            messages.insert(1, {"role": message["role"], "content": message["content"]})
            current_tokens += message_tokens

        return messages

    def get_all_messages(self) -> List[Dict[str, str]]:
        return [self._system_prompt] + self._history

    def _apply_memory_policy(self) -> List[Dict]:
        if self._memory_policy in (None, "full"):
            return self._history

        query_stages = {"query", "reformulate"}
        last_idx = len(self._history) - 1
        query_segments: List[List[int]] = []
        current: List[int] = []
        prev_idx: Optional[int] = None
        prev_is_query = False

        for idx, msg in enumerate(self._history):
            is_query = msg.get("stage") in query_stages
            if is_query:
                if current and prev_is_query and prev_idx is not None and idx == prev_idx + 1:
                    current.append(idx)
                else:
                    if current:
                        query_segments.append(current)
                    current = [idx]
            else:
                if current:
                    query_segments.append(current)
                    current = []
            prev_idx = idx
            prev_is_query = is_query

        if current:
            query_segments.append(current)

        if self._memory_policy == "forget_queries":
            drop_set = set()
            for segment in query_segments:
                if last_idx in segment:
                    continue
                drop_set.update(segment)
            return [msg for idx, msg in enumerate(self._history) if idx not in drop_set]

        if self._memory_policy == "forget_queries_keep_reason":
            redacted: List[Dict[str, Any]] = []
            for msg in self._history:
                if msg.get("role") == "assistant" and msg.get("stage") in query_stages:
                    redacted.append({**msg, "content": self._redact_query_text(msg.get("content", ""))})
                else:
                    redacted.append(msg)
            return redacted

        if self._memory_policy == "forget_queries_keep_reasoning":
            redacted: List[Dict[str, Any]] = []
            for msg in self._history:
                if msg.get("role") == "assistant" and msg.get("stage") in query_stages:
                    reasoning = msg.get("reasoning")
                    redacted.append(
                        {**msg, "content": self._redact_query_text(msg.get("content", ""), reasoning=reasoning)}
                    )
                else:
                    redacted.append(msg)
            return redacted

        if self._memory_policy == "forget_queries_half":
            if not query_segments:
                return self._history
            cutoff = len(query_segments) // 2
            drop_set = set()
            for segment in query_segments[:cutoff]:
                if last_idx in segment:
                    continue
                drop_set.update(segment)
            return [msg for idx, msg in enumerate(self._history) if idx not in drop_set]

        return self._history

    def _redact_query_text(self, content: str, reasoning: Optional[str] = None) -> str:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            content = re.sub(r'("query"\\s*:\\s*)"(.*?)"', r'\\1"[REDACTED]"', content, flags=re.DOTALL)
            content = re.sub(r"('query'\\s*:\\s*)'(.*?)'", r"\\1'[REDACTED]'", content, flags=re.DOTALL)
            if not reasoning:
                content = re.sub(r'("reasoning"\\s*:\\s*)"(.*?)"', r'\\1"[REDACTED]"', content, flags=re.DOTALL)
                content = re.sub(r'("reasoning_content"\\s*:\\s*)"(.*?)"', r'\\1"[REDACTED]"', content, flags=re.DOTALL)
            if reasoning:
                content = f"{content}\\nReasoning: {reasoning}"
            return content

        self._redact_query_fields(parsed)
        self._strip_reasoning_fields(parsed)
        if reasoning:
            self._inject_reasoning(parsed, reasoning)

        inner = parsed.get("content")
        if isinstance(inner, str):
            try:
                inner_parsed = json.loads(inner)
            except json.JSONDecodeError:
                inner_parsed = None
            if inner_parsed is not None:
                self._redact_query_fields(inner_parsed)
                self._strip_reasoning_fields(inner_parsed)
                if reasoning:
                    self._inject_reasoning(inner_parsed, reasoning)
                parsed["content"] = json.dumps(inner_parsed, ensure_ascii=False)

        return json.dumps(parsed, ensure_ascii=False)

    def _inject_reasoning(self, obj: Any, reasoning: str) -> None:
        if isinstance(obj, dict):
            if "parsed" in obj and isinstance(obj["parsed"], dict):
                obj["parsed"]["reasoning"] = reasoning
                return
            if "content" in obj and isinstance(obj["content"], dict):
                obj["content"]["reasoning"] = reasoning
                return
            obj.setdefault("reasoning", reasoning)

    def _strip_reasoning_fields(self, obj: Any) -> None:
        if isinstance(obj, dict):
            obj.pop("reasoning", None)
            obj.pop("reasoning_content", None)
            for value in obj.values():
                self._strip_reasoning_fields(value)
        elif isinstance(obj, list):
            for item in obj:
                self._strip_reasoning_fields(item)

    def _redact_query_fields(self, obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "query":
                    obj[key] = "[REDACTED]"
                else:
                    self._redact_query_fields(value)
        elif isinstance(obj, list):
            for item in obj:
                self._redact_query_fields(item)

    def clone(self) -> "ConversationHistory":
        from copy import deepcopy
        cloned = ConversationHistory(
            system_role=self._system_prompt["role"],
            system_prompt=self._system_prompt["content"],
            memory_policy=self._memory_policy,
        )
        cloned._history = deepcopy(self._history)
        return cloned

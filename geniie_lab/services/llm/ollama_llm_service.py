# Standard library
import json
import sys
from typing import Callable, Protocol, Type, TypeVar

# Third-party libraries
from dotenv import load_dotenv
from openai import BadRequestError, OpenAI
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


class RawPromptInstruction:
    def __init__(self, prompt: str, stage_name: str | None = None):
        self.prompt = prompt
        self.stage_name = stage_name

    def generate(self) -> str:
        return self.prompt


class OllamaLLMService:
    def __init__(self, log_llm_io: bool = False):
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # required, but unused
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
        exc: Exception,
    ) -> None:
        payload: dict = {
            "response_model": response_model_name,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if isinstance(exc, BadRequestError):
            body = getattr(exc, "body", None)
            if isinstance(body, dict):
                payload["error_body"] = body
            elif isinstance(body, str):
                payload["error_body"] = body
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
        except Exception as exc:
            self._emit_llm_error(stage, model, response_model.__name__, exc)
            raise

    def get_tokenizer(self, model_name: str) -> Callable[[str], int]:
        enc = tiktoken.get_encoding("cl100k_base")
        return lambda text: len(enc.encode(text))

    def get_max_tokens(self, model_name: str) -> int:
        return 128000

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

    def generate_structured(
        self,
        model: str,
        temperature: float,
        memory: ConversationHistory,
        prompt: str,
        response_model: Type[T],
        reasoning_mode: str | None = None,
        stage: str | None = None,
    ) -> T:
        instruction = RawPromptInstruction(prompt=prompt, stage_name=stage)
        return self._call_llm_with_pydantic_response(
            model=model,
            temperature=temperature,
            memory=memory,
            instruction=instruction,
            response_model=response_model,
            reasoning_mode=reasoning_mode,
        )

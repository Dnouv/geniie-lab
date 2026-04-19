# Standard library
import json
import os
import sys
from typing import Any, Callable, Dict, List, Protocol, Type, TypeVar

# Third-party libraries
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

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


class GeminiLLMService:
    def __init__(self, log_llm_io: bool = False):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
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

    def _call_llm_and_parse(
        self,
        model: str,
        temperature: float,
        memory: ConversationHistory,
        instruction: InstructionWithGenerate,
        response_model: Type[T]
    ) -> T:

        stage = instruction_stage(instruction)
        memory.add_user_message(instruction.generate(), stage=stage)
        openai_messages = memory.get_messages(
            tokenizer=self.get_tokenizer(model),
            max_tokens=self.get_max_tokens(model)
        )
        self._emit_llm_io(stage, model, "input", {"messages": openai_messages})

        system_prompt = openai_messages[0]['content'] if openai_messages and openai_messages[0]['role'] == 'system' else None
        gemini_contents: List[Dict[str, Any]] = []
        for msg in openai_messages[1:]:
            role = "model" if msg["role"] == "assistant" else "user"
            gemini_contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })

        response = self.client.models.generate_content(
            model=model,
            contents=gemini_contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                response_mime_type="application/json",
                response_schema=response_model,
            ),
        )
        if response.text is None:
            raise ValueError(f"Response text is None for {response_model.__name__}.")
        self._emit_llm_io(stage, model, "output", {"message": response.text})
        reasoning = extract_message_reasoning(getattr(response, "candidates", None)) if stage in {"query", "reformulate"} else None
        memory.add_assistant_response(response.text, stage=stage, reasoning=reasoning)
        data = json.loads(response.text)
        return response_model(**data)

    def get_tokenizer(self, model_name: str) -> Callable[[str], int]:
        def count_fn(text: str) -> int:
            resp = self.client.models.count_tokens(model=model_name, contents=text)
            token_count = getattr(resp, "total_tokens", None)
            if token_count is None:
                raise ValueError(f"Token count is None for model {model_name}.")
            return token_count
        return count_fn

    def get_max_tokens(self, model_name: str) -> int:
        return 1_048_576

    def create_query(self, model: str, temperature: float, memory: ConversationHistory, instruction: QueryFormulationInstruction, reasoning_mode: str | None = None) -> Query:

        query = self._call_llm_and_parse(model, temperature, memory, instruction, Query)
        return query

    def recreate_query(self, model: str, temperature: float, memory: ConversationHistory, instruction: QueryReFormulationInstruction, reasoning_mode: str | None = None) -> Query:

        query = self._call_llm_and_parse(model, temperature, memory, instruction, Query)
        return query

    def create_clicks(self, model: str, temperature: float, memory: ConversationHistory, instruction: ClickInstruction, reasoning_mode: str | None = None) -> Clicks:

        return self._call_llm_and_parse(model, temperature, memory, instruction, Clicks)

    def calc_relevance_judgement(self, model: str, temperature: float, memory: ConversationHistory, instruction: RelevanceJudgementInstruction, reasoning_mode: str | None = None) -> RelevanceJudgement:

        return self._call_llm_and_parse(model, temperature, memory, instruction, RelevanceJudgement)

    def decide_next_action(self, model: str, temperature: float, memory: ConversationHistory, instruction: NextActionInstruction, reasoning_mode: str | None = None) -> NextAction:
        return self._call_llm_and_parse(model, temperature, memory, instruction, NextAction)

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
        return self._call_llm_and_parse(model, temperature, memory, instruction, response_model)

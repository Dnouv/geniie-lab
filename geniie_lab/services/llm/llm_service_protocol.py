# Standard library
from typing import Callable, Protocol

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


class LLMServiceProtocol(Protocol):
    def create_query(self, model: str, temperature: float, memory: ConversationHistory, instruction: QueryFormulationInstruction, reasoning_mode: str | None = None) -> Query:
        ...
    def create_clicks(self, model: str, temperature: float, memory: ConversationHistory, instruction: ClickInstruction, reasoning_mode: str | None = None) -> Clicks:
        ...
    def recreate_query(self, model: str, temperature: float, memory: ConversationHistory, instruction: QueryReFormulationInstruction, reasoning_mode: str | None = None) -> Query:
        ...
    def calc_relevance_judgement(self, model: str, temperature: float, memory: ConversationHistory, instruction: RelevanceJudgementInstruction, reasoning_mode: str | None = None) -> RelevanceJudgement:
        ...
    def decide_next_action(self, model: str, temperature: float, memory: ConversationHistory, instruction: NextActionInstruction, reasoning_mode: str | None = None) -> NextAction:
        ...
    def get_tokenizer(self, model_name: str) -> Callable[[str], int]: ...
    def get_max_tokens(self, model_name: str) -> int: ...

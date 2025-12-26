from typing import Optional, Any

from geniie_lab.dataclasses.instruction import (
    ClickInstruction,
    NextActionInstruction,
    QueryFormulationInstruction,
    QueryReFormulationInstruction,
    RelevanceJudgementInstruction,
)


def instruction_stage(instruction) -> Optional[str]:
    if isinstance(instruction, QueryFormulationInstruction):
        return "query"
    if isinstance(instruction, QueryReFormulationInstruction):
        return "reformulate"
    if isinstance(instruction, ClickInstruction):
        return "click"
    if isinstance(instruction, RelevanceJudgementInstruction):
        return "relevance"
    if isinstance(instruction, NextActionInstruction):
        return "action"
    return None


def extract_message_reasoning(message: Any) -> Optional[str]:
    if message is None:
        return None
    for key in ("reasoning", "reasoning_content"):
        value = getattr(message, key, None)
        if value:
            return str(value)
    extra = getattr(message, "model_extra", None)
    if isinstance(extra, dict):
        for key in ("reasoning", "reasoning_content"):
            value = extra.get(key)
            if value:
                return str(value)
    if hasattr(message, "to_dict"):
        try:
            data = message.to_dict()
        except Exception:
            data = None
        if isinstance(data, dict):
            for key in ("reasoning", "reasoning_content"):
                value = data.get(key)
                if value:
                    return str(value)
    if isinstance(message, dict):
        for key in ("reasoning", "reasoning_content"):
            value = message.get(key)
            if value:
                return str(value)
    return None

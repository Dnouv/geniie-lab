from enum import Enum
from typing import Annotated, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Enums
class Relevance(str, Enum):
    """Document relevance enumeration."""
    RELEVANT = "Relevant"
    NOT_RELEVANT = "NotRelevant"

class Action(Enum):
    """Enum describing possible user actions."""
    SUBMIT_NEW_QUERY = "SUBMIT_NEW_QUERY"
    CLICK_DOCUMENT = "CLICK_DOCUMENT"
    GO_NEXT_RESULT_PAGE = "GO_NEXT_RESULT_PAGE"
    END_TASK = "END_TASK"

# Models
class Query(BaseModel):
    """A model for submitting a query to a search tool."""
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "hubble telescope scientific discoveries",
                    "reason": "Target specific discoveries to retrieve relevant documents.",
                }
            ]
        }
    )

    query: str = Field(
        ...,
        title="query",
        min_length=1,
        description="A non-empty query string submitted to the search tool.",
        examples=["hubble telescope scientific discoveries"],
    )
    start: Optional[int] = Field(
        0,
        title="start",
        description=(
            "The starting index of the search results. Optional; defaults to task start_offset."
        )
    )
    size: Optional[int] = Field(
        10,
        title="size",
        description=(
            "The number of documents per search result page. Optional; defaults to task serp_size."
        )
    )
    reason: str = Field(
        ...,
        title="reason",
        min_length=1,
        description="A brief non-empty explanation of the intent behind your query.",
        examples=["Target specific discoveries to retrieve relevant documents."],
    )

    @field_validator("query")
    @classmethod
    def _validate_non_empty_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must be non-empty")
        return value

    @field_validator("reason")
    @classmethod
    def _validate_non_empty_reason(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("reason must be non-empty")
        return value

    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        schema = super().model_json_schema(*args, **kwargs)
        properties = schema.get("properties", {})
        for key in ("start", "size"):
            properties.pop(key, None)
        schema["properties"] = properties
        if "required" in schema:
            schema["required"] = [key for key in schema["required"] if key not in {"start", "size"}]
        return schema

class Clicks(BaseModel):
    """
    A model for selecting multiple documents from search results.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "ranking_list": [1, 3, 8],
                    "reason": "These results appear most likely relevant based on title/snippet.",
                },
                {
                    "ranking_list": [],
                    "reason": "None of the current results appears relevant.",
                },
            ]
        }
    )

    ranking_list: List[Annotated[int, Field(ge=1, le=100)]] = Field(
        ...,
        title="ranking_list",
        min_length=0,
        max_length=10,
        description=(
            "The ranking numbers of results to examine in full text (up to 10 unique positive integers). "
            "Each rank must be a separate array item; do not concatenate multiple ranks into one number."
        ),
        examples=[[1, 3, 8], []],
    )
    reason: str = Field(
        ...,
        title="reason",
        min_length=1,
        description="A brief non-empty explanation for selecting these documents.",
        examples=["These results appear most likely relevant based on title/snippet."],
    )

    @field_validator("ranking_list")
    @classmethod
    def _validate_ranking_list(cls, value: List[int]) -> List[int]:
        if len(value) != len(set(value)):
            raise ValueError("ranking_list must not contain duplicate ranks")
        if any(rank < 1 for rank in value):
            raise ValueError("ranking_list must contain only positive ranks")
        return value

    @field_validator("reason")
    @classmethod
    def _validate_reason(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("reason must be non-empty")
        return value

class RelevanceJudgement(BaseModel):
    """A model for labeling a document's relevance."""
    label: Relevance = Field(
        ...,
        title="label",
        description=(
            "The relevance label of the document based on the information need "
            "specified in the topic file."
        )
    )
    reason: str = Field(
        ...,
        title="reason",
        description="A brief explanation supporting your judgment."
    )

class NextAction(BaseModel):
    """A model for specifying the next step toward completing a task."""
    action: Action = Field(
        ...,
        title="action",
        description="The next step to take toward completing the given task."
    )
    reason: str = Field(
        ...,
        title="reason",
        description="A brief explanation for choosing this action."
    )

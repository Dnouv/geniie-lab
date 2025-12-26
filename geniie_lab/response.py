from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

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
    query: str = Field(
        ...,
        title="query",
        description="The query string submitted to the search tool."
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
        description="A brief explanation of the intent behind your query."
    )

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
    ranking_list: List[int] = Field(
        ...,
        title="ranking_list",
        description=(
            "The ranking number of the documents in the result to examine the full text. "
        )
    )
    reason: str = Field(
        ...,
        title="reason",
        description="A brief explanation for selecting these documents."
    )

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

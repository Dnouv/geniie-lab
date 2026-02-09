# Standard library
from dataclasses import dataclass
from textwrap import dedent
from typing import Union, Optional, List

# Local application imports
from geniie_lab.dataclasses.description import (
    CorpusDescription,
    TaskDescription,
    ToolDescription,
)
from geniie_lab.dataclasses.serp import FullText, Serp
from geniie_lab.dataclasses.topic import (
    TitleDescriptionNarrativeTopic, FullTopic,
    TitleDescriptionTopic,
    TitleNarrativeTopic,
    TitleOnlyTopic
)

@dataclass
class QueryFormulationInstruction:
    instruction: str
    task: TaskDescription
    corpus: CorpusDescription
    tool: ToolDescription
    topic: Union[TitleOnlyTopic, TitleDescriptionTopic, TitleNarrativeTopic, TitleDescriptionNarrativeTopic, FullTopic]

    def generate(self) -> str:
        instruction = f"""
            **Instruction**:
            {self.instruction}
            ============================
            **Task Description**: {self.task.description}
            **Corpus Description**: {self.corpus.description}
            **Search Tool Description**: {self.tool.description}
            **Topic Description**: {self.topic}
        """
        return dedent(instruction).strip()

@dataclass
class ClickInstruction:
    instruction: str
    serp: Serp
    exclude_docids: Optional[List[str]] = None

    def generate(self) -> str:
        exclude_block = ""
        if self.exclude_docids:
            exclude_preview = self.exclude_docids[:200]
            suffix = "" if len(exclude_preview) == len(self.exclude_docids) else f"\n...(and {len(self.exclude_docids) - len(exclude_preview)} more)"
            exclude_block = f"""
            **Previously clicked docids (do NOT select again)**:
            {exclude_preview}{suffix}
            """
        content = f"""
            **Instruction**:
            {self.instruction}
            ============================
            **Search results**:
            {self.serp.results}

            **Note**: Before response, ensure that all numbers in ranking_list match in the search results.
            {exclude_block}
        """
        return dedent(content).strip()

@dataclass
class RelevanceJudgementInstruction:
    instruction: str
    fulltext: FullText

    def generate(self) -> str:
        instruction = f"""
            **Instruction**:
            {self.instruction}
            ============================
            **Document**: {self.fulltext.title}
            {self.fulltext.text}
        """
        return dedent(instruction).strip()

@dataclass
class QueryReFormulationInstruction:
    instruction: str
    task: TaskDescription
    corpus: CorpusDescription
    tool: ToolDescription
    topic: Union[TitleOnlyTopic, TitleDescriptionTopic, TitleNarrativeTopic, TitleDescriptionNarrativeTopic, FullTopic]

    def generate(self) -> str:
        instruction = f"""
            **Instruction**:
            {self.instruction}
            ============================
            **Task Description**: {self.task.description}
            **Corpus Description**: {self.corpus.description}
            **Search Tool Description**: {self.tool.description}
            **Topic Description**: {self.topic}
        """
        return dedent(instruction).strip()


@dataclass
class NextActionInstruction:
    instruction: str
    task: TaskDescription

    def generate(self) -> str:
        instruction = f"""
            **Instruction**:
            {self.instruction}
            ============================
            **Task Description**: {self.task.description}
        """
        return dedent(instruction).strip()

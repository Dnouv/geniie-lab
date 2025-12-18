import sys
import pprint
from dataclasses import dataclass
import ir_datasets
import ir_measures
from typing import Protocol, Dict, Type
from collections import defaultdict
from itertools import islice

from geniie_lab.dataclasses.setting import ExperimentSettings, ExperimentState, StageConfig, Error
from geniie_lab.dataclasses.description import ModelDescription, ToolDescription
from geniie_lab.dataclasses.topic import (
    TitleDescriptionNarrativeTopic, FullTopic,
    TitleDescriptionTopic,
    TitleNarrativeTopic,
    TitleOnlyTopic,
    TopicList,
    BaseTopic
)
from geniie_lab.dataclasses.instruction import (
    ClickInstruction,
    QueryFormulationInstruction,
    QueryReFormulationInstruction,
    RelevanceJudgementInstruction,
    NextActionInstruction,
)
from geniie_lab.dataclasses.output import (
    ClickExperimentOutput,
    QueryExperimentOutput,
    QueryReformulationExperimentOutput,
    RankingExperimentOutput,
    RelevanceJudgementExperimentOutput,
    NextActionOutput,
)
from geniie_lab.memory import ConversationHistory
from geniie_lab.response import Action, NextAction
from geniie_lab.services.llm.llm_service_factory import LLMServiceFactory
from geniie_lab.services.llm.llm_service_protocol import LLMServiceProtocol
from geniie_lab.services.measure_service import MeasureService, Qrels, Run
from geniie_lab.services.opensearch.opensearch_client_factory import OpenSearchClientFactory
from geniie_lab.services.opensearch.opensearch_client_protocol import OpenSearchClientProtocol

class ExperimentStage(Protocol):
    def __init__(self, config: StageConfig): ...
    def run(self, settings: ExperimentSettings, state: ExperimentState, llm_service: LLMServiceProtocol, model: ModelDescription, tool: ToolDescription, opensearch_client: OpenSearchClientProtocol, stage_name: str) -> ExperimentState: ...
    
class QueryFormulationStage:
    DEFAULT_INSTRUCTION = """
        Review the provided descriptions of task, corpus, tool and search topic. Formulate a search query.
    """

    def __init__(self, config: StageConfig):
        self.config = config

    def run(self, settings: ExperimentSettings, state: ExperimentState, llm_service: LLMServiceProtocol, model: ModelDescription, tool: ToolDescription, opensearch_client: OpenSearchClientProtocol, stage_name: str) -> ExperimentState:
        print("\n--- Running: Query Formulation Stage ---", file=sys.stderr)
        instruction_text = self.config.instruction or self.DEFAULT_INSTRUCTION
        qf_instruction = QueryFormulationInstruction(instruction=instruction_text, task=settings.task, corpus=settings.corpus, tool=tool, topic=state.topic)

        state.query = llm_service.create_query(model.name, model.temperature, state.memory, qf_instruction)

        output = QueryExperimentOutput(
            session_name = settings.name,
            model = model.name,
            task = settings.task.name,
            dataset = settings.topicset.name,
            topic_id = state.topic.id,
            query = state.query.query,
            start = settings.task.start_offset,
            size = settings.task.serp_size
        )
        print(output.to_json(ensure_ascii=False))
        return state

class RankingStage:
    def __init__(self, config: StageConfig):
        self.config = config

    def run(self, settings: ExperimentSettings, state: ExperimentState, llm_service: LLMServiceProtocol, model: ModelDescription, tool: ToolDescription, opensearch_client: OpenSearchClientProtocol, stage_name: str) -> ExperimentState:
        print("\n--- Running: Ranking Stage ---", file=sys.stderr)

        query_text = None
        start_offset = 0
        query_text = state.query.query
        start_offset = state.query.start

        state.serp = opensearch_client.search_index_with_snippets(query_text, start=start_offset, size=settings.task.serp_size)
        state.docids = [item.docid for item in state.serp.results] if state.serp and state.serp.results else []

        dataset = ir_datasets.load(settings.topicset.name)
        if callable(dataset):
            dataset = dataset()

        qrels = Qrels()
        for row in dataset.qrels_iter():
            if row.query_id == state.topic.id:
                qrels.add(row.query_id, row.doc_id, row.relevance)

        total_rels = sum(1 for item in qrels if item.relevance and item.relevance > 0)
        cum_rel_found = len(state.judged_correct_relevant_docids)
        cum_recall = (cum_rel_found / total_rels) if total_rels else 0.0

        run = Run()
        for result in state.serp.results:
            run.add(state.topic.id, result.docid, result.ranking)
        results = MeasureService().calc(settings.task.measurement, qrels, run)

        # Additional recall@100 using top-100 retrieval (no extra LLM tokens).
        recall_run = Run()
        top100_docids = opensearch_client.search_docids(query_text, start=0, size=100)
        for idx, docid in enumerate(top100_docids, start=1):
            recall_run.add(state.topic.id, docid, idx)
        recall_metrics = MeasureService().calc([ir_measures.Recall@100], qrels, recall_run)
        results.update(recall_metrics)
        results["CumRelFound"] = cum_rel_found
        results["CumRecall"] = cum_recall

        output = RankingExperimentOutput(
            session_name=settings.name,
            model=model.name,
            ranker=tool.ranking_model,
            task=settings.task.name,
            dataset=settings.topicset.name,
            topic_id=state.topic.id,
            doc_ids=state.docids,
            start = settings.task.start_offset,
            size=settings.task.serp_size,
            performance=results
        )
        print(output.to_json(ensure_ascii=False))
        return state
    
class ClickStage:
    DEFAULT_INSTRUCTION = """
            Review the search topic, submitted query, and retrieved search results. Then, select a set of documents that are likely to contain relevant information to the search topic. Return an empty list if none of the results appears relevant.
        """

    def __init__(self, config: StageConfig):
        self.config = config

    def run(self, settings: ExperimentSettings, state: ExperimentState, llm_service: LLMServiceProtocol, model: ModelDescription, tool: ToolDescription, opensearch_client: OpenSearchClientProtocol, stage_name: str) -> ExperimentState:
        if not state.serp:
            state.error = "SERP not found, cannot run ClickStage."
            return state

        print("\n--- Running: Click Stage ---", file=sys.stderr)
        instruction_text = self.config.instruction or self.DEFAULT_INSTRUCTION
        click_instruction = ClickInstruction(
            instruction=instruction_text,
            serp=state.serp,
            exclude_docids=sorted(state.clicked_docids),
        )

        def valid_ranks(ranks: list[int], max_rank: int) -> list[int]:
            seen = set()
            cleaned = []
            for r in ranks:
                if isinstance(r, bool):
                    continue
                if isinstance(r, (int, float)) and int(r) == r:
                    r_int = int(r)
                    if 1 <= r_int <= max_rank and r_int not in seen:
                        seen.add(r_int)
                        cleaned.append(r_int)
            return cleaned

        before_clicked = set(state.clicked_docids)
        state.clicks = llm_service.create_clicks(model.name, model.temperature, state.memory, click_instruction)
        state.clicks.ranking_list = valid_ranks(state.clicks.ranking_list, len(state.serp.results))

        clicked_docids = []
        duplicate_docids = []
        for rank in state.clicks.ranking_list:
            docid = state.serp.results[rank - 1].docid
            clicked_docids.append(docid)
            if docid in before_clicked:
                duplicate_docids.append(docid)
            state.clicked_docids.add(docid)

        output = ClickExperimentOutput(
            session_name=settings.name,
            model=model.name,
            task=settings.task.name,
            dataset=settings.topicset.name,
            topic_id=state.topic.id,
            rankings=state.clicks.ranking_list,
            doc_ids=clicked_docids,
            duplicate_doc_ids=duplicate_docids if duplicate_docids else None,
        )
        print(output.to_json(ensure_ascii=False))

        return state

class RelevanceJudgementStage:
    DEFAULT_INSTRUCTION = """
            Evaluate the relevance of the document based on the topic description, submitted query, and the full text provided.
            Your response should include:
            - `label` (required): Indicate whether the document is `Relevant` or `NotRelevant`.
        """

    def __init__(self, config: StageConfig):
        self.config = config

    def run(self, settings: ExperimentSettings, state: ExperimentState, llm_service: LLMServiceProtocol, model: ModelDescription, tool: ToolDescription, opensearch_client: OpenSearchClientProtocol, stage_name: str) -> ExperimentState:
        if not state.clicks or not state.serp or not state.clicks.ranking_list:
            state.error = "Clicks/SERP not found or no documents clicked, cannot run RelevanceJudgementStage."
            return state

        dataset = ir_datasets.load(settings.topicset.name)
        if callable(dataset):
            dataset = dataset()

        qrels = Qrels()
        for row in dataset.qrels_iter():
            if row.query_id == state.topic.id:
                qrels.add(row.query_id, row.doc_id, row.relevance)
                
        print("\n--- Running: Relevance Judgement Stage ---", file=sys.stderr)
        for click_index in state.clicks.ranking_list:
            if click_index < 1 or click_index > len(state.serp.results):
                state.error = f"Invalid click index {click_index} for SERP results."
                return state
            click_docid = state.serp.results[click_index-1].docid

            fulltext_or_error = opensearch_client.fetch_fulltext(click_docid)
            if isinstance(fulltext_or_error, Error):
                state.error = f"Failed to fetch full text for {click_docid}: {fulltext_or_error.error_text}"
                return state

            state.fulltext = fulltext_or_error

            instruction_text = self.config.instruction or self.DEFAULT_INSTRUCTION
            rj_instruction = RelevanceJudgementInstruction(instruction=instruction_text, fulltext=state.fulltext)

            state.relevance_judgement = llm_service.calc_relevance_judgement(model.name, model.temperature, state.memory, rj_instruction)
            qrel_label = qrels.get(state.topic.id, click_docid, default=0)
            state.judged_docids.add(click_docid)
            label_value = getattr(state.relevance_judgement.label, "value", str(state.relevance_judgement.label))
            if qrel_label and label_value == "Relevant":
                state.judged_correct_relevant_docids.add(click_docid)

            output = RelevanceJudgementExperimentOutput(
                session_name = settings.name,
                model = model.name,
                task = settings.task.name,
                dataset = settings.topicset.name,
                topic_id = state.topic.id,
                docid = click_docid,
                label = f"{state.relevance_judgement.label}",
                qrel_label=qrel_label
            )
            print(output.to_json(ensure_ascii=False))

        return state


class QueryReFormulationStage:
    DEFAULT_INSTRUCTION = """
        Review the provided descriptions of task, corpus, tool, search topic, your previous query and search results. Re-Formulate a search query
    """

    def __init__(self, config: StageConfig):
        self.config = config

    def run(self, settings: ExperimentSettings, state: ExperimentState, llm_service: LLMServiceProtocol, model: ModelDescription, tool: ToolDescription, opensearch_client: OpenSearchClientProtocol, stage_name: str) -> ExperimentState:
        if not state.serp or not state.query:
            state.error = "SERP or original query not found, cannot run QueryReFormulationStage."
            return state

        print("\n--- Running: Query Re-formulation Stage ---", file=sys.stderr)
        instruction_text = self.config.instruction or self.DEFAULT_INSTRUCTION
        qrf_instruction = QueryReFormulationInstruction(instruction=instruction_text)

        state.query = llm_service.recreate_query(model.name, model.temperature, state.memory, qrf_instruction)

        output = QueryReformulationExperimentOutput(
            session_name = settings.name,
            model = model.name,
            task = settings.task.name,
            dataset = settings.topicset.name,
            topic_id = state.topic.id,
            query = state.query.query,
            start = settings.task.start_offset,
            size=settings.task.serp_size
        )
        print(output.to_json(ensure_ascii=False))

        return state

class NextActionStage:
    DEFAULT_INSTRUCTION = """
        Based on the task description, topic details, and your progress so far, decide on the next action to take toward completing the task.

        Choose one of the following actions and provide a brief explanation for your choice:
        - **Submit_New_Query**: Submit a new search query.
        - **Click_Document**: Select a set of documents from the current search results.
        - **GO_NEXT_RESULT_PAGE**: Navigate to the next page of search results for the current query.
        - **End_Task**: End the current task.
    """

    def __init__(self, config: StageConfig):
        self.config = config

    def run(self, settings: ExperimentSettings, state: ExperimentState, llm_service: LLMServiceProtocol, model: ModelDescription, tool: ToolDescription, opensearch_client: OpenSearchClientProtocol, stage_name: str) -> ExperimentState:
        print("\n--- Running: Next Action Stage ---", file=sys.stderr)
        instruction_text = self.config.instruction or self.DEFAULT_INSTRUCTION
        next_action_instruction = NextActionInstruction(instruction=instruction_text, task=settings.task)

        state.next_action = llm_service.decide_next_action(model.name, model.temperature, state.memory, next_action_instruction)

        action_name = None
        if state.next_action and state.next_action.action:
            action_name = state.next_action.action.name

        output = NextActionOutput(
            session_name = settings.name,
            model = model.name,
            task = settings.task.name,
            dataset = settings.topicset.name,
            topic_id = state.topic.id,
            action = action_name,
            action_num = state.action_num,
            reason = state.next_action.reason
        )
        print(output.to_json(ensure_ascii=False))

        return state

class ExperimentRunner:
    def __init__(self, settings: ExperimentSettings):
        self.settings = settings

        self.stage_runners: Dict[str, ExperimentStage] = {
            "query": QueryFormulationStage(self.settings.stages.get("query", StageConfig())),
            "ranking": RankingStage(self.settings.stages.get("ranking", StageConfig())),
            "click": ClickStage(self.settings.stages.get("click", StageConfig())),
            "relevance": RelevanceJudgementStage(self.settings.stages.get("relevance", StageConfig())),
            "reformulate": QueryReFormulationStage(self.settings.stages.get("reformulate", StageConfig())),
            "next_action": NextActionStage(self.settings.stages.get("next_action", StageConfig())),
        }

        self.action_stage_map = {
            Action.SUBMIT_NEW_QUERY: ["query", "ranking", "click", "relevance", "next_action"],
            Action.CLICK_DOCUMENT: ["click", "relevance", "next_action"],
            Action.GO_NEXT_RESULT_PAGE: ["ranking", "click", "relevance", "next_action"],
        }

        self.topic_list_map: Dict[Type[BaseTopic], Type[TopicList]] = {
            TitleOnlyTopic: TopicList[TitleOnlyTopic],
            TitleDescriptionTopic: TopicList[TitleDescriptionTopic],
            TitleNarrativeTopic: TopicList[TitleNarrativeTopic],
            TitleDescriptionNarrativeTopic: TopicList[TitleDescriptionNarrativeTopic],
            FullTopic: TopicList[FullTopic],  # Optional, same as above
        }

        self.llm_factory = LLMServiceFactory()
        self.opensearch_client_factory = OpenSearchClientFactory()
        self.topics = self._load_topics()
        self._filter_topics_by_ids()
        self._filter_topics_by_min_rels()
        self._apply_max_topics()

    def _filter_topics_by_min_rels(self) -> None:
        if not self.settings.min_relevant_docs:
            return
        dataset = ir_datasets.load(self.settings.topicset.name)
        rel_counts: Dict[str, int] = defaultdict(int)
        for qrel in dataset.qrels_iter():
            if getattr(qrel, "relevance", 0) and qrel.relevance > 0:
                rel_counts[qrel.query_id] += 1
        min_rels = self.settings.min_relevant_docs
        self.topics = [topic for topic in self.topics if rel_counts.get(topic.id, 0) >= min_rels]
        if not self.topics:
            print(f"[WARNING] No topics meet min_relevant_docs >= {min_rels}.", file=sys.stderr)

    def _filter_topics_by_ids(self) -> None:
        if not self.settings.topic_ids:
            return
        wanted = set(self.settings.topic_ids)
        list_cls = type(self.topics)
        filtered = list_cls()
        for topic in self.topics:
            if topic.id in wanted:
                filtered.append(topic)
        self.topics = filtered
        missing = wanted - {topic.id for topic in self.topics}
        if missing:
            print(f"[WARNING] Topic IDs not found: {sorted(missing)}", file=sys.stderr)

    def _load_topics(self) -> TopicList:
        dataset = ir_datasets.load(self.settings.topicset.name)
        if callable(dataset):
            dataset = dataset()

        topic_cls: Type[BaseTopic] = self.settings.topicset.topic_class
        list_cls = self.topic_list_map[topic_cls]

        topics = list_cls()

        query_iter = dataset.queries_iter()
        if self.settings.max_topics and not self.settings.min_relevant_docs:
            query_iter = islice(query_iter, self.settings.max_topics)

        for raw in query_iter:
            topic = topic_cls.from_ir_datasets(raw)
            topics.append(topic)

        return topics

    def _apply_max_topics(self) -> None:
        if not self.settings.max_topics:
            return
        list_cls = type(self.topics)
        limited = list_cls()
        for topic in islice(self.topics, self.settings.max_topics):
            limited.append(topic)
        self.topics = limited
    
    def _get_topic_mapper(self):
        topic_cls: Type[BaseTopic] = self.settings.topicset.topic_class
        name = self.settings.topicset.name.lower()

        if "text" in dir(next(ir_datasets.load(name).queries_iter())):
            # Format: query_id, text
            return lambda q: topic_cls(id=q.query_id, title=q.text)

        # Format: query_id, title, description, narrative
        return lambda q: topic_cls(
            id=q.query_id,
            title=q.title,
            description=getattr(q, "description", None),
            narrative=getattr(q, "narrative", None)
        )

    def run(self):
        print(f"\n{'='*20} Experimental Setting: {self.settings.name} {'='*20}", file=sys.stderr)
        for model in self.settings.models:
            print(f"\n{'='*20} Model: {model.name} ({model.type}) {'='*20}", file=sys.stderr)

            for tool in self.settings.tools:
                print(f"\n{'='*20} Ranker: {tool.ranking_model} ({tool.name}) {'='*20}", file=sys.stderr)
                opensearch_client = self.opensearch_client_factory.create_opensearch_client(settings=self.settings, tool=tool)

                for topic in self.topics:
                    llm_service = self.llm_factory.create_llm_service(model.type)
                    print(f"\n{'--'*10} Topic: {topic.id} ({topic.title}) {'--'*10}", file=sys.stderr)

                    memory = ConversationHistory(system_role=model.system_role, system_prompt=model.system_prompt)
                    state = ExperimentState(topic=topic, memory=memory)

                    state.next_action = NextAction(action=Action.SUBMIT_NEW_QUERY, reason="initial bootstrap")

                    while state.action_num < self.settings.max_actions:
                        stage_names = []

                        if state.next_action and state.next_action.action != Action.END_TASK:
                            action_enum = getattr(state.next_action, "action", None)
                            stage_names = self.action_stage_map.get(action_enum, [])

                        for stage_name in stage_names:
                            if stage_name not in self.stage_runners:
                                print(f"[ERROR] Unknown stage: {stage_name}", file=sys.stderr)
                                sys.exit(1)

                            if stage_name == "ranking" and action_enum == Action.GO_NEXT_RESULT_PAGE:
                                state.query.start += 10

                            stage_runner = self.stage_runners[stage_name]
                            state = stage_runner.run(self.settings, state, llm_service, model, tool, opensearch_client, stage_name)
                            
                            if state.error:
                                print(f"[WARNING] in stage '{stage_name}': {state.error}. Stopping pipeline for this topic.", file=sys.stderr)
                                
                                if self.settings.full_log:
                                    print(f"\n{'--'*10} Full Log {'--'*10}", file=sys.stderr)
                                    all_messages = state.memory.get_all_messages()
                                    pprint.pprint(all_messages, stream=sys.stderr)

                                state.error = None
                                return
                            
                            if stage_name == "next_action" and state.next_action.action == Action.END_TASK:
                                print(f"\n{'='*20} Agent decided to end the task. {'='*20}", file=sys.stderr)

                                if self.settings.full_log:
                                    print(f"\n{'--'*10} Full Log {'--'*10}", file=sys.stderr)
                                    all_messages = state.memory.get_all_messages()
                                    pprint.pprint(all_messages, stream=sys.stderr)
                                
                                return

                            state.action_num += 1

                    if self.settings.full_log:
                        print(f"\n{'--'*10} Full Log {'--'*10}", file=sys.stderr)
                        all_messages = state.memory.get_all_messages()
                        pprint.pprint(all_messages, stream=sys.stderr)

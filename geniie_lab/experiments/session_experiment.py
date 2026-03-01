import sys
import json
import pprint
import time
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
)
from geniie_lab.dataclasses.output import (
    ClickExperimentOutput,
    QueryExperimentOutput,
    QueryReformulationExperimentOutput,
    RankingExperimentOutput,
    RelevanceJudgementExperimentOutput,
)
from geniie_lab.memory import ConversationHistory
from geniie_lab.experiments.prompt_utils import render_instruction
from geniie_lab.response import Clicks
from geniie_lab.services.llm.llm_service_factory import LLMServiceFactory
from geniie_lab.services.llm.llm_service_protocol import LLMServiceProtocol
from geniie_lab.services.measure_service import MeasureService, Qrels, Run
from geniie_lab.services.opensearch.opensearch_client_factory import OpenSearchClientFactory
from geniie_lab.services.opensearch.opensearch_client_protocol import OpenSearchClientProtocol

class ExperimentStage(Protocol):
    def __init__(self, config: StageConfig): ...
    def run(self, settings: ExperimentSettings, state: ExperimentState, llm_service: LLMServiceProtocol, model: ModelDescription, tool: ToolDescription, opensearch_client: OpenSearchClientProtocol) -> ExperimentState: ...


class QueryFormulationStage:
    DEFAULT_INSTRUCTION = """
        Review the provided descriptions of task, corpus, tool and search topic. Formulate a search query.
    """

    def __init__(self, config: StageConfig):
        self.config = config

    def run(self, settings: ExperimentSettings, state: ExperimentState, llm_service: LLMServiceProtocol, model: ModelDescription, tool: ToolDescription, opensearch_client: OpenSearchClientProtocol) -> ExperimentState:
        print("\n--- Running: Query Formulation Stage ---", file=sys.stderr)
        instruction_text = self.config.instruction or self.DEFAULT_INSTRUCTION
        instruction_text = render_instruction(instruction_text, state)
        qf_instruction = QueryFormulationInstruction(instruction=instruction_text, task=settings.task, corpus=settings.corpus, tool=tool, topic=state.topic)

        state.query = llm_service.create_query(model.name, model.temperature, state.memory, qf_instruction, model.reasoning_mode)

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
    
    @staticmethod
    def _set_metrics_context(settings: ExperimentSettings, state: ExperimentState, results: Dict[str, float]) -> None:
        if not settings.memory_metrics:
            return
        parts = []
        for key in settings.memory_metrics:
            if key not in results:
                continue
            value = results[key]
            if isinstance(value, float):
                parts.append(f"{key}: {value:.4f}")
            else:
                parts.append(f"{key}: {value}")
        if not parts:
            return
        state.metrics_context = "Latest retrieval metrics:\n" + "\n".join(parts)

    def run(self, settings: ExperimentSettings, state: ExperimentState, llm_service: LLMServiceProtocol, model: ModelDescription, tool: ToolDescription, opensearch_client: OpenSearchClientProtocol) -> ExperimentState:
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

        # Additional recall@100 measurement using top-100 (no extra LLM tokens).
        recall_run = Run()
        top100_docids = opensearch_client.search_docids(query_text, start=0, size=100)
        for idx, docid in enumerate(top100_docids, start=1):
            recall_run.add(state.topic.id, docid, idx)
        recall_metrics = MeasureService().calc([ir_measures.Recall@100], qrels, recall_run)
        results.update(recall_metrics)

        for docid in top100_docids:
            qrel_label = qrels.get(state.topic.id, docid, default=0)
            if qrel_label and qrel_label > 0:
                state.retrieved_relevant_docids_top100.add(docid)
        cum_recall_100 = (len(state.retrieved_relevant_docids_top100) / total_rels) if total_rels else 0.0
        results["CumRelFound@100"] = len(state.retrieved_relevant_docids_top100)
        results["CumRecall@100"] = cum_recall_100
        results["CumRelFound"] = cum_rel_found
        results["CumRecall"] = cum_recall

        self._set_metrics_context(settings, state, results)

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

    def run(self, settings: ExperimentSettings, state: ExperimentState, llm_service: LLMServiceProtocol, model: ModelDescription, tool: ToolDescription, opensearch_client: OpenSearchClientProtocol) -> ExperimentState:
        if not state.serp:
            state.error = "SERP not found, cannot run ClickStage."
            return state

        print("\n--- Running: Click Stage ---", file=sys.stderr)
        instruction_text = self.config.instruction or self.DEFAULT_INSTRUCTION
        instruction_text = render_instruction(instruction_text, state)
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
        state.clicks = llm_service.create_clicks(model.name, model.temperature, state.memory, click_instruction, model.reasoning_mode)
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

    def run(self, settings: ExperimentSettings, state: ExperimentState, llm_service: LLMServiceProtocol, model: ModelDescription, tool: ToolDescription, opensearch_client: OpenSearchClientProtocol) -> ExperimentState:
        if not state.clicks or not state.serp:
            state.error = "Clicks/SERP not found, cannot run RelevanceJudgementStage."
            return state
        if not state.clicks.ranking_list:
            print("[INFO] No clicked documents; skipping Relevance Judgement Stage.", file=sys.stderr)
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
            instruction_text = render_instruction(instruction_text, state)
            rj_instruction = RelevanceJudgementInstruction(instruction=instruction_text, fulltext=state.fulltext)

            state.relevance_judgement = llm_service.calc_relevance_judgement(model.name, model.temperature, state.memory, rj_instruction, model.reasoning_mode)

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

    def run(self, settings: ExperimentSettings, state: ExperimentState, llm_service: LLMServiceProtocol, model: ModelDescription, tool: ToolDescription, opensearch_client: OpenSearchClientProtocol) -> ExperimentState:
        if not state.serp or not state.query:
            state.error = "SERP or original query not found, cannot run QueryReFormulationStage."
            return state

        print("\n--- Running: Query Re-formulation Stage ---", file=sys.stderr)
        instruction_text = self.config.instruction or self.DEFAULT_INSTRUCTION
        instruction_text = render_instruction(instruction_text, state)
        qrf_instruction = QueryReFormulationInstruction(
            instruction=instruction_text,
            task=settings.task,
            corpus=settings.corpus,
            tool=tool,
            topic=state.topic,
        )

        state.query = llm_service.recreate_query(model.name, model.temperature, state.memory, qrf_instruction, model.reasoning_mode)

        output = QueryReformulationExperimentOutput(
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
    
class ExperimentRunner:
    def __init__(self, settings: ExperimentSettings):
        self.settings = settings

        if not self.settings.plan:
            print("[ERROR] No plan provided in settings. Please specify the order of stages to run.", file=sys.stderr)
            sys.exit(1)
        
        self.stage_runners: Dict[str, ExperimentStage] = {
            "query": QueryFormulationStage(self.settings.stages.get("query", StageConfig())),
            "ranking": RankingStage(self.settings.stages.get("ranking", StageConfig())),
            "click": ClickStage(self.settings.stages.get("click", StageConfig())),
            "relevance": RelevanceJudgementStage(self.settings.stages.get("relevance", StageConfig())),
            "reformulate": QueryReFormulationStage(self.settings.stages.get("reformulate", StageConfig())),
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

    @staticmethod
    def _increment_counter(counter: Dict[str, int], key: str) -> None:
        counter[key] = counter.get(key, 0) + 1

    @staticmethod
    def _cleanup_orphan_stage_prompt(state: ExperimentState, stage_name: str) -> None:
        # If a stage call fails after adding its user prompt, remove that dangling prompt
        # so fallback/default behavior is not learned from malformed turns.
        messages = state.memory.get_all_messages()
        if len(messages) <= 1:
            return
        last = messages[-1]
        if last.get("role") == "user" and last.get("stage") == stage_name:
            state.memory.remove_last_message()

    def _emit_stage_failure_event(
        self,
        *,
        stage_name: str,
        topic_id: str,
        model_name: str,
        tool_name: str,
        attempt: int,
        retries: int,
        exc: Exception,
        fallback_applied: bool,
    ) -> None:
        payload = {
            "event": "stage_failure",
            "stage": stage_name,
            "topic_id": topic_id,
            "model": model_name,
            "tool": tool_name,
            "attempt": attempt,
            "max_attempts": retries + 1,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "fallback_applied": fallback_applied,
        }
        print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)

    @staticmethod
    def _is_transient_provider_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return "rate limit" in text or "timeout" in text or "overloaded" in text

    @staticmethod
    def _is_retryable_stage(stage_name: str) -> bool:
        # Allow retries for all stages, including query/reformulate.
        return True

    def _apply_resilient_fallback(
        self,
        *,
        stage_name: str,
        state: ExperimentState,
        settings: ExperimentSettings,
        model: ModelDescription,
    ) -> bool:
        if stage_name == "click":
            state.clicks = Clicks(ranking_list=[], reason="stage_failed")
            output = ClickExperimentOutput(
                session_name=settings.name,
                model=model.name,
                task=settings.task.name,
                dataset=settings.topicset.name,
                topic_id=state.topic.id,
                rankings=[],
                doc_ids=[],
                duplicate_doc_ids=None,
            )
            print(output.to_json(ensure_ascii=False))
            return True

        if stage_name == "relevance":
            # Skip relevance for this step and continue pipeline.
            return True

        if stage_name == "reformulate" and state.query:
            # Keep current query unchanged and continue.
            output = QueryReformulationExperimentOutput(
                session_name=settings.name,
                model=model.name,
                task=settings.task.name,
                dataset=settings.topicset.name,
                topic_id=state.topic.id,
                query=state.query.query,
                start=settings.task.start_offset,
                size=settings.task.serp_size,
            )
            print(output.to_json(ensure_ascii=False))
            return True

        return False

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
        memory_policy = self.settings.memory_policy or "full"
        print(f"Memory policy: {memory_policy}", file=sys.stderr)
        for model in self.settings.models:
            print(f"\n{'='*20} Model: {model.name} ({model.type}) {'='*20}", file=sys.stderr)

            for tool in self.settings.tools:
                print(f"\n{'='*20} Ranker: {tool.ranking_model} ({tool.name}) {'='*20}", file=sys.stderr)
                opensearch_client = self.opensearch_client_factory.create_opensearch_client(settings=self.settings, tool=tool)

                for topic in self.topics:
                    llm_service = self.llm_factory.create_llm_service(model.type, log_llm_io=self.settings.log_llm_io)
                    print(f"\n{'--'*10} Topic: {topic.id} ({topic.title}) {'--'*10}", file=sys.stderr)

                    memory = ConversationHistory(
                        system_role=model.system_role,
                        system_prompt=model.system_prompt,
                        memory_policy=self.settings.memory_policy,
                    )
                    state = ExperimentState(topic=topic, memory=memory)

                    try:
                        stop_topic = False
                        for stage_name in self.settings.plan:
                            stage_runner = self.stage_runners[stage_name]
                            retries = self.settings.stage_failure_retries if self.settings.failure_policy == "resilient" else 0
                            attempt = 0
                            while True:
                                attempt += 1
                                try:
                                    state = stage_runner.run(self.settings, state, llm_service, model, tool, opensearch_client)
                                    break
                                except Exception as exc:
                                    self._increment_counter(state.stage_failures, stage_name)
                                    self._cleanup_orphan_stage_prompt(state, stage_name)
                                    should_retry = (
                                        attempt <= retries
                                        and not self._is_transient_provider_error(exc)
                                        and self._is_retryable_stage(stage_name)
                                    )
                                    if should_retry:
                                        self._emit_stage_failure_event(
                                            stage_name=stage_name,
                                            topic_id=state.topic.id,
                                            model_name=model.name,
                                            tool_name=tool.name,
                                            attempt=attempt,
                                            retries=retries,
                                            exc=exc,
                                            fallback_applied=False,
                                        )
                                        continue

                                    fallback_applied = False
                                    if self.settings.failure_policy == "resilient":
                                        fallback_applied = self._apply_resilient_fallback(
                                            stage_name=stage_name,
                                            state=state,
                                            settings=self.settings,
                                            model=model,
                                        )
                                        if fallback_applied:
                                            self._increment_counter(state.stage_fallbacks, stage_name)
                                    self._emit_stage_failure_event(
                                        stage_name=stage_name,
                                        topic_id=state.topic.id,
                                        model_name=model.name,
                                        tool_name=tool.name,
                                        attempt=attempt,
                                        retries=retries,
                                        exc=exc,
                                        fallback_applied=fallback_applied,
                                    )
                                    if not fallback_applied:
                                        state.error = f"{type(exc).__name__}: {exc}"
                                        stop_topic = True
                                    break

                            if state.error:
                                print(f"[WARNING] in stage '{stage_name}': {state.error}. Stopping pipeline for this topic.", file=sys.stderr)
                                state.error = None
                                stop_topic = True
                            if stop_topic:
                                break
                    finally:
                        if self.settings.full_log and state and state.memory:
                            print(f"\n{'--'*10} Full Log {'--'*10}", file=sys.stderr)
                            all_messages = state.memory.get_all_messages()
                            pprint.pprint(all_messages, stream=sys.stderr)
                    if self.settings.topic_sleep_seconds > 0:
                        time.sleep(self.settings.topic_sleep_seconds)

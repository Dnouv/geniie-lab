# Third-party libraries
from dotenv import load_dotenv
import ir_measures
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Local application imports
from geniie_lab.dataclasses.description import (
    CorpusDescription,
    ModelDescription,
    TaskDescription,
    ToolDescription,
    TopicDescription,
)
from geniie_lab.dataclasses.topic import TitleOnlyTopic
from geniie_lab.dataclasses.setting import ExperimentSettings, StageConfig
from geniie_lab.experiments.session_experiment import ExperimentRunner

load_dotenv()

my_settings = ExperimentSettings(
    name="touche2022_session_splade",
    task=TaskDescription(
        name="High-Recall Retrieval",
        description="Find as many different relevant documents as possible for a given search topic from a given document collection using a provided search tool.",
        measurement=[ir_measures.nDCG@10, ir_measures.MRR@10],
        start_offset=0,
        serp_size=20,
    ),
    topicset=TopicDescription(
        name="argsme/2020-04-01/processed/touche-2022-task-1",
        type="ir_datasets",
        topic_class=TitleOnlyTopic
    ),
    corpus=CorpusDescription(
        name="Touché 2022 Task 1 (ArgsMe processed)",
        description="Argument retrieval benchmark (Touché 2022 Task 1) on ArgsMe processed corpus.",
        index_name="touche_2022_splade",
    ),
    models=[
        ModelDescription(
            type="openai",
            name="openai/gpt-oss-120b",
            system_prompt="You're a helpful assistant",
            temperature=0.0,
        ),
    ],
    tools=[
        ToolDescription(
            name="opensearch",
            ranking_model="splade",
            encode_model="naver/splade-cocondenser-ensembledistil",
            index_name="touche_2022_splade",
            port=9200,
            description="Touché 2022 (ArgsMe processed) corpus searchable with SPLADE term expansion ranking.",
        ),
    ],
    stages={
        "query": StageConfig(
            instruction="""
                Review the provided descriptions of task, corpus, tool and search topic. Then, formulate a search query.
            """,
        ),
        "ranking": StageConfig(
            instruction=""
        ),
        "click": StageConfig(
            instruction="""
                You are given a SERP with results numbered 1–N.
                Select up to 10 NEW documents that look relevant to the topic (previously clicked docids are shown; do NOT select them again).
                Return an empty list if none of the results appears relevant.
            """,
        ),
        "relevance": StageConfig(
            instruction="""
                Evaluate the relevance of the document based on the search topic description and its narrative.
            """,
        ),
        "reformulate": StageConfig(
            instruction="""
                Formulate another search query to find new relevant documents.
            """,
        ),
    },
    plan=["query", "ranking"] + (["click", "relevance", "reformulate", "ranking"] * 10),
    max_topics=3,
    memory_policy="forget_queries_keep_reason",
    memory_metrics=["RR@10", "nDCG@10", "R@100", "CumRecall", "CumRecall@100"],
    full_log=True,
    # Keep raw LLM I/O logs and use resilient stage handling:
    # retry up to 2 extra times per stage (3 total attempts), then fallback where supported.
    log_llm_io=True,
    failure_policy="resilient",
    stage_failure_retries=2,
)

if __name__ == "__main__":
    runner = ExperimentRunner(settings=my_settings)
    runner.run()

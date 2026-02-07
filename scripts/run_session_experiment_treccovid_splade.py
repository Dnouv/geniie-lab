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
    name="treccovid_session_splade",
    task=TaskDescription(
        name="High-Recall Retrieval",
        description="Find as many different relevant documents as possible for a given search topic from a given document collection using a provided search tool.",
        measurement=[ir_measures.nDCG@10, ir_measures.MRR@10],
        start_offset=0,
        serp_size=20,
    ),
    topicset=TopicDescription(
        name="beir/trec-covid",
        type="ir_datasets",
        topic_class=TitleOnlyTopic
    ),
    corpus=CorpusDescription(
        name="TREC-COVID",
        description=(
            "The corpus consists of English-language biomedical research articles and preprints "
            "on COVID-19 and related coronaviruses, including both studies published up to mid-2020 "
            "and earlier historical research on viruses such as SARS and MERS. Each document is "
            "represented by its title and abstract."
        ),
        index_name="trec_covid_splade",
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
            index_name="trec_covid_splade",
            port=9200,
            description="TREC-COVID corpus searchable with SPLADE term expansion ranking.",
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
    plan=["query", "ranking"] + (["click", "relevance", "reformulate", "ranking"] * 19),
    max_topics=3,
    memory_policy="forget_queries_keep_reason",
    memory_metrics=["RR@10", "nDCG@10", "R@100", "CumRecall", "CumRecall@100"],
    full_log=True
)

if __name__ == "__main__":
    runner = ExperimentRunner(settings=my_settings)
    runner.run()

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
from geniie_lab.dataclasses.topic import (
    TitleDescriptionNarrativeTopic, FullTopic,
    TitleDescriptionTopic,
    TitleNarrativeTopic,
    TitleOnlyTopic
)
from geniie_lab.dataclasses.setting import ExperimentSettings, StageConfig
from geniie_lab.experiments.session_experiment import ExperimentRunner

load_dotenv()

my_settings = ExperimentSettings(
    name="my_session_experiment_splade",
    task=TaskDescription(
        name="High-Recall Retrieval",
        description="Find as many different relevant documents as possible for a given search topic from a given document collection using a provided search tool.",
        measurement=[ir_measures.nDCG@10, ir_measures.MRR@10],
        start_offset=0,
        serp_size=20,
    ),
    topicset=TopicDescription(
        name="beir/dbpedia-entity/test",
        type="ir_datasets",
        topic_class=TitleOnlyTopic
    ),
    corpus=CorpusDescription(
        name="DBPedia Entity",
        description=(
            "An entity-centric document collection derived from DBpedia, where each document "
            "represents a Wikipedia/DBpedia entity with a short descriptive abstract. "
            "It supports entity-focused search over people, places, organizations, and concepts."
        ),
        index_name="dbpedia_entity_splade",
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
            index_name="dbpedia_entity_splade",
            port=9200,
            device="mps",
            description="BEIR DBPedia Entity corpus searchable with SPLADE term expansion ranking.",
        ),
    ],
    stages={
        "query": StageConfig(
            instruction="""
                Review the provided descriptions of task, corpus, tool and search topic. Then, formulate a search query.
                Respond ONLY with a JSON object in this exact format (no markdown, no prose, no code fences):
                {
                "query": "your query",
                "reason": "short sentence"
                }
            """,
        ),
        "ranking": StageConfig(
            instruction=""
        ),
        "click": StageConfig(
            instruction="""
                You are given a SERP with results numbered 1–N.
                Select up to 10 NEW documents that look relevant to the topic (previously clicked docids are shown; do NOT select them again).

                Respond ONLY with a single JSON object in the exact multiline format below:

                {
                "ranking_list": [
                    1
                    3
                    8
                ],
                "reason": "short sentence"
                }

                Rules:
                - Each rank MUST appear on its own line inside the array. Make sure to not miss any line breaks.
                - No combined numbers (e.g., 128).
                - Every rank must be within 1..N (only ranks shown in the SERP).
                - Do not duplicate ranks.
                - Do not output any keys other than `ranking_list` and `reason`.
                - If none are relevant, output:

                {
                "ranking_list": [

                ],
                "reason": "None look relevant"
                }
            """,
        ),
        "relevance": StageConfig(
            instruction="""
                Evaluate the relevance of the document based on the search topic description and its narrative.
                Return ONLY this JSON:
                {
                "label": "Relevant" or "NotRelevant",
                "reason": "short sentence"
                }
                Do not output ranking_list or any other keys.
            """,
        ),
        "reformulate": StageConfig(
            instruction="""
                Formulate another search query to find new relevant documents. Make sure to consider previous relevant documents found our goal here
                is to maximize the recall rate. The search index in use is SPLADE, so try to include informative terms and entity variants that can
                surface new relevant documents beyond what has already been found.
                Your output MUST be ONLY a JSON object in this exact format (no markdown, no prose, no code fences):
                {
                "query": "your query",
                "reason": "short sentence"
                }
            """,
        ),
    },
    plan=["query", "ranking"] + (["click", "relevance", "reformulate", "ranking"] * 19),
    max_topics=5,
    topic_ids=["INEX_LD‑2010069", "INEX_LD‑20120511", "INEX_XER‑86", "INEX_XER-140", "INEX_XER-144"],
    # min_relevant_docs=100,
    memory_policy="forget_queries_half",
    memory_metrics=["RR@10", "nDCG@10", "R@100", "CumRecall", "CumRecall@100"],
    full_log=True
)

if __name__ == "__main__":
    runner = ExperimentRunner(settings=my_settings)
    runner.run()

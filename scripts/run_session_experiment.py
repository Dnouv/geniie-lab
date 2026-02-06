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
    name="my_session_experiment",
    task=TaskDescription(
        name="High-Recall Retrieval",
        description="Find as many different relevant documents as possible for a given search topic from a given document collection using a provided search tool.",
        measurement=[ir_measures.nDCG@10, ir_measures.MRR@10],
        start_offset=0,
        serp_size=20,
        # name = "High-Precision Retrieval",
        # description = "Find the most relevant documents at the top rank for a given search topic from a given document collection using a provided search tool.",
        # measurement=[ir_measures.nDCG@10, ir_measures.MRR@10],
        # start_offset=0,
        # serp_size=10,
        # name = "High-Diversity Retrieval",
        # description = "Find a diverse set of relevant documents for a given search topic from a given document collection using a provided search tool.",
        # measurement=[ir_measures.alpha_nDCG@20],
        # start_offset=0,
        # serp_size=20,
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
        index_name="dbpedia_entity_bm25",
    ),
    models=[
        ModelDescription(
            type="openai",
            name="openai/gpt-oss-120b",
            system_prompt="You're a helpful assistant",
            temperature=0.0,
        ),
        # ModelDescription(
        #     type="azure",
        #     name="gpt-4.1-mini",
        #     system_prompt="You're a helpful assistant",
        #     temperature=0.0,
        # ),
        # ModelDescription(
        #     type="gemini",
        #     name="gemini-2.0-flash-lite-001",
        #     system_prompt="You're a helpful assistant",
        #     temperature=0.0,
        # ),
        # ModelDescription(
        #     type="ollama",
        #     name="qwen2.5:72b-instruct-q4_K_M",
        #     system_prompt="You're a helpful assistant",
        #     temperature=0.0,
        # ),
        # ModelDescription(
        #     type="ollama",
        #     name="llama3.3:70b-instruct-q4_K_M",
        #     system_prompt="You're a helpful assistant",
        #     temperature=0.0,
        # ),
        # ModelDescription(
        #     type="openrouter",
        #     name="openai/gpt-4.1-mini",
        #     system_prompt="You're a helpful assistant",
        #     temperature=0.0,
        # )
    ],
    tools=[
        ToolDescription(
            name="opensearch",
            ranking_model="bm25",
            index_name="dbpedia_entity_bm25",
            port=9200,
            description="It allows you to perform searches using keywords only and employs the BM25 ranking model to order results.",
        ),
        # ToolDescription(
        #     name="opensearch",
        #     ranking_model="bm25_rm3",
        #     index_name="dbpedia_entity_bm25",
        #     port=9200,
        #     prf_docs=10,
        #     prf_terms=20,
        #     description="BM25 with RM3-style pseudo relevance feedback.",
        # ),
        # ToolDescription(
        #     name="opensearch",
        #     ranking_model="bm25_dense",
        #     encode_model="sentence-transformers/all-MiniLM-L6-v2",
        #     index_name="dbpedia_entity_bm25",
        #     port=9200,
        #     rerank_top_k=100,
        #     description="BM25 candidate generation with dense bi-encoder reranking.",
        # ),
        # ToolDescription(
        #     name="opensearch",
        #     ranking_model="bm25_cross_encoder",
        #     encode_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        #     index_name="dbpedia_entity_bm25",
        #     port=9200,
        #     rerank_top_k=100,
        #     description="BM25 candidate generation with cross-encoder reranking.",
        # ),
        # ToolDescription(
        #     name="opensearch",
        #     ranking_model="splade",
        #     encode_model="naver/splade-cocondenser-ensembledistil",
        #     index_name="dbpedia_entity_splade",
        #     port=9200,
        #     description="BEIR DBPedia Entity corpus searchable with SPLADE term expansion ranking.",
        # ),
        # ToolDescription(
        #     name="opensearch",
        #     ranking_model="dpr",
        #     encode_model="sentence-transformers/msmarco-distilbert-base-tas-b",
        #     index_name="dbpedia_entity_dpr",
        #     port=9200,
        #     description="BEIR DBPedia Entity corpus searchable with DPR dense retrieval ranking.",
        # ),
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
                is to maximize the recall rate, the search index in use is BM25 so try to come up with strategies that can help retrieve more relevant documents.
                Your output MUST be ONLY a JSON object in this exact format (no markdown, no prose, no code fences):
                {
                "query": "your query",
                "reason": "short sentence"
                }
            """,
        ),
    },
    plan=["query", "ranking"] + (["click", "relevance", "reformulate", "ranking"] * 19),
    max_topics=3,
    topic_ids=["INEX_LD-2009039", "INEX_LD-2009063", "INEX_LD-20120411"],
    min_relevant_docs=100,
    memory_policy="full",
    memory_metrics=["RR@10", "nDCG@10", "R@100", "CumRecall", "CumRecall@100"],
    full_log=True
)

if __name__ == "__main__":
    runner = ExperimentRunner(settings=my_settings)
    runner.run()

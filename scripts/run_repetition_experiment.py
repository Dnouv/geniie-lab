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
from geniie_lab.experiments.repetition_experiment import ExperimentRunner

load_dotenv()

my_settings = ExperimentSettings(
    name="my_repetition_experiment",
    task=TaskDescription(
        name="High-Recall Retrieval",
        description="Find as many different relevant documents as possible for a given search topic from a given document collection using a provided search tool.",
        measurement=[ir_measures.nDCG@10, ir_measures.MRR@10],
        start_offset=0,
        serp_size=10,
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
        name="beir/dbpedia-entity/dev",
        type="ir_datasets",
        topic_class=TitleOnlyTopic
    ),
    corpus=CorpusDescription(
        name="DBPedia Entity",
        description="A BEIR benchmark corpus built from DBPedia passages describing Wikipedia entities.",
        index_name="dbpedia_entity_bm25",
    ),
    models=[
        ModelDescription(
            type="openai",
            name="gpt-4.1-mini-2025-04-14",
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
            description="BEIR DBPedia Entity corpus searchable with keyword-only BM25 ranking.",
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
            """,
        ),
        "ranking": StageConfig(
            instruction=""
        ),
        "click": StageConfig(
            instruction="""
                Select a set of documents that are likely to contain relevant information to the search topic. Return an empty list if none of the results appears relevant.
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
    max_topics=1,
    #
    # The last stage in the plan will be repeated for loop_num_per_topic times.
    # All the previous items will be executed only once.
    #
    # Repeat the initial query formulation stage
    plan=["query"],
    #
    # Repeat the click stage on the SERP retrieved by the initial query
    # plan=["query", "ranking", "click"],
    #
    # Repeat the relevance judgement stage on the clicked documents
    # plan=["query", "ranking", "click", "relevance"],
    #
    # Repeat the query reformulation stage
    # plan=["query", "ranking", "click", "relevance", "reformulate"],
    #
    loop_num_per_topic=5,
    #
    # Please note that full_log keeps the last iteration of the last stage.
    #
    full_log=False
)

if __name__ == "__main__":
    runner = ExperimentRunner(settings=my_settings)
    runner.run()

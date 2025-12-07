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
            description="BEIR DBPedia Entity corpus searchable with keyword-only BM25 ranking.",
        ),
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
            """,
        ),
        "ranking": StageConfig(
            instruction=""
        ),
        "click": StageConfig(
            instruction="""
                Select the results most likely to be relevant to the topic. Return an empty list if none look relevant.
                Use ONLY the rank numbers shown in the SERP (e.g., 1–10). Do not invent larger numbers or document IDs/titles.
                When listing multiple ranks, separate them with commas in the JSON array (e.g., [2, 3, 5]); do NOT concatenate them into a single number (e.g., [235]).
                Respond with JSON only: {"ranking_list": [rank_numbers], "reason": "<one short sentence>"}.
                Examples (follow exactly):
                  - Valid: {"ranking_list": [1, 3], "reason": "These best match the topic"}
                  - Valid empty: {"ranking_list": [], "reason": "None look relevant"}
                  - Invalid (out of range): {"ranking_list": [27]}
                  - Invalid (non-ranks): {"ranking_list": ["docA", "docB"]}
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
    max_topics=1,
    full_log=True
)

if __name__ == "__main__":
    runner = ExperimentRunner(settings=my_settings)
    runner.run()

# Archived JSON Prompt Variants

These are the JSON-structured stage prompts used before the simplification pass.
They are kept here for reproducibility and future ablations.

## Query (JSON)
```
Review the provided descriptions of task, corpus, tool and search topic. Then, formulate a search query.
Respond ONLY with a JSON object in this exact format (no markdown, no prose, no code fences):
{
"query": "your query",
"reason": "short sentence"
}
```

## Click (JSON, duplicate-avoidance)
```
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
```

## Relevance (JSON)
```
Evaluate the relevance of the document based on the search topic description and its narrative.
Return ONLY this JSON:
{
"label": "Relevant" or "NotRelevant",
"reason": "short sentence"
}
Do not output ranking_list or any other keys.
```

## Reformulate (JSON, BM25)
```
Formulate another search query to find new relevant documents. Make sure to consider previous relevant documents found our goal here
is to maximize the recall rate, the search index in use is BM25 so try to come up with strategies that can help retrieve more relevant documents.
Your output MUST be ONLY a JSON object in this exact format (no markdown, no prose, no code fences):
{
"query": "your query",
"reason": "short sentence"
}
```

## Reformulate (JSON, SPLADE)
```
Formulate another search query to find new relevant documents. Make sure to consider previous relevant documents found our goal here
is to maximize the recall rate. The search index in use is SPLADE, so try to include informative terms and entity variants that can
surface new relevant documents beyond what has already been found.
Your output MUST be ONLY a JSON object in this exact format (no markdown, no prose, no code fences):
{
"query": "your query",
"reason": "short sentence"
}
```

## Reformulate (JSON, SPLADE + metrics variant)
```
Formulate another search query to find new relevant documents. Make sure to consider previous relevant documents found our goal here
is to maximize the recall rate. The search index in use is SPLADE, so try to include informative terms and entity variants that can
surface new relevant documents beyond what has already been found.
Here are the {metrics} so far, please use them to guide your reformulation. We need to improve our recall!
Try to maximize the recall rate by finding new relevant documents in every reformulation should reach near 0.9 recall if possible.
Try to learn from the documents text you have found relevant so far and use them to guide your next query reformulation.
Your output MUST be ONLY a JSON object in this exact format (no markdown, no prose, no code fences):
{
"query": "your query",
"reason": "short sentence"
}
```

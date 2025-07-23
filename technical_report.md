# Technical Report

## Task 1:

- Submission File Name: `retrieval_submission.json`
- Description of this submission: Our approach involves preprocessing the legal corpus and indexing it into a Qdrant vector database using a custom-finetuned embedding model. For retrieval, we employ a hybrid search strategy combining dense vector search with sparse keyword matching, followed by a reranking step to refine the final results.

## Task 2:

- Submission File Name: `qna_submission.json`
- Description of this submission: Our solution for legal question answering uses the `Llama-3.1-8B-Instruct` model with a Tree-of-Thought prompting strategy from `prompts_TOT.json`. This guides the model to reason step-by-step, adapting to various question types to ensure accurate and well-supported answers.

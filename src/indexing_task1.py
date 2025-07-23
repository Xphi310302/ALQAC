#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Indexing script for ALQAC data.
Processes the preprocessed data, creates embeddings, and indexes them in a vector database.
"""

import json
import os
import uuid
from pathlib import Path
from typing import List, Dict

from llama_index.core.schema import (
    TextNode,
    NodeRelationship,
    RelatedNodeInfo,
    ObjectType,
    QueryBundle,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from law_search.vector_db import QdrantCollection

# Hardcoded configurations
MODEL_NAME = "phi010402/finetune-alqac-question-generation-2"
MODEL_CACHE_DIR = "../models"
COLLECTION_NAME = "law_sections"
TOP_K = 2
SPARSE_TOP_K = 12


def process_json_content(json_content: List[Dict], file_name: str) -> List[TextNode]:
    """
    Process JSON content and create TextNodes with relationships.

    Args:
        json_content: List containing the document json
        file_name: Name of the source file

    Returns:
        List of TextNodes with established relationships
    """
    nodes = []
    for content in json_content:
        section_id = list(content.keys())[0]
        section_data = list(content.values())[0]
        # Create text node
        node = TextNode(
            text=section_data,
            id_=str(uuid.uuid4()),
            metadata={
                "doc_id": file_name,
                "section_id": section_id,
                "title": section_data.split("\n\n")[0],
            },
        )
        nodes.append(node)

    for i, node in enumerate(nodes):
        if i > 0:
            node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                node_id=nodes[i - 1].node_id,
                node_type=ObjectType.TEXT,
                hash=nodes[i - 1].hash,
            )
        if i < len(nodes) - 1:
            node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                node_id=nodes[i + 1].node_id,
                node_type=ObjectType.TEXT,
                hash=nodes[i + 1].hash,
            )
    return nodes


def setup_embedding_model() -> None:
    """Initialize and setup the embedding model."""
    embed_model = HuggingFaceEmbedding(
        model_name=MODEL_NAME,
        trust_remote_code=True,
        cache_folder=os.path.join(MODEL_CACHE_DIR, MODEL_NAME),
    )
    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    Settings.db = QdrantCollection(collection_name=COLLECTION_NAME)


def index_documents(output_path: Path = Path("../output")) -> List[TextNode]:
    """
    Index documents from the output path into the vector database.

    Args:
        output_path: Path to the directory containing JSON files to index

    Returns:
        List of created TextNodes
    """
    nodes = []
    for filename in output_path.glob("*"):
        print(f"Indexing {filename.stem}")
        try:
            with open(filename, "r", encoding="utf-8") as file:
                json_content = json.load(file)
            nodes.extend(process_json_content(json_content, filename.stem))
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    try:
        Settings.db.insert_nodes(nodes)
        print(f"Successfully indexed {len(nodes)} nodes")
    except Exception as e:
        print(f"Error inserting nodes: {e}")

    return nodes


def retrieve(query: str) -> Dict:
    """
    Retrieve relevant documents for a given query.

    Args:
        query: Query string to search for

    Returns:
        Dictionary containing search results
    """
    retriever_engine = Settings.db._index.as_retriever(
        similarity_top_k=TOP_K,
        sparse_top_k=SPARSE_TOP_K,
        vector_store_query_mode="hybrid",
        node_postprocessor=[],
    )

    result_nodes = retriever_engine._retrieve(
        QueryBundle(
            query_str=query,
        )
    )

    result_dict = {"result": []}
    for node in result_nodes:
        if node.score < 0.5:
            if TOP_K > 1:
                continue
        else:
            result_dict["result"].append(
                {
                    "document": node.node.metadata["doc_id"],
                    "id": node.node.metadata["section_id"],
                    "score": node.score,
                    "text": node.node.text,
                }
            )
    return result_dict


def calculate_precision(retrieved_articles, relevant_articles):
    """
    Calculates precision for a single question.
    A retrieved article is correct if its (law_id, article_id) tuple matches a relevant article.
    """
    retrieved_set = {(item["document"], item["id"]) for item in retrieved_articles}
    relevant_set = {(item["law_id"], item["article_id"]) for item in relevant_articles}

    correctly_retrieved = len(retrieved_set.intersection(relevant_set))
    total_retrieved = len(retrieved_set)

    if total_retrieved == 0:
        return 0.0

    return correctly_retrieved / total_retrieved


def calculate_recall(retrieved_articles, relevant_articles):
    """
    Calculates recall for a single question.
    A retrieved article is correct if its (law_id, article_id) tuple matches a relevant article.
    """
    retrieved_set = {(item["document"], item["id"]) for item in retrieved_articles}
    relevant_set = {(item["law_id"], item["article_id"]) for item in relevant_articles}

    correctly_retrieved = len(retrieved_set.intersection(relevant_set))
    total_relevant = len(relevant_set)

    if total_relevant == 0:
        return 0.0

    return correctly_retrieved / total_relevant


def calculate_f2_score(precision, recall):
    """
    Calculates the F2 score based on the provided formula.
    """
    if (4 * precision + recall) == 0:
        return 0.0

    return (5 * precision * recall) / (4 * precision + recall)


def evaluate_retrieval(
    dataset_path="../ALQAC_2025_data/alqac25_private_test_Task_1.json",
    evaluation_path="evaluation",
):
    """
    Evaluate retrieval performance on a test dataset.

    Args:
        dataset_path: Path to the evaluation dataset
        evaluation_path: Path to save evaluation results
    """
    os.makedirs(evaluation_path, exist_ok=True)

    with open(dataset_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    all_precision = []
    all_recall = []
    all_f2_scores = []
    details = []

    for idx, item in enumerate(json_data):
        query = item["text"]
        relevant_articles = item["relevant_articles"]

        retrieved_results = retrieve(query)
        retrieved_articles = retrieved_results.get("result", [])
        precision = calculate_precision(retrieved_articles, relevant_articles)
        recall = calculate_recall(retrieved_articles, relevant_articles)
        f2 = calculate_f2_score(precision, recall)

        all_precision.append(precision)
        all_recall.append(recall)
        all_f2_scores.append(f2)
        details.append(
            {
                "question_id": item.get("question_id", idx),
                "query": query,
                "precision": precision,
                "recall": recall,
                "f2_score": f2,
                "retrieved_articles": retrieved_articles,
                "relevant_articles": relevant_articles,
            }
        )

    if all_f2_scores:
        average_f2 = sum(all_f2_scores) / len(all_f2_scores)
        average_precision = sum(all_precision) / len(all_precision)
        average_recall = sum(all_recall) / len(all_recall)
        print(f"Average Precision: {average_precision:.4f}")
        print(f"Average Recall: {average_recall:.4f}")
        print(f"Average F2-Score: {average_f2:.4f}")
    else:
        print("Could not calculate F2-Score, no data processed.")

    with open(
        f"{evaluation_path}/detailed_metrics_{TOP_K}_{SPARSE_TOP_K}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "average_precision": average_precision if all_f2_scores else None,
                "average_recall": average_recall if all_f2_scores else None,
                "average_f2_score": average_f2 if all_f2_scores else None,
                "details": details,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def main():
    """Main execution function."""
    print("Setting up embedding model...")
    setup_embedding_model()

    print("Indexing documents...")
    index_documents()

    print("Running evaluation...")
    evaluate_retrieval()


if __name__ == "__main__":
    main()

from llama_index.core.schema import QueryBundle


def retrieve(query, retriever_engine, top_k=2, is_postprocessed=True, threshold=0.5):
    result_nodes = retriever_engine._retrieve(QueryBundle(query_str=query))
    results = []
    for node in result_nodes:
        if not is_postprocessed or node.score >= threshold or top_k == 1:
            results.append(
                {
                    "document": node.node.metadata["doc_id"],
                    "id": node.node.metadata["section_id"],
                    "score": node.score,
                    "text": node.node.text,
                }
            )
    return {"result": results}

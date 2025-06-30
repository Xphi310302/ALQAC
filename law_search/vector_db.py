"""Vector database management using Qdrant."""

from pathlib import Path
from typing import List

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from qdrant_client.models import CollectionInfo

# from .config import MODEL_CACHE_DIR, VECTOR_DIMENSION


class QdrantCollection:
    """Manages vector storage using local Qdrant."""

    def __init__(self, collection_name: str = "law_sections"):
        """Initialize Qdrant collection."""
        # Create data directory with proper permissions
        current_dir = Path(__file__).parent.parent
        data_dir = current_dir / "qdrant"
        data_dir.mkdir(parents=True, exist_ok=True)

        print(f"Initializing Qdrant with data directory: {data_dir}")

        # Initialize local Qdrant client with explicit path
        try:
            self.client = QdrantClient(path=str(data_dir))
            self.collection_name = collection_name

            # Initialize vector store
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=collection_name,
                enable_hybrid=True,
                fastembed_sparse_model="Qdrant/bm25",
                batch_size=20,
            )

            self._index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
            )
            print("Vector store initialized successfully")
        except Exception as e:
            print(f"Error initializing Qdrant: {str(e)}")
            raise

    def insert_nodes(self, nodes: List[TextNode]) -> None:
        self._index.insert_nodes(nodes)

    def get_collection_stats(self) -> CollectionInfo:
        """Get collection statistics."""
        return self.client.get_collection(self.collection_name)

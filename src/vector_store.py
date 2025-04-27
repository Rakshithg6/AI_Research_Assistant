import chromadb
from chromadb.config import Settings
import os
from typing import List, Dict, Optional

class VectorStore:
    def __init__(self, persist_directory: str = "chroma_db"):
        self.persist_directory = persist_directory
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            is_persistent=True
        ))

    def create_collection(self, collection_name: str):
        """Create a new collection or get existing one."""
        try:
            collection = self.client.get_collection(name=collection_name)
        except Exception:
            collection = self.client.create_collection(name=collection_name)
        return collection

    def add_documents(self, collection_name: str, documents: List[str], metadata: Optional[List[Dict]] = None):
        """Add documents to the vector store."""
        collection = self.create_collection(collection_name)
        
        # Generate IDs for the documents
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        # If no metadata is provided, create empty metadata for each document
        if metadata is None:
            metadata = [{"source": "document"} for _ in documents]
        
        collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadata
        )

    def query(self, collection_name: str, query_text: str, n_results: int = 5) -> List[Dict]:
        """Query the vector store for similar documents."""
        collection = self.client.get_collection(name=collection_name)
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        
        return {
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0]
        }

    def delete_collection(self, collection_name: str):
        """Delete a collection from the vector store."""
        self.client.delete_collection(name=collection_name) 
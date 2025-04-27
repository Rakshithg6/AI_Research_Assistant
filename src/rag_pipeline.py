import google.generativeai as genai
from typing import List, Dict
import os
from dotenv import load_dotenv
from .vector_store import VectorStore
from .document_processor import DocumentProcessor

class RAGPipeline:
    def __init__(self):
        load_dotenv()
        
        # Configure Gemini with API key
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
        genai.configure(api_key=api_key)
        
        # Initialize Gemini 1.5 Pro model with safety settings
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        # Initialize components
        self.vector_store = VectorStore()
        self.document_processor = DocumentProcessor()

    def process_and_store_document(self, file_path: str, collection_name: str):
        """Process a document and store it in the vector database."""
        # Process the document into chunks
        chunks = self.document_processor.process_document(file_path)
        
        # Store chunks in vector store
        self.vector_store.add_documents(
            collection_name=collection_name,
            documents=chunks,
            metadata=[{"source": file_path} for _ in chunks]
        )

    def generate_response(self, query: str, collection_name: str) -> str:
        """Generate a response using RAG."""
        # Retrieve relevant chunks
        results = self.vector_store.query(
            collection_name=collection_name,
            query_text=query
        )

        # Construct prompt with context
        context = "\n".join(results["documents"])
        prompt = f"""Based on the following context, please answer the question. 
        If you cannot answer based on the context, say so.

        Context:
        {context}

        Question: {query}
        
        Instructions:
        1. Use only the provided context to answer
        2. If the context doesn't contain relevant information, say so
        3. Be concise but thorough
        4. Cite specific parts of the context if relevant
        """

        # Generate response
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def list_collections(self) -> List[str]:
        """List all available collections."""
        return self.vector_store.client.list_collections()

    def delete_collection(self, collection_name: str):
        """Delete a collection."""
        self.vector_store.delete_collection(collection_name) 
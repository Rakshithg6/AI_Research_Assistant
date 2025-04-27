import google.generativeai as genai
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
from .rag_pipeline import RAGPipeline
from .tools.summarize import SummarizeTool
from .tools.extract_kpis import ExtractKPIsTool
from .tools.generate_report import GenerateReportTool
from .tools.search_web import SearchWebTool

class Agent:
    def __init__(self):
        load_dotenv()
        
        # Initialize Gemini
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        
        # Configure model with appropriate settings
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
        
        # Initialize RAG pipeline
        self.rag_pipeline = RAGPipeline()
        
        # Initialize tools
        self.tools = {
            "summarize": SummarizeTool(),
            "extract_kpis": ExtractKPIsTool(),
            "generate_report": GenerateReportTool(),
            "search_web": SearchWebTool()
        }

    def process_document(self, file_path: str, collection_name: str):
        """Process and store a document in the vector database."""
        self.rag_pipeline.process_and_store_document(file_path, collection_name)

    def select_tool(self, query: str) -> str:
        """Select the most appropriate tool based on the query."""
        prompt = f"""Given the following user query, select the most appropriate tool from the available options.
        
        Available Tools:
        - summarize: For summarizing content or documents
        - extract_kpis: For extracting key metrics and numbers
        - generate_report: For creating structured reports
        - search_web: For finding recent information
        
        Query: {query}
        
        Return only the name of the most appropriate tool."""

        response = self.model.generate_content(prompt)
        selected_tool = response.text.strip().lower()
        
        if selected_tool not in self.tools:
            return "summarize"  # Default to summarize if tool selection fails
        return selected_tool

    def execute_query(self, query: str, collection_name: str) -> Dict[str, Any]:
        """Execute a query using the most appropriate tool and RAG pipeline."""
        # First, get relevant context from RAG
        context = self.rag_pipeline.generate_response(query, collection_name)
        
        # Select appropriate tool
        tool_name = self.select_tool(query)
        tool = self.tools[tool_name]
        
        # Execute tool with correct arguments
        if tool_name == "generate_report":
            result_text = tool.execute(
                topic=query,
                context=context,
                collection_name=collection_name
            )
        else:
            result_text = tool.execute(
                content=context,
                query=query,
                collection_name=collection_name
            )
        
        return {
            "tool_used": tool_name,
            "context": context,
            "result": result_text
        }

    def list_collections(self) -> List[str]:
        """List all available collections."""
        return self.rag_pipeline.list_collections()

    def delete_collection(self, collection_name: str):
        """Delete a collection."""
        self.rag_pipeline.delete_collection(collection_name)

    def get_available_tools(self) -> Dict[str, str]:
        """Get list of available tools and their descriptions."""
        return {name: str(tool) for name, tool in self.tools.items()} 
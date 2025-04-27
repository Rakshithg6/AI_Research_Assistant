import requests
from .base_tool import BaseTool
import os
from typing import List, Dict
from dotenv import load_dotenv
import google.generativeai as genai

class SearchWebTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="search_web",
            description="Fetches recent web results using Gemini's knowledge"
        )
        
        # Configure Gemini
        load_dotenv()
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
        genai.configure(api_key=api_key)
        
        # Initialize model with configuration
        generation_config = {
            "temperature": 0.3,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 1024,
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

    def execute(self, query: str, **kwargs) -> str:
        """
        Use Gemini to answer the user's query directly, as accurately and informatively as possible.
        """
        prompt = f"""Answer the following question as accurately and informatively as possible, using your latest knowledge:\n\nQuestion: {query}\n\nIf you do not know the answer or your knowledge may be outdated, say so clearly."""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error processing search request: {str(e)}" 
import google.generativeai as genai
from .base_tool import BaseTool
import os
from dotenv import load_dotenv

class SummarizeTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="summarize",
            description="Summarizes a section or full document"
        )
        
        # Configure Gemini
        load_dotenv()
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
        genai.configure(api_key=api_key)
        
        # Initialize model with configuration
        generation_config = {
            "temperature": 0.3,  # Lower temperature for more focused summaries
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

    def execute(self, content: str, **kwargs) -> str:
        """Summarize the given content."""
        prompt = f"""Please provide a concise summary of the following content:

        {content}
        
        Instructions:
        1. Focus on the key points and main ideas
        2. Be clear and objective
        3. Maintain the original meaning
        4. Use bullet points for clarity if appropriate
        5. Include important numbers or statistics if present"""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating summary: {str(e)}" 
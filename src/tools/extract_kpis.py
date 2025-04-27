import google.generativeai as genai
from .base_tool import BaseTool
from typing import List, Dict
import os
from dotenv import load_dotenv

class ExtractKPIsTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="extract_kpis",
            description="Extracts KPIs and numeric metrics from content"
        )
        
        # Configure Gemini
        load_dotenv()
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
        genai.configure(api_key=api_key)
        
        # Initialize model with configuration
        generation_config = {
            "temperature": 0.1,  # Very low temperature for precise extraction
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

    def execute(self, content: str, **kwargs) -> List[Dict]:
        """Extract KPIs from the given content."""
        prompt = f"""Extract all KPIs and numeric metrics from the following content. 
If there are no KPIs or numeric metrics, respond with: 'No KPIs or numeric metrics found in the provided content.'

Content:
{content}

Instructions:
1. Only extract what is explicitly present in the content.
2. Do not suggest or invent KPIs.
3. Format each KPI as: [Metric Name]: [Value] [Unit] (if applicable)
4. Include context or description where relevant
5. Group related metrics together
6. Highlight significant trends or changes
7. Note any time periods or dates associated with metrics."""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error extracting KPIs: {str(e)}" 
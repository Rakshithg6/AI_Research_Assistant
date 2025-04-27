import google.generativeai as genai
from .base_tool import BaseTool
import os
from dotenv import load_dotenv

class GenerateReportTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="generate_report",
            description="Creates a brief report based on retrieved information"
        )
        
        # Configure Gemini
        load_dotenv()
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
        genai.configure(api_key=api_key)
        
        # Initialize model with configuration
        generation_config = {
            "temperature": 0.5,  # Balanced temperature for creative yet focused reports
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,  # Longer output for detailed reports
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

    def execute(self, topic: str, context: str, **kwargs) -> str:
        """Generate a report based on the topic and context."""
        prompt = f"""Please generate a comprehensive report on the following topic using the provided context.
        
        Topic: {topic}
        
        Context:
        {context}
        
        Instructions:
        1. Structure the report with the following sections:
           - Executive Summary
           - Key Findings
           - Detailed Analysis
           - Recommendations (if applicable)
           - Conclusions
        
        2. Guidelines:
           - Be clear, professional, and objective
           - Support findings with data from the context
           - Use bullet points or numbered lists where appropriate
           - Highlight critical insights
           - Include relevant metrics and KPIs
           - Suggest actionable recommendations if applicable"""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating report: {str(e)}" 
import PyPDF2
from typing import List, Dict
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def process_pdf(self, file_path: str) -> List[str]:
        """Process a PDF file and return chunks of text."""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text()
            
        return self.chunk_text(text)

    def process_txt(self, file_path: str) -> List[str]:
        """Process a TXT file and return chunks of text."""
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return self.chunk_text(text)

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks using langchain's text splitter."""
        return self.text_splitter.split_text(text)

    def process_document(self, file_path: str) -> List[str]:
        """Process a document based on its file extension."""
        _, ext = os.path.splitext(file_path)
        if ext.lower() == '.pdf':
            return self.process_pdf(file_path)
        elif ext.lower() == '.txt':
            return self.process_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}") 
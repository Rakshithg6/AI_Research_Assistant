# Smart AI Research Assistant

A lightweight AI-powered assistant that can process documents, answer questions, and act autonomously using various tools.

![image](https://github.com/user-attachments/assets/3e866c54-15ad-4ba1-a36a-99c3341e9d7f)


## Features

- Document Processing (PDF & TXT)
- RAG (Retrieval-Augmented Generation) Pipeline
- Vector Database Storage
- Autonomous Tool Usage
- Multiple Project Support
- Web-based GUI Interface

## Setup Instructions

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the root directory and add your API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```
5. Run the application:
   ```bash
   streamlit run app.py
   ```

## Project Structure

- `app.py`: Main Streamlit application
- `src/`
  - `document_processor.py`: Document processing and chunking
  - `vector_store.py`: ChromaDB integration
  - `rag_pipeline.py`: RAG implementation
  - `tools/`: Autonomous tools implementation
  - `agent.py`: Agent behavior and decision making

## How It Works

### RAG Pipeline
1. Documents are uploaded and processed
2. Content is chunked and stored in ChromaDB
3. User queries trigger relevant chunk retrieval
4. Gemini Pro processes chunks and generates responses

### Agentic Behavior
- Autonomous tool selection based on user intent
- Tool chaining for complex queries
- Context-aware responses

## Tools Available

- `summarize`: Summarizes document sections
- `extract_kpis`: Extracts key metrics
- `generate_report`: Creates reports from context
- `search_web`: Fetches recent web results 

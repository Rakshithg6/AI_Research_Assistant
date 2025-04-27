import streamlit as st
import os
from src.agent import Agent
from dotenv import load_dotenv
import json
import re

# Set page config (must be the first Streamlit command)
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stSidebar {
        background-color: #f5f5f5;
        padding: 2rem 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    .user-message {
        background-color: #f0f7ff;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()

# Check for API key
if not os.getenv("GOOGLE_API_KEY"):
    st.error("⚠️ GOOGLE_API_KEY not found in environment variables. Please set it up in your .env file.")
    st.stop()

# Initialize the agent
@st.cache_resource
def get_agent():
    try:
        return Agent()
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        return None

agent = get_agent()
if not agent:
    st.error("Failed to initialize the AI agent. Please check your configuration.")
    st.stop()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_tool" not in st.session_state:
    st.session_state.selected_tool = "auto"

# Sidebar
with st.sidebar:
    st.title("AI Research Assistant")
    st.markdown("---")

    # Tool selection
    st.subheader("Select Tool")
    tool_options = {
        "auto": "Auto Select (Recommended)",
        "summarize": "Summarize Content",
        "extract_kpis": "Extract KPIs",
        "generate_report": "Generate Report",
        "search_web": "Search Web"
    }
    selected_tool = st.radio(
        "Choose a tool:",
        options=list(tool_options.keys()),
        format_func=lambda x: tool_options[x],
        key="tool_selector"
    )

    # File upload
    st.markdown("---")
    st.subheader("Document Upload")
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt"])
    collection_name = "default"  # Always use default collection

    if uploaded_file is not None:
        try:
            # Create uploads directory if it doesn't exist
            os.makedirs("uploads", exist_ok=True)
            
            # Save the uploaded file
            file_path = os.path.join("uploads", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process the document
            with st.spinner("Processing document..."):
                agent.process_document(file_path, collection_name)
                st.success("Document processed successfully!")
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")

# Main content
st.title("AI Research Assistant")
st.markdown("---")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "tool_used" in message:
            st.caption(f"Tool used: {message['tool_used']}")

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                if st.session_state.selected_tool == "auto":
                    result = agent.execute_query(prompt, collection_name)
                else:
                    tool = agent.tools[st.session_state.selected_tool]
                    context = agent.rag_pipeline.generate_response(prompt, collection_name)
                    if st.session_state.selected_tool == "generate_report":
                        result_text = tool.execute(
                            topic=prompt,
                            context=context,
                            collection_name=collection_name
                        )
                    else:
                        result_text = tool.execute(
                            content=context,
                            query=prompt,
                            collection_name=collection_name
                        )
                    result = {
                        "tool_used": st.session_state.selected_tool,
                        "context": context,
                        "result": result_text
                    }
                
                # Display response
                st.markdown(result["result"])
                
                # Add assistant message to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["result"],
                    "tool_used": result["tool_used"]
                })
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

# Sidebar utilities
with st.sidebar:
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.experimental_rerun()

    if st.button("Export Chat History"):
        try:
            chat_history = json.dumps(st.session_state.messages, indent=2)
            st.download_button(
                label="Download Chat History",
                data=chat_history,
                file_name="chat_history.json",
                mime="application/json"
            )
        except Exception as e:
            st.error(f"Error exporting chat history: {str(e)}") 
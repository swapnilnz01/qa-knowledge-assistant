"""
QA Knowledge Assistant - Web Interface
Run with: streamlit run app.py
"""

import streamlit as st
from main import QAKnowledgeAssistant
from main_enhanced import EnhancedQAAssistant

# Page configuration
st.set_page_config(
    page_title="QA Knowledge Assistant",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("🤖 QA Knowledge Assistant")
st.markdown("""
This AI-powered assistant can answer questions about QA testing best practices, 
strategies, and methodologies using **Retrieval-Augmented Generation (RAG)**.

**How it works:**
1. Your question is converted to a vector (embedding)
2. We search our QA knowledge base for similar content
3. Relevant information is retrieved and presented
""")

# Initialize assistant (with caching to avoid reloading)
@st.cache_resource
def load_assistant():
    return QAKnowledgeAssistant()
    # return EnhancedQAAssistant(use_llm=True, llm_provider="openai")  # Change to "anthropic" for Claude integration 

assistant = load_assistant()

# Sidebar with example questions
st.sidebar.header("📋 Example Questions")
example_questions = [
    "What is the testing pyramid?",
    "How do I test REST APIs?",
    "What are the bug severity levels?",
    "Explain boundary value analysis",
    "What is contract testing?",
    "How to do performance testing?",
    "What are OWASP top 10 vulnerabilities?",
    "Best practices for test automation",
    "What is exploratory testing?",
    "How to integrate tests in CI/CD?"
]

for question in example_questions:
    if st.sidebar.button(question, key=question):
        st.session_state.selected_question = question

# Main chat interface
st.header("💬 Ask Your Question")

# Get question from user or sidebar selection
question = st.text_input(
    "Type your QA-related question:",
    value=st.session_state.get('selected_question', ''),
    placeholder="e.g., What is regression testing?"
)

# Search button
if st.button("🔍 Get Answer", type="primary"):
    if question:
        with st.spinner("Searching knowledge base..."):
            # Get relevant chunks
            relevant_chunks = assistant.search_knowledge(question, top_k=3)
            
            if relevant_chunks:
                st.success("✅ Found relevant information!")
                
                # Display answer in expandable sections
                st.subheader("📚 Relevant Information:")
                
                for i, chunk in enumerate(relevant_chunks, 1):
                    with st.expander(f"📖 Source {i}", expanded=(i==1)):
                        st.markdown(chunk)
                
                # Show similarity info
                st.info(f"Retrieved {len(relevant_chunks)} relevant sections from knowledge base")
            else:
                st.error("❌ No relevant information found. Try rephrasing your question.")
    else:
        st.warning("⚠️ Please enter a question first!")

# Clear button
if st.button("🗑️ Clear"):
    st.session_state.selected_question = ""
    st.rerun()

# Footer with tech stack info
st.markdown("---")
st.markdown("""
**🛠️ Tech Stack:**
- Vector DB: ChromaDB
- Embeddings: SentenceTransformers (all-MiniLM-L6-v2)
- Framework: LangChain concepts
- UI: Streamlit

**🚀 Next Steps:**
- Add LLM integration (OpenAI/Anthropic) for natural language answers
- Expand knowledge base with more QA documents
- Add conversation memory
- Implement source citation
""")

# Display knowledge base stats
with st.sidebar:
    st.markdown("---")
    st.header("📊 Knowledge Base Stats")
    st.metric("Total Chunks", assistant.collection.count())
    st.metric("Embedding Model", "all-MiniLM-L6-v2")
    st.metric("Vector Dimensions", "384")

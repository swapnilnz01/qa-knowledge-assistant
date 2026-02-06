# 🤖 QA Knowledge Assistant - RAG Project

A **Retrieval-Augmented Generation (RAG)** system that answers QA testing questions using vector search and embeddings. Built with ChromaDB, SentenceTransformers, and Streamlit.

## 🎯 What You'll Learn

- ✅ How RAG systems work
- ✅ Vector databases and embeddings
- ✅ Semantic search implementation
- ✅ Building AI-powered applications
- ✅ LangChain concepts in practice

## 🏗️ Architecture

```
User Question
    ↓
Convert to Vector (Embedding)
    ↓
Search Vector Database (ChromaDB)
    ↓
Retrieve Top 3 Similar Chunks
    ↓
Return Relevant Information
```

## 📋 Prerequisites

- Python 3.9 or higher
- pip package manager
- 2GB free disk space (for embedding model)

## 🚀 Quick Start

### 1. Clone/Download Project Files

Make sure you have these files:
```
qa-knowledge-assistant/
├── requirements.txt
├── main.py
├── app.py
├── qa_knowledge_base.txt
└── README.md
```

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

**Note:** First run will download the embedding model (~80MB). This is normal!

### 3. Run the Application

**Option A: Command Line Interface**
```bash
python main.py
```

**Option B: Web Interface (Recommended)**
```bash
streamlit run app.py
```

Then open your browser to: `http://localhost:8501`

## 💡 How to Use

### Web Interface

1. Open the Streamlit app
2. Click example questions in sidebar OR type your own
3. Click "Get Answer"
4. Review the relevant sections retrieved

### Command Line

1. Run `python main.py`
2. Wait for the demo to complete
3. Enter interactive mode
4. Type questions and press Enter
5. Type 'quit' to exit




## 🧪 Example Questions to Try

```
- What is the testing pyramid?
- How do I test REST APIs?
- Explain bug severity levels
- What is exploratory testing?
- Best practices for CI/CD integration
- What is contract testing for microservices?
- How to do performance testing?
- What are common security vulnerabilities?
```

## 🔍 How It Works

### 1. Knowledge Base Chunking
```python
# The knowledge file is split into sections
chunks = split_by_double_newlines(qa_knowledge_base.txt)
# Example: "## API Testing Best Practices\n\nWhen testing REST APIs..."
```

### 2. Embedding Generation
```python
# Each chunk is converted to a 384-dimensional vector
embedding = model.encode("What is the testing pyramid?")
# Result: [0.123, -0.456, 0.789, ...] (384 numbers)
```

### 3. Vector Storage
```python
# Vectors stored in ChromaDB
collection.add(documents=[chunk], embeddings=[embedding])
```

### 4. Semantic Search
```python
# User question is embedded and compared
query_embedding = model.encode("How to test APIs?")
similar_chunks = collection.query(query_embedding, top_k=3)
```

### 5. Result Retrieval
```python
# Top 3 most similar chunks are returned
# Similarity measured by cosine distance
```

## 📊 Technical Details

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Vector DB | ChromaDB | Store and search embeddings |
| Embeddings | SentenceTransformers | Convert text to vectors |
| Model | all-MiniLM-L6-v2 | Fast, efficient embeddings |
| UI | Streamlit | Web interface |
| Language | Python 3.9+ | Implementation |

## 🎓 Learning Checkpoints

After completing this project, you should understand:

- [x] What embeddings are and how they work
- [x] How vector databases enable semantic search
- [x] The difference between keyword search and semantic search
- [x] Basic RAG architecture
- [x] How to chunk documents for retrieval
- [x] How to measure similarity between texts

## 🚀 Next Steps (Enhancements)

### Phase 2: Add LLM Integration
```python
# Current: Returns raw chunks
# Next: Use GPT/Claude to generate natural answers

def generate_answer(query, chunks):
    prompt = f"Answer this question: {query}\nContext: {chunks}"
    return openai.chat.completions.create(...)
```

### Phase 3: Add More Features
- [ ] Conversation memory (chat history)
- [ ] Source citation with confidence scores
- [ ] Upload custom documents
- [ ] Multi-language support
- [ ] Question refinement suggestions

### Phase 4: Production Ready
- [ ] Add logging and monitoring
- [ ] Implement caching for common queries
- [ ] Deploy to cloud (AWS/Azure/GCP)
- [ ] Add authentication
- [ ] API endpoint creation

## 🐛 Troubleshooting

### "Model not found" error
```bash
# Manually download the model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### "Port already in use" (Streamlit)
```bash
# Use a different port
streamlit run app.py --server.port 8502
```

### ChromaDB errors
```bash
# Clear the database
rm -rf chroma_db/  # Mac/Linux
rmdir /s chroma_db  # Windows
```

## 📚 Resources to Learn More

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [SentenceTransformers](https://www.sbert.net/)
- [What are Embeddings?](https://platform.openai.com/docs/guides/embeddings)
- [RAG Explained](https://www.pinecone.io/learn/retrieval-augmented-generation/)

## 🎯 Interview Talking Points

When discussing this project in interviews:

1. **Architecture**: "I built a RAG system using vector embeddings and semantic search"
2. **Technology**: "Used ChromaDB for vector storage and SentenceTransformers for embeddings"
3. **Problem Solved**: "Created a QA knowledge assistant that finds relevant information using semantic similarity, not just keyword matching"
4. **Key Learning**: "Understood how embeddings capture semantic meaning and enable similarity search"
5. **Next Steps**: "Planning to integrate GPT-4 for natural language generation and deploy to AWS"

## 📝 License

Free to use for learning and interview preparation!

## 🤝 Contributing

This is a learning project. Feel free to:
- Add more QA knowledge documents
- Improve chunking strategies
- Enhance the UI
- Add new features

---

**Built as a learning project for Gen AI interviews** 🚀

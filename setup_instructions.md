# QA Knowledge Assistant - Setup Instructions

## Prerequisites

1. **Python 3.9+** installed
2. **pip** package manager
3. Code editor (VS Code recommended)
4. OpenAI API key OR Anthropic API key (we'll use free alternatives if needed)

## Installation Steps

### 1. Create Project Directory
```bash
mkdir qa-knowledge-assistant
cd qa-knowledge-assistant
```

### 2. Create Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Packages
```bash
pip install langchain
pip install langchain-community
pip install openai
pip install chromadb
pip install sentence-transformers
pip install python-dotenv
pip install streamlit
```

### 4. Create Project Structure
```
qa-knowledge-assistant/
├── .env                    # API keys (don't commit this!)
├── requirements.txt        # Package dependencies
├── data/                   # QA knowledge documents
│   └── qa_docs.txt
├── main.py                # Main RAG implementation
├── app.py                 # Streamlit UI
└── README.md
```

### 5. Get API Key (Choose ONE option)

**Option A: OpenAI (Paid but easiest)**
- Go to: https://platform.openai.com/api-keys
- Create account and get API key
- Cost: ~$0.002 per request (very cheap for testing)

**Option B: Use Free Local Models**
- We'll use Sentence Transformers (free, runs locally)
- No API key needed!

**Option C: Anthropic Claude**
- Go to: https://console.anthropic.com/
- Get API key
- Cost: Similar to OpenAI

### 6. Create .env File
```bash
# If using OpenAI
OPENAI_API_KEY=kjey here
# If using Anthropic
ANTHROPIC_API_KEY=your_key_here
```

## Verify Installation

Run this test:
```python
import langchain
import chromadb
from sentence_transformers import SentenceTransformer

print("All packages installed successfully!")
```

## Next Steps

Once setup is complete, we'll:
1. Create sample QA documentation
2. Build the RAG pipeline
3. Create a simple interface
4. Test with real queries

---

**Ready?** Let me know when you've completed the setup, or if you hit any errors!

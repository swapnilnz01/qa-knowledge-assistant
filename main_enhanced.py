"""
QA Knowledge Assistant - ENHANCED VERSION with LLM
This version adds natural language generation using OpenAI/Anthropic
(Optional - requires API key)
"""

import os
from typing import List
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# Optional: Uncomment when you have an API key
import openai
# from anthropic import Anthropic


class EnhancedQAAssistant:
    """RAG-based assistant with LLM integration for natural answers"""
    
    def __init__(
        self, 
        knowledge_file: str = "qa_knowledge_base.txt",
        use_llm: bool = False,
        llm_provider: str = "openai"  # or "anthropic"
    ):
        """
        Initialize the enhanced assistant
        
        Args:
            knowledge_file: Path to knowledge base
            use_llm: Whether to use LLM for answer generation
            llm_provider: "openai" or "anthropic"
        """
        print("🚀 Initializing Enhanced QA Assistant...")
        
        # Vector database setup (same as before)
        self.client = chromadb.Client()
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name="qa_knowledge",
            embedding_function=self.embedding_function
        )
        
        # Load knowledge
        self.knowledge_file = knowledge_file
        self._load_knowledge_base()
        
        # LLM setup
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        
        if use_llm:
            self._setup_llm()
        
        print("✅ Enhanced Assistant ready!")
    
    def _setup_llm(self):
        """Setup LLM client"""
        if self.llm_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("⚠️  OPENAI_API_KEY not found in environment")
                self.use_llm = False
            # Uncomment when ready:
            self.llm_client = openai.OpenAI(api_key=api_key)
        
        elif self.llm_provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                print("⚠️  ANTHROPIC_API_KEY not found in environment")
                self.use_llm = False
            # Uncomment when ready:
            # self.llm_client = Anthropic(api_key=api_key)
    
    def _load_knowledge_base(self):
        """Load knowledge base (same as basic version)"""
        if not os.path.exists(self.knowledge_file):
            return
        
        if self.collection.count() > 0:
            print(f"📚 Knowledge base already loaded ({self.collection.count()} chunks)")
            return
        
        with open(self.knowledge_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        chunks = self._chunk_text(content)
        
        for i, chunk in enumerate(chunks):
            self.collection.add(documents=[chunk], ids=[f"chunk_{i}"])
        
        print(f"✅ Loaded {len(chunks)} knowledge chunks")
    
    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Chunk text into smaller pieces"""
        sections = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            if len(current_chunk) + len(section) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = section
            else:
                current_chunk += "\n\n" + section if current_chunk else section
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def search_knowledge(self, query: str, top_k: int = 3) -> List[str]:
        """Search knowledge base"""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        if results and results['documents']:
            return results['documents'][0]
        return []
    
    def generate_with_openai(self, query: str, context: List[str]) -> str:
        """
        Generate answer using OpenAI GPT
        
        UNCOMMENT WHEN READY TO USE:
        """
        # Combine context
        combined_context = "\n\n".join(context)
        
        # Create prompt
        prompt = f"""You are a QA testing expert. Answer the following question based on the provided context.
        
Context:
{combined_context}

Question: {query}

Instructions:
- Provide a clear, concise answer
- Use information from the context
- If the context doesn't fully answer the question, say so
- Format your response in a helpful way

Answer:"""
        
        # UNCOMMENT THIS WHEN YOU HAVE AN API KEY:

        response = self.llm_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful QA testing expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
        
        # For now, return placeholder
        return f"[LLM would generate answer here based on context]\n\nContext used:\n{combined_context}"
    
    def generate_with_anthropic(self, query: str, context: List[str]) -> str:
        """
        Generate answer using Anthropic Claude
        
        UNCOMMENT WHEN READY TO USE:
        """
        combined_context = "\n\n".join(context)
        
        prompt = f"""Answer this QA testing question based on the context provided.

Context:
{combined_context}

Question: {query}

Provide a clear, helpful answer."""
        
        # UNCOMMENT THIS WHEN YOU HAVE AN API KEY:
        message = self.llm_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return message.content[0].text

        
        return f"[Claude would generate answer here]\n\nContext:\n{combined_context}"
    
    def ask(self, question: str) -> str:
        """
        Ask a question and get an answer
        
        Args:
            question: User's question
        
        Returns:
            Generated answer (with or without LLM)
        """
        print(f"\n❓ Question: {question}")
        
        # Search for relevant context
        relevant_chunks = self.search_knowledge(question, top_k=3)
        
        if not relevant_chunks:
            return "❌ No relevant information found in knowledge base."
        
        # Generate answer
        if self.use_llm:
            if self.llm_provider == "openai":
                answer = self.generate_with_openai(question, relevant_chunks)
            else:
                answer = self.generate_with_anthropic(question, relevant_chunks)
        else:
            # Fallback: just return chunks
            answer = "📚 Relevant information found:\n\n"
            for i, chunk in enumerate(relevant_chunks, 1):
                answer += f"**Section {i}:**\n{chunk}\n\n"
        
        print(f"\n💡 Answer:\n{answer}")
        return answer


# USAGE EXAMPLE
def example_usage():
    """
    How to use the enhanced version
    """
    
    print("="*60)
    print("EXAMPLE 1: Without LLM (FREE)")
    print("="*60)
    
    # Basic version (no API key needed)
    assistant_basic = EnhancedQAAssistant(use_llm=False)
    assistant_basic.ask("What is the testing pyramid?")
    
    print("\n" + "="*60)
    print("EXAMPLE 2: With OpenAI (REQUIRES API KEY)")
    print("="*60)
    
    # With OpenAI (uncomment when you have API key)
    # os.environ["OPENAI_API_KEY"] = "your-key-here"
    # assistant_openai = EnhancedQAAssistant(use_llm=True, llm_provider="openai")
    # assistant_openai.ask("What is the testing pyramid?")
    
    print("\n⚠️  OpenAI integration commented out - add API key to enable")
    
    print("\n" + "="*60)
    print("EXAMPLE 3: With Anthropic Claude (REQUIRES API KEY)")
    print("="*60)
    
    # With Anthropic (uncomment when you have API key)
    # os.environ["ANTHROPIC_API_KEY"] = "your-key-here"
    # assistant_claude = EnhancedQAAssistant(use_llm=True, llm_provider="anthropic")
    # assistant_claude.ask("What is the testing pyramid?")
    
    print("\n⚠️  Anthropic integration commented out - add API key to enable")


if __name__ == "__main__":
    example_usage()

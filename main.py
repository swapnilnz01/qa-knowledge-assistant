"""
QA Knowledge Assistant - RAG Implementation
This uses FREE local models - no API key needed!
"""

import os
from typing import List
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

class QAKnowledgeAssistant:
    """RAG-based QA Knowledge Assistant using local embeddings"""
    
    def __init__(self, knowledge_file: str = "qa_knowledge_base.txt"):
        """Initialize the assistant with knowledge base"""
        print("🚀 Initializing QA Knowledge Assistant...")
        
        # Initialize ChromaDB (vector database)
        self.client = chromadb.Client()
        
        # Use sentence-transformers for embeddings (FREE, runs locally)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create embedding function for ChromaDB
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="qa_knowledge",
            embedding_function=self.embedding_function
        )
        
        # Load knowledge base
        self.knowledge_file = knowledge_file
        self._load_knowledge_base()
        
        print("✅ Assistant ready!")
    
    def _load_knowledge_base(self):
        """Load and chunk the knowledge base into vector database"""
        
        if not os.path.exists(self.knowledge_file):
            print(f"⚠️  Knowledge file not found: {self.knowledge_file}")
            return
        
        # Check if already loaded
        if self.collection.count() > 0:
            print(f"📚 Knowledge base already loaded ({self.collection.count()} chunks)")
            return
        
        print("📖 Loading knowledge base...")
        
        # Read the knowledge file
        with open(self.knowledge_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into chunks (simple approach: split by sections)
        chunks = self._chunk_text(content)
        
        # Add to vector database
        for i, chunk in enumerate(chunks):
            self.collection.add(
                documents=[chunk],
                ids=[f"chunk_{i}"]
            )
        
        print(f"✅ Loaded {len(chunks)} knowledge chunks")
    
    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Split text into chunks
        Simple strategy: split by double newlines (sections)
        """
        # Split by double newlines (paragraphs/sections)
        sections = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # If adding this section exceeds chunk size, save current chunk
            if len(current_chunk) + len(section) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = section
            else:
                current_chunk += "\n\n" + section if current_chunk else section
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def search_knowledge(self, query: str, top_k: int = 3) -> List[str]:
        """
        Search the knowledge base for relevant information
        
        Args:
            query: User's question
            top_k: Number of relevant chunks to return
        
        Returns:
            List of relevant text chunks
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # Extract documents from results
        if results and results['documents']:
            return results['documents'][0]
        
        return []
    
    def generate_answer(self, query: str) -> str:
        """
        Generate answer using retrieved knowledge
        
        NOTE: This version doesn't use an LLM for generation.
        It just returns the relevant chunks.
        We'll add LLM integration next!
        """
        # Search for relevant knowledge
        relevant_chunks = self.search_knowledge(query, top_k=3)
        
        if not relevant_chunks:
            return "❌ I couldn't find relevant information in my knowledge base."
        
        # For now, just format and return the chunks
        # (We'll add LLM-based answer generation in next version)
        answer = "📚 Here's what I found in the QA knowledge base:\n\n"
        
        for i, chunk in enumerate(relevant_chunks, 1):
            answer += f"**Relevant Section {i}:**\n{chunk}\n\n"
        
        return answer
    
    def ask(self, question: str) -> str:
        """
        Main interface: Ask a question and get an answer
        
        Args:
            question: User's question about QA
        
        Returns:
            Generated answer
        """
        print(f"\n❓ Question: {question}")
        answer = self.generate_answer(question)
        print(f"\n💡 Answer:\n{answer}")
        return answer


def main():
    """Demo the QA Knowledge Assistant"""
    
    # Create assistant
    assistant = QAKnowledgeAssistant()
    
    print("\n" + "="*60)
    print("🎯 QA KNOWLEDGE ASSISTANT - DEMO")
    print("="*60)
    
    # Test questions
    test_questions = [
        "What is the testing pyramid?",
        "How do I test REST APIs?",
        "What are the bug severity levels?",
        "What is exploratory testing?",
        "How should I integrate tests in CI/CD?"
    ]
    
    for question in test_questions:
        print("\n" + "-"*60)
        assistant.ask(question)
        print("-"*60)
    
    print("\n✅ Demo complete!")
    
    # Interactive mode
    print("\n" + "="*60)
    print("💬 INTERACTIVE MODE")
    print("="*60)
    print("Ask me anything about QA! (Type 'quit' to exit)")
    
    while True:
        question = input("\n❓ Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not question:
            continue
        
        assistant.ask(question)


if __name__ == "__main__":
    main()

"""
Quick Test Script - Verify Installation
Run this to check if everything is set up correctly
"""

def test_imports():
    """Test if all required packages are installed"""
    print("🔍 Testing imports...")
    
    try:
        import chromadb
        print("✅ ChromaDB installed")
    except ImportError:
        print("❌ ChromaDB not found. Run: pip install chromadb")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ SentenceTransformers installed")
    except ImportError:
        print("❌ SentenceTransformers not found. Run: pip install sentence-transformers")
        return False
    
    try:
        import streamlit
        print("✅ Streamlit installed")
    except ImportError:
        print("❌ Streamlit not found. Run: pip install streamlit")
        return False
    
    return True


def test_embedding_model():
    """Test if embedding model can be loaded"""
    print("\n🔍 Testing embedding model...")
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded successfully")
        
        # Test encoding
        test_text = "This is a test"
        embedding = model.encode(test_text)
        print(f"✅ Test embedding created (dimension: {len(embedding)})")
        
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


def test_chromadb():
    """Test ChromaDB functionality"""
    print("\n🔍 Testing ChromaDB...")
    
    try:
        import chromadb
        from chromadb.utils import embedding_functions
        
        # Create test client
        client = chromadb.Client()
        
        # Create test collection
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        collection = client.get_or_create_collection(
            name="test_collection",
            embedding_function=ef
        )
        
        # Add test document
        collection.add(
            documents=["This is a test document about QA testing"],
            ids=["test_1"]
        )
        
        # Query test
        results = collection.query(
            query_texts=["What is QA?"],
            n_results=1
        )
        
        print("✅ ChromaDB working correctly")
        print(f"✅ Test query returned: {len(results['documents'][0])} results")
        
        return True
    except Exception as e:
        print(f"❌ ChromaDB error: {e}")
        return False


def test_knowledge_file():
    """Check if knowledge base file exists"""
    print("\n🔍 Testing knowledge base file...")
    
    import os
    
    if os.path.exists("qa_knowledge_base.txt"):
        with open("qa_knowledge_base.txt", 'r') as f:
            content = f.read()
        
        print(f"✅ Knowledge base file found ({len(content)} characters)")
        return True
    else:
        print("❌ qa_knowledge_base.txt not found!")
        print("   Make sure this file is in the same directory")
        return False


def run_all_tests():
    """Run all verification tests"""
    print("="*60)
    print("🚀 QA KNOWLEDGE ASSISTANT - INSTALLATION VERIFICATION")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Package Imports", test_imports()))
    results.append(("Embedding Model", test_embedding_model()))
    results.append(("ChromaDB", test_chromadb()))
    results.append(("Knowledge File", test_knowledge_file()))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! You're ready to go!")
        print("\nNext steps:")
        print("1. Run the CLI demo: python main.py")
        print("2. Or run the web interface: streamlit run app.py")
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Make sure qa_knowledge_base.txt is in the same folder")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()

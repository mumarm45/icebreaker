def get_sample_documents():
 # Sample data for the lab - AI/ML focused documents
    SAMPLE_DOCUMENTS = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
        "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
        "Natural language processing enables computers to understand, interpret, and generate human language.",
        "Computer vision allows machines to interpret and understand visual information from the world.",
        "Reinforcement learning is a type of machine learning where agents learn to make decisions through rewards and penalties.",
        "Supervised learning uses labeled training data to learn a mapping from inputs to outputs.",
        "Unsupervised learning finds hidden patterns in data without labeled examples.",
        "Transfer learning leverages knowledge from pre-trained models to improve performance on new tasks.",
        "Generative AI can create new content including text, images, code, and more.",
        "Large language models are trained on vast amounts of text data to understand and generate human-like text."
    ]

    # Consistent query examples used throughout the lab
    DEMO_QUERIES = {
        "basic": "What is machine learning?",
        "technical": "neural networks deep learning", 
        "learning_types": "different types of learning",
        "advanced": "How do neural networks work in deep learning?",
        "applications": "What are the applications of AI?",
        "comprehensive": "What are the main approaches to machine learning?",
        "specific": "supervised learning techniques"
    }

    # print(f"📄 Loaded {len(SAMPLE_DOCUMENTS)} sample documents")
    # print(f"🔍 Prepared {len(DEMO_QUERIES)} consistent demo queries")
    # for i, doc in enumerate(SAMPLE_DOCUMENTS[:3], 1):
    #     print(f"{i}. {doc}")
    # print("...")
    
    return SAMPLE_DOCUMENTS, DEMO_QUERIES
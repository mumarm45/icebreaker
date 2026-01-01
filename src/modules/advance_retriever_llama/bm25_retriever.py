from modules.advance_retriever_llama.advance_retriever_llama import lab
from modules.advance_retriever_llama.imports import BM25Retriever

def perform_bm25_retrieval():
    import Stemmer
    
    # Create BM25 retriever with default parameters
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=lab.nodes,
        similarity_top_k=3,
        stemmer=Stemmer.Stemmer("english"),
        language="english"
    )
    
    query = lab.DEMO_QUERIES["technical"]  # "neural networks deep learning"
    nodes = bm25_retriever.retrieve(query)
    
    print(f"Query: {query}")
    print("BM25 analyzes exact keyword matches with sophisticated scoring")
    print(f"Retrieved {len(nodes)} nodes:")
    
    for i, node in enumerate(nodes, 1):
        score = node.score if hasattr(node, 'score') and node.score else 0
        print(f"{i}. BM25 Score: {score:.4f}")
        print(f"   Text: {node.text[:100]}...")
        
        # Highlight which query terms appear in the text
        text_lower = node.text.lower()
        query_terms = query.lower().split()
        found_terms = [term for term in query_terms if term in text_lower]
        if found_terms:
            print(f"   → Found terms: {found_terms}")
        print()
    
    print("BM25 vs TF-IDF Comparison:")
    print("TF-IDF Problem: Linear term frequency scaling")
    print("  Example: 10 occurrences → score of 10, 100 occurrences → score of 100")
    print("BM25 Solution: Saturation function")
    print("  Example: 10 occurrences → high score, 100 occurrences → slightly higher score")
    print()
    print("TF-IDF Problem: No document length consideration")
    print("  Example: Long documents dominate results")
    print("BM25 Solution: Length normalization (b parameter)")
    print("  Example: Scores adjusted based on document length vs. average")
    print()
    print("Key BM25 Parameters:")
    print("- k1 ≈ 1.2: Term frequency saturation (how quickly scores plateau)")
    print("- b ≈ 0.75: Document length normalization (0=none, 1=full)")
    print("- IDF weighting: Rare terms get higher scores")
        
if __name__ == "__main__":
    perform_bm25_retrieval()
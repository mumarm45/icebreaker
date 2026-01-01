from modules.advance_retriever_llama.advance_retriever_llama import lab
from llama_index.core.retrievers import (
    QueryFusionRetriever,
)
def perform_query_fusion_retriever():
    print("=" * 60)
    print("6.1 RECIPROCAL RANK FUSION MODE DEMONSTRATION")
    print("=" * 60)

    # Create QueryFusionRetriever with RRF mode
    base_retriever = lab.vector_index.as_retriever(similarity_top_k=5)

    print("Testing QueryFusionRetriever with reciprocal_rerank mode:")
    print("This demonstrates how RRF works within the query fusion framework")

    # Use the same query for consistency across all fusion modes
    query = lab.DEMO_QUERIES["comprehensive"]  # "What are the main approaches to machine learning?"

    try:
        # Create query fusion retriever with RRF mode
        rrf_query_fusion = QueryFusionRetriever(
            [base_retriever],
            similarity_top_k=3,
            num_queries=3,
            mode="reciprocal_rerank",
            use_async=False,
            verbose=True
        )
        
        print(f"\nQuery: {query}")
        print("QueryFusionRetriever will:")
        print("1. Generate query variations using LLM")
        print("2. Retrieve results for each variation")
        print("3. Apply Reciprocal Rank Fusion")
        
        nodes = rrf_query_fusion.retrieve(query)
        
        print(f"\nRRF Query Fusion Results:")
        for i, node in enumerate(nodes, 1):
            print(f"{i}. Final RRF Score: {node.score:.4f}")
            print(f"   Text: {node.text[:100]}...")
            print()
        
        print("RRF Benefits in Query Fusion Context:")
        print("- Automatically handles query variations of different quality")
        print("- No bias toward queries that return higher raw scores")
        print("- Stable performance across diverse query formulations")
        
    except Exception as e:
        print(f"QueryFusionRetriever error: {e}")

def perform_query_fusion_retriever_manual():
    base_retriever = lab.vector_index.as_retriever(similarity_top_k=5)
    print("Demonstrating RRF concept manually with query variations...")
        
    # Manual demonstration with query variations derived from the main query
    query_variations = [
        lab.DEMO_QUERIES["comprehensive"],  # Original query
        "machine learning approaches and methods",
            "different ML techniques and algorithms"
        ]
        
    print("Manual RRF with Query Variations:")
    all_results = {}
        
    for i, query_var in enumerate(query_variations):
            print(f"\nQuery variation {i+1}: {query_var}")
            nodes = base_retriever.retrieve(query_var)
            
            # Apply RRF scoring
            for rank, node in enumerate(nodes):
                node_id = node.node.node_id
                if node_id not in all_results:
                    all_results[node_id] = {
                        'node': node,
                        'rrf_score': 0,
                        'query_ranks': []
                    }
                
                # Calculate RRF contribution: 1 / (rank + k)
                k = 60  # Standard RRF parameter
                rrf_contribution = 1.0 / (rank + 1 + k)
                all_results[node_id]['rrf_score'] += rrf_contribution
                all_results[node_id]['query_ranks'].append((i, rank + 1))
        
    # Sort by final RRF score
    sorted_results = sorted(
            all_results.values(), 
            key=lambda x: x['rrf_score'], 
            reverse=True
        )
        
    print(f"\nCombined RRF Results (top 3):")
    for i, result in enumerate(sorted_results[:3], 1):
        print(f"{i}. Final RRF Score: {result['rrf_score']:.4f}")
        print(f"   Query ranks: {result['query_ranks']}")
        print(f"   Text: {result['node'].text[:100]}...")
        print()
        
    print("RRF Formula Demonstration:")
    print("For each document: RRF_score = Σ(1 / (rank + 60))")
    print("- Rank 1 in query: 1/(1+60) = 0.0164")
    print("- Rank 2 in query: 1/(2+60) = 0.0161")
    print("- Rank 3 in query: 1/(3+60) = 0.0159")
    print("Documents appearing in multiple queries get higher combined scores")

if __name__ == "__main__":
    perform_query_fusion_retriever()
    perform_query_fusion_retriever_manual()
from modules.advance_retriever_llama.advance_retriever_llama import lab
from llama_index.core.retrievers import (
    QueryFusionRetriever,
)
def perform_query_fusion_retriever():
    
    base_retriever = lab.vector_index.as_retriever(similarity_top_k=5)

    print("Testing QueryFusionRetriever with relative_score mode:")
    print("This mode preserves score magnitudes while normalizing across query variations")

    # Use the same query for consistency across all fusion modes
    query = lab.DEMO_QUERIES["comprehensive"]  # "What are the main approaches to machine learning?"

    try:
        # Create query fusion retriever with relative score mode
        rel_score_fusion = QueryFusionRetriever(
            [base_retriever],
            similarity_top_k=3,
            num_queries=3,
            mode="relative_score",
            use_async=False,
            verbose=False
        )
        
        print(f"\nQuery: {query}")
        print("QueryFusionRetriever with relative_score will:")
        print("1. Generate query variations")
        print("2. Normalize scores within each variation (score/max_score)")
        print("3. Combine normalized scores")
        
        nodes = rel_score_fusion.retrieve(query)
        
        print(f"\nRelative Score Fusion Results:")
        for i, node in enumerate(nodes, 1):
            print(f"{i}. Combined Relative Score: {node.score:.4f}")
            print(f"   Text: {node.text[:100]}...")
            print()
        
        print("Relative Score Benefits in Query Fusion:")
        print("- Preserves confidence information from embedding model")
        print("- Ensures fair contribution from each query variation")
        print("- More interpretable than rank-only methods")
        
    except Exception as e:
     print(f"QueryFusionRetriever error: {e}")
def perform_query_fusion_retriever_manual():
    print("Demonstrating Relative Score concept manually...")
    base_retriever = lab.vector_index.as_retriever(similarity_top_k=5)

    # Manual demonstration with query variations derived from the main query
    query_variations = [
        lab.DEMO_QUERIES["comprehensive"],  # Original query
        "machine learning approaches and methods",
        "different ML techniques and algorithms"
    ]
    
    print("Manual Relative Score Fusion with Query Variations:")
    all_results = {}
    query_max_scores = []
    
    # Step 1: Get results and find max scores for each query
    for i, query_var in enumerate(query_variations):
        print(f"\nQuery variation {i+1}: {query_var}")
        nodes = base_retriever.retrieve(query_var)
        scores = [node.score or 0 for node in nodes]
        max_score = max(scores) if scores else 1.0
        query_max_scores.append(max_score)
        
        print(f"Max score for this query: {max_score:.4f}")
        
        # Store results with normalization info
        for node in nodes:
            node_id = node.node.node_id
            original_score = node.score or 0
            normalized_score = original_score / max_score if max_score > 0 else 0
            
            if node_id not in all_results:
                all_results[node_id] = {
                    'node': node,
                    'combined_score': 0,
                    'contributions': []
                }
            
            all_results[node_id]['combined_score'] += normalized_score
            all_results[node_id]['contributions'].append({
                'query': i,
                'original': original_score,
                'normalized': normalized_score
            })
    
    # Step 2: Sort by combined relative score
    sorted_results = sorted(
        all_results.values(),
        key=lambda x: x['combined_score'],
        reverse=True
    )
    
    print(f"\nCombined Relative Score Results (top 3):")
    for i, result in enumerate(sorted_results[:3], 1):
        print(f"{i}. Combined Score: {result['combined_score']:.4f}")
        print(f"   Score breakdown:")
        for contrib in result['contributions']:
            print(f"     Query {contrib['query']}: {contrib['original']:.3f} → {contrib['normalized']:.3f}")
        print(f"   Text: {result['node'].text[:100]}...")
        print()
    
    print("Relative Score Normalization Process:")
    print("1. For each query variation, find max_score")
    print("2. Normalize: normalized_score = original_score / max_score")
    print("3. Sum normalized scores across all query variations")
    print("4. Documents with consistently high scores across queries win")

if __name__ == "__main__":
    perform_query_fusion_retriever()
    perform_query_fusion_retriever_manual()         
        
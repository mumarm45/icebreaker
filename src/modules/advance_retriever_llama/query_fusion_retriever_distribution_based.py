from modules.advance_retriever_llama.advance_retriever_llama import lab
from llama_index.core.retrievers import (
    QueryFusionRetriever,
)
def perform_query_fusion_retriever():

    base_retriever = lab.vector_index.as_retriever(similarity_top_k=8)

    print("Testing QueryFusionRetriever with dist_based_score mode:")
    print("This mode uses statistical analysis for the most sophisticated score fusion")

    # Use the same query for consistency across all fusion modes
    query = lab.DEMO_QUERIES["comprehensive"]  # "What are the main approaches to machine learning?"

    try:
        # Create query fusion retriever with distribution-based mode
        dist_fusion = QueryFusionRetriever(
            [base_retriever],
            similarity_top_k=3,
            num_queries=3,
            mode="dist_based_score",
            use_async=False,
            verbose=False
        )
        
        print(f"\nQuery: {query}")
        print("QueryFusionRetriever with dist_based_score will:")
        print("1. Generate query variations")
        print("2. Analyze score distributions for each variation")
        print("3. Apply statistical normalization (z-score, percentiles)")
        print("4. Combine with distribution-aware weighting")
        
        nodes = dist_fusion.retrieve(query)
        
        print(f"\nDistribution-Based Fusion Results:")
        for i, node in enumerate(nodes, 1):
            print(f"{i}. Statistically Normalized Score: {node.score:.4f}")
            print(f"   Text: {node.text[:100]}...")
            print()
        
        print("Distribution-Based Benefits in Query Fusion:")
        print("- Accounts for score distribution differences between query variations")
        print("- Statistically robust against outliers and noise")
        print("- Adapts weighting based on query variation reliability")
        
    except Exception as e:
        print(f"QueryFusionRetriever error: {e}")
def perform_query_fusion_retriever_manual():
    print("Demonstrating Distribution-Based concept manually...")
    base_retriever = lab.vector_index.as_retriever(similarity_top_k=8)
    if not SCIPY_AVAILABLE:
        print("⚠️ Full statistical analysis requires scipy")
        
    # Manual demonstration with query variations derived from the main query
    query_variations = [
            lab.DEMO_QUERIES["comprehensive"],  # Original query
            "machine learning approaches and methods",
            "different ML techniques and algorithms"
        ]
        
    print("Manual Distribution-Based Fusion with Query Variations:")
    all_results = {}
    variation_stats = []
        
    # Step 1: Collect results and analyze distributions
    for i, query_var in enumerate(query_variations):
        print(f"\nQuery variation {i+1}: {query_var}")
        nodes = base_retriever.retrieve(query_var)
        scores = [node.score or 0 for node in nodes]
            
        # Calculate distribution statistics
        mean_score = np.mean(scores) if scores else 0
        std_score = np.std(scores) if len(scores) > 1 else 1
        min_score = np.min(scores) if scores else 0
        max_score = np.max(scores) if scores else 1
            
        stats_info = {
                'mean': mean_score,
                'std': std_score,
                'min': min_score,
                'max': max_score,
                'nodes': nodes,
                'scores': scores
            }
        variation_stats.append(stats_info)
            
        print(f"Distribution stats: mean={mean_score:.3f}, std={std_score:.3f}")
        print(f"Score range: [{min_score:.3f}, {max_score:.3f}]")
            
        # Apply z-score normalization
        for node, score in zip(nodes, scores):
            node_id = node.node.node_id
                
            # Z-score normalization
            if std_score > 0:
                z_score = (score - mean_score) / std_score
            else:
                z_score = 0
                
            # Convert to [0,1] using sigmoid
            normalized_score = 1 / (1 + np.exp(-z_score))
                
            if node_id not in all_results:
                all_results[node_id] = {
                        'node': node,
                        'combined_score': 0,
                        'contributions': []
                    }
                
                all_results[node_id]['combined_score'] += normalized_score
                all_results[node_id]['contributions'].append({
                    'query': i,
                    'original': score,
                    'z_score': z_score,
                    'normalized': normalized_score
                })
        
        # Step 2: Sort by combined distribution-based score
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )
        
        print(f"\nCombined Distribution-Based Results (top 3):")
        for i, result in enumerate(sorted_results[:3], 1):
            print(f"{i}. Combined Score: {result['combined_score']:.4f}")
            print(f"   Statistical breakdown:")
            for contrib in result['contributions']:
                print(f"     Query {contrib['query']}: {contrib['original']:.3f} → "
                    f"z={contrib['z_score']:.2f} → {contrib['normalized']:.3f}")
            print(f"   Text: {result['node'].text[:100]}...")
            print()
        
        print("Distribution-Based Process:")
        print("1. Calculate mean and std for each query variation")
        print("2. Z-score normalize: z = (score - mean) / std")
        print("3. Sigmoid transform: normalized = 1 / (1 + exp(-z))")
        print("4. Sum normalized scores across variations")
        print("5. Results reflect statistical significance across all query forms")

    # Show fusion mode comparison summary
    print("\n" + "=" * 60)
    print("FUSION MODES COMPARISON SUMMARY")
    print("=" * 60)
    print("All three modes tested with the same query for direct comparison:")
    print(f"Query: {query}")
    print()
    print("Mode Characteristics:")
    print("• RRF (reciprocal_rerank): Most robust, rank-based, scale-invariant")
    print("• Relative Score: Preserves confidence, normalizes by max score")  
    print("• Distribution-Based: Most sophisticated, statistical normalization")
    print()
    print("Choose based on your use case:")
    print("- Production stability → RRF")
    print("- Score interpretability → Relative Score")

if __name__ == "__main__":
    perform_query_fusion_retriever()
    perform_query_fusion_retriever_manual()
from modules.advance_retriever_llama.advance_retriever_llama import lab
from modules.advance_retriever_llama.imports import VectorIndexRetriever


def perform_vector_index_retrieval():
    # Basic vector retriever
    vector_retriever = VectorIndexRetriever(
        index=lab.vector_index,
        similarity_top_k=3
    )

    # Alternative creation method
    # alt_retriever = lab.vector_index.as_retriever(similarity_top_k=3)

    query = lab.DEMO_QUERIES["basic"]  # "What is machine learning?"
    nodes = vector_retriever.retrieve(query)

    print(f"Query: {query}")
    print(f"Retrieved {len(nodes)} nodes:")
    for i, node in enumerate(nodes, 1):
        print(f"{i}. Score: {node.score:.4f}")
        print(f"   Text: {node.text[:100]}...")
        print()

if __name__ == "__main__":
    perform_vector_index_retrieval()    
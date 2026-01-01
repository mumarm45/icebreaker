from modules.advance_retriever_llama.advance_retriever_llama import lab
from llama_index.core.indices.document_summary import (
    DocumentSummaryIndexLLMRetriever,
    DocumentSummaryIndexEmbeddingRetriever,
)
def perform_document_summary_index_retrieval():
    # LLM-based document summary retriever
    doc_summary_retriever_llm = DocumentSummaryIndexLLMRetriever(
        lab.document_summary_index,
        choice_top_k=3  # Number of documents to select
    )

    # Embedding-based document summary retriever  
    doc_summary_retriever_embedding = DocumentSummaryIndexEmbeddingRetriever(
        lab.document_summary_index,
        similarity_top_k=3  # Number of documents to select
    )

    query = lab.DEMO_QUERIES["learning_types"]  # "different types of learning"

    print(f"Query: {query}")

    print("\nA) LLM-based Document Summary Retriever:")
    print("Uses LLM to select relevant documents based on summaries")
    try:
        nodes_llm = doc_summary_retriever_llm.retrieve(query)
        print(f"Retrieved {len(nodes_llm)} nodes")
        for i, node in enumerate(nodes_llm[:2], 1):
            print(f"{i}. Score: {node.score:.4f}" if hasattr(node, 'score') and node.score else f"{i}. (Document summary)")
            print(f"   Text: {node.text[:80]}...")
            print()
    except Exception as e:
        print(f"LLM-based retrieval demo: {str(e)[:100]}...")

    print("B) Embedding-based Document Summary Retriever:")
    print("Uses vector similarity between query and document summaries")
    try:
        nodes_emb = doc_summary_retriever_embedding.retrieve(query)
        print(f"Retrieved {len(nodes_emb)} nodes")
        for i, node in enumerate(nodes_emb[:2], 1):
            print(f"{i}. Score: {node.score:.4f}" if hasattr(node, 'score') and node.score else f"{i}. (Document summary)")
            print(f"   Text: {node.text[:80]}...")
            print()
    except Exception as e:
        print(f"Embedding-based retrieval demo: {str(e)[:100]}...")

    print("Document Summary Index workflow:")
    print("1. Generates summaries for each document using LLM")
    print("2. Uses summaries to select relevant documents")
    print("3. Returns full content from selected documents")


if __name__ == "__main__":
    perform_document_summary_index_retrieval()    
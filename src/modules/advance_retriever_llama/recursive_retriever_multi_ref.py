from modules.advance_retriever_llama.advance_retriever_llama import lab
from llama_index.core.retrievers import (
    RecursiveRetriever,
)
from llama_index.core import Document, VectorStoreIndex

def perform_recursive_retriever_multi_ref():
    
    # Create documents with references
    docs_with_refs = []
    for i, doc in enumerate(lab.documents):
        # Add reference metadata
        ref_doc = Document(
            text=doc.text,
            metadata={
                "doc_id": f"doc_{i}",
                "references": [f"doc_{j}" for j in range(len(lab.documents)) if j != i][:2]
            }
        )
        docs_with_refs.append(ref_doc)

    # Create index with referenced documents
    ref_index = VectorStoreIndex.from_documents(docs_with_refs)

    # Create retriever mapping
    retriever_dict = {
        f"doc_{i}": ref_index.as_retriever(similarity_top_k=1)
        for i in range(len(docs_with_refs))
    }

    # Base retriever
    base_retriever = ref_index.as_retriever(similarity_top_k=2)

    # Add the root retriever to the dictionary
    retriever_dict["vector"] = base_retriever

    # Recursive retriever
    recursive_retriever = RecursiveRetriever(
        "vector",
        retriever_dict=retriever_dict,
        query_engine_dict={},
        verbose=True
    )

    query = lab.DEMO_QUERIES["applications"]  # "What are the applications of AI?"
    try:
        nodes = recursive_retriever.retrieve(query)
        print(f"Query: {query}")
        print(f"Recursively retrieved {len(nodes)} nodes")
        for i, node in enumerate(nodes[:3], 1):
            print(f"{i}. Score: {node.score:.4f}" if hasattr(node, 'score') and node.score else f"{i}. (Recursive)")
            print(f"   Text: {node.text[:100]}...")
            print()
    except Exception as e:
        print(f"Query: {query}")
        print(f"Recursive retriever demo: {str(e)}")
        print("Note: Recursive retriever requires specific node reference setup")
        
        # Fallback to basic retrieval for demonstration
        print("\nFalling back to basic retrieval demonstration...")
        base_nodes = base_retriever.retrieve(query)
        for i, node in enumerate(base_nodes[:2], 1):
            print(f"{i}. Score: {node.score:.4f}")
            print(f"   Text: {node.text[:100]}...")
            print()


if __name__ == "__main__":
    perform_recursive_retriever_multi_ref()
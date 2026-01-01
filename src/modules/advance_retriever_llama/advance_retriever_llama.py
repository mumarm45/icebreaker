from modules.advance_retriever_llama.data import get_sample_documents
from llama_index.core.node_parser import SentenceSplitter

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
from dotenv import load_dotenv
import os
from llama_index.core import (
    VectorStoreIndex, 
    Document,
    Settings,
    DocumentSummaryIndex,
    KeywordTableIndex
)

class AdvancedRetrieversLab:
    def __init__(self):
        print("🚀 Initializing Advanced Retrievers Lab...")
        
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        Settings.llm = create_anthropic_llm()
        
        SAMPLE_DOCUMENTS, DEMO_QUERIES = get_sample_documents()
        self.documents = [Document(text=text) for text in SAMPLE_DOCUMENTS]
        self.nodes = SentenceSplitter().get_nodes_from_documents(self.documents)
        self.SAMPLE_DOCUMENTS = SAMPLE_DOCUMENTS
        self.DEMO_QUERIES = DEMO_QUERIES
        self.document_summary_index = DocumentSummaryIndex.from_documents(self.documents)
        self.keyword_index = KeywordTableIndex.from_documents(self.documents)
        
        print("📊 Creating indexes...")
        self.vector_index = VectorStoreIndex.from_documents(self.documents)
        
        print("✅ Advanced Retrievers Lab Initialized!")
        print(f"📄 Loaded {len(self.documents)} documents")
        print(f"🔢 Created {len(self.nodes)} nodes")

def create_anthropic_llm(params=None):
    load_dotenv()
    params = params or {}

    api_key = params.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Please set the ANTHROPIC_API_KEY in your .env file")

    model = params.get("model", "claude-3-haiku-20240307")
    max_tokens = params.get("max_tokens", 400)
    temperature = params.get("temperature", 0.7)

    llm = Anthropic(
        api_key=api_key,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    
    return llm


# Initialize the lab
lab = AdvancedRetrieversLab()
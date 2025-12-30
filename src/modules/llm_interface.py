"""Module for interfacing with IBM watsonx.ai LLMs."""

import logging
from typing import Dict, Any, Optional


import config

logger = logging.getLogger(__name__)


def create_embedding_model():
    provider = (getattr(config, "EMBEDDING_PROVIDER", "voyage") or "voyage").lower()

    if provider in {"hf", "huggingface", "hugging_face"}:
        return create_huggingface_embedding()
    elif provider in {"openai"}:
        return create_openai_embedding()
    elif provider in {"voyage", "voyageai", "anthropic"}:
        return create_voyage_embedding()

    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {provider}")


def create_llm(
    temperature: float = config.TEMPERATURE,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
):
    return create_anthropic_llm(
            temperature=temperature,
            max_tokens=max_new_tokens,
        )

def create_voyage_embedding():
    from llama_index.embeddings.voyageai import VoyageEmbedding
    import os

    api_key = config.VOYAGE_API_KEY
    if not api_key:
        raise ValueError("VOYAGE_API_KEY environment variable is not set")

    embedding = VoyageEmbedding(
        voyage_api_key=api_key,
        model_name="voyage-2",
    )
    logger.info("Created Voyage AI Embedding model")
    return embedding


def create_openai_embedding():
    from llama_index.embeddings.openai import OpenAIEmbedding
    import os

    api_key = config.OPENAI_API_KEY
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    embedding = OpenAIEmbedding(api_key=api_key)
    logger.info("Created OpenAI Embedding model")
    return embedding


def create_huggingface_embedding():
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except Exception as e:
        raise ImportError(
            "Hugging Face embeddings require a LlamaIndex Hugging Face embedding integration. "
            "Install llama-index-embeddings-huggingface (or equivalent for your LlamaIndex version)."
        ) from e

    model_id = getattr(config, "HF_EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
    embedding = HuggingFaceEmbedding(model_name=model_id)
    logger.info(f"Created Hugging Face Embedding model: {model_id}")
    return embedding



def create_anthropic_llm(
    temperature: float = config.TEMPERATURE,
    max_tokens: int = config.MAX_NEW_TOKENS,
):
    from llama_index.llms.anthropic import Anthropic

    api_key = config.ANTHROPIC_API_KEY
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set")

    model = config.ANTHROPIC_MODEL_ID

    llm = Anthropic(
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    logger.info(f"Created Anthropic LLM model: {model}")
    return llm

def change_llm_model(new_model_id: str) -> None:
    """Change the LLM model to use.
    
    Args:
        new_model_id: New LLM model ID to use.
    """
    global config
    config.LLM_MODEL_ID = new_model_id
    logger.info(f"Changed LLM model to: {new_model_id}")
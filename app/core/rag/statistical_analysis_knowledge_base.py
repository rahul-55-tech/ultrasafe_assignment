import asyncio
import json
import os
import sys

from app.core.rag.embedding_service import EmbeddingService
from app.core.rag.vector_store import VectorStore
from configurations.config import settings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

KB_NAMESPACE = "analysis-techniques"


def load_knowledge_base(file_path: str):
    """Loads the knowledge base from a JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Knowledge base file not found at {file_path}")
        return []


def prepare_text_for_embedding(technique: dict) -> str:
    """Creates a descriptive text string for a given analysis technique."""
    text = (
        f"Technique: {technique['name']}. "
        f"Description: {technique['description']} "
        f"Data Requirements: {technique['data_requirements']} "
        f"Common Use Cases: {', '.join(technique['use_cases'])}. "
        f"Keywords: {', '.join(technique['keywords'])}."
    )
    return text


async def main():
    logger.info("Starting knowledge base indexing process...")
    embedding_service = EmbeddingService()
    vector_store = VectorStore()
    # Load knowledge base
    kb_file_path = "app/data/statistical_methods.json"
    techniques = load_knowledge_base(kb_file_path)

    if not techniques:
        logger.info("No techniques found to index. Exiting.")
        return
    logger.info(f"Found {len(techniques)} techniques to index.")

    texts_to_embed = []
    vectors_to_upsert = []

    for technique in techniques:
        text_content = prepare_text_for_embedding(technique)
        texts_to_embed.append(text_content)
        # metadata should contain with technique info
        metadata = {
            "technique_id": technique["technique_id"],
            "name": technique["name"],
            "description": technique["description"],
            "data_requirements": technique["data_requirements"],
            "source": "internal_kb",
            "text": text_content,
        }
        vector_id = f"kb_{technique['technique_id']}"
        vectors_to_upsert.append((vector_id, metadata))

    logger.info("Generating embeddings for all techniques...")
    embeddings = embedding_service.get_embeddings(texts_to_embed)
    logger.info("Embeddings generated.")
    # Combine IDs, embeddings, and metadata
    final_vectors = [
        (vector_id, embeddings[i], metadata)
        for i, (vector_id, metadata) in enumerate(vectors_to_upsert)
    ]
    # Upsert to Pinecone in the specified namespace
    logger.info(
        f"Upserting {len(final_vectors)} vectors to Pinecone index "
        f"'{settings.pinecone_index_name}' in namespace '{KB_NAMESPACE}'..."
    )
    vector_store.upsert_vectors(final_vectors, namespace=KB_NAMESPACE)
    print("Upsert complete.")
    index_stats = await vector_store.get_vector_count(namespace=KB_NAMESPACE)
    logger.info("Knowledge base indexing finished successfully!")


if __name__ == "__main__":
    asyncio.run(main())

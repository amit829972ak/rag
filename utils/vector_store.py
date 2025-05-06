import numpy as np
import faiss
import json
import logging
import os
from utils.openai_utils import get_embedding
from utils.db_utils import get_all_knowledge_items, add_knowledge_item

# Set up logging
logger = logging.getLogger(__name__)

def initialize_vector_store(model_version=None):
    """
    Initialize the FAISS vector store with sample knowledge.
    
    Args:
        model_version (str, optional): The specific model version to use for embeddings.
        
    Returns:
        tuple: (FAISS index, list of documents)
    """
    try:
        # Get all knowledge items from the database
        items = get_all_knowledge_items()
        
        # If there are no items in the database, create an empty index
        # (We'll skip sample knowledge creation until an API key is provided)
        if not items:
            # Check if we have an OpenAI API key for adding sample knowledge
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                sample_knowledge = [
                    "Retrieval-Augmented Generation (RAG) is a technique that enhances large language models by incorporating external knowledge.",
                    "RAG combines the generative capabilities of LLMs with information retrieval to produce more accurate, up-to-date, and verifiable responses.",
                    "The key benefit of RAG is that it helps reduce hallucination in AI responses.",
                    "Multimodal AI systems can process and understand multiple types of data inputs, including text, images, audio, and video.",
                    "Image analysis in AI involves techniques like object detection, image classification, and scene understanding."
                ]
                
                # Get embeddings and add to database
                for text in sample_knowledge:
                    try:
                        embedding = get_embedding(text, api_key=api_key, model_version=model_version)
                        add_knowledge_item(text, embedding, "Sample Knowledge")
                    except Exception as e:
                        logger.error(f"Error adding knowledge item: {str(e)}")
                        continue
                
                # Refresh items from database
                items = get_all_knowledge_items()
            else:
                logger.info("No API key available. Skipping sample knowledge creation.")
        
        # If we have no items, create a minimal empty index
        if not items:
            logger.warning("No knowledge items could be loaded. Creating empty index.")
            # Create a minimal empty index with correct dimensions
            dimension = 1536  # Standard OpenAI embedding dimension
            index = faiss.IndexFlatL2(dimension)
            return (index, [])
        
        # Extract embeddings and texts
        embeddings = [item['embedding'] for item in items]
        texts = [item['content'] for item in items]
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        
        # Add vectors to the index
        index.add(embeddings_array)
        
        return (index, texts)
    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}")
        # Create fallback index
        dimension = 1536  # Standard OpenAI embedding dimension
        index = faiss.IndexFlatL2(dimension)
        return (index, [])


def search_vector_store(vector_store, query_embedding, k=3):
    """
    Search the vector store for documents similar to the query.
    
    Args:
        vector_store (tuple): (FAISS index, list of documents)
        query_embedding (list): The query embedding
        k (int): Number of results to return
        
    Returns:
        list: The most relevant documents
    """
    try:
        # Unpack vector store
        index, texts = vector_store
        
        # If there are no documents, return empty result
        if not texts:
            print("Warning: No documents in vector store to search")
            return []
        
        # Convert query embedding to numpy array
        query_embedding_array = np.array([query_embedding]).astype('float32')
        
        # Adjust k if there are fewer documents than requested
        actual_k = min(k, len(texts))
        if actual_k < k:
            print(f"Warning: Only {actual_k} documents in vector store (requested {k})")
        
        # Search for similar vectors
        distances, indices = index.search(query_embedding_array, actual_k)
        
        # Get the corresponding texts
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(texts):  # Ensure index is valid
                results.append({
                    "content": texts[idx],
                    "score": float(distances[0][i])
                })
        
        return results
    except Exception as e:
        print(f"Error searching vector store: {str(e)}")
        return []

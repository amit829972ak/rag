import numpy as np
import faiss
import logging
from utils.gemini_utils import get_embedding as gemini_get_embedding
from utils.openai_utils import get_embedding as openai_get_embedding
from utils.db_utils import add_knowledge_item, get_all_knowledge_items
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_vector_store():
    """
    Initialize the FAISS vector store with sample knowledge.
    
    Returns:
        tuple: (FAISS index, list of documents)
    """
    try:
        # Get all knowledge items from the database
        knowledge_items = get_all_knowledge_items()
        
        if not knowledge_items:
            # If no items in database, try to create sample knowledge
            logger.info("No knowledge items found in database, creating sample knowledge")
            
            # Check for API keys
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            google_api_key = os.environ.get("GOOGLE_API_KEY")
            
            embedding_func = None
            if openai_api_key:
                embedding_func = lambda text: openai_get_embedding(text, openai_api_key)
            elif google_api_key:
                embedding_func = lambda text: gemini_get_embedding(text, google_api_key)
            else:
                logger.warning("No API key available. Skipping sample knowledge creation.")
                return create_empty_index()
            
            # Sample knowledge data
            sample_knowledge = [
                {
                    "content": "The Multimodal RAG Chatbot can process text, images, and documents using AI.",
                    "source": "application_info"
                },
                {
                    "content": "This chatbot uses either OpenAI's GPT models or Google's Gemini models for generating responses.",
                    "source": "application_info"
                },
                {
                    "content": "To use image analysis, upload an image through the sidebar and then ask questions about it.",
                    "source": "how_to_use"
                },
                {
                    "content": "Documents such as PDFs, CSVs, and text files can be uploaded and analyzed by the chatbot.",
                    "source": "how_to_use"
                },
                {
                    "content": "RAG stands for Retrieval-Augmented Generation, which means the chatbot retrieves relevant information before generating a response.",
                    "source": "technical_info"
                }
            ]
            
            # Create embeddings and add to database
            for item in sample_knowledge:
                embedding = embedding_func(item["content"])
                if embedding:
                    add_knowledge_item(item["content"], embedding, item["source"])
                    
            # Refresh knowledge items from database
            knowledge_items = get_all_knowledge_items()
            
        if not knowledge_items:
            logger.warning("No knowledge items could be loaded. Creating empty index.")
            return create_empty_index()
        
        # Extract embeddings and texts from knowledge items
        embeddings = []
        documents = []
        
        for item in knowledge_items:
            if 'embedding' in item and item['embedding']:
                embeddings.append(item['embedding'])
                documents.append(item['content'])
        
        if not embeddings:
            logger.warning("No valid embeddings found. Creating empty index.")
            return create_empty_index()
            
        # Convert to numpy array and normalize
        embeddings_array = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings_array)
        
        # Determine dimensions from the first embedding
        dimension = embeddings_array.shape[1]
        
        # Create and train the index
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_array)
        
        return (index, documents)
        
    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}")
        return create_empty_index()

def create_empty_index():
    """Create an empty FAISS index for when no data is available."""
    # Default dimension for embeddings (varies by model but 1536 is common for OpenAI)
    dimension = 1536
    empty_index = faiss.IndexFlatIP(dimension)
    return (empty_index, [])

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
        if not query_embedding:
            logger.warning("No query embedding provided for search")
            return []
            
        index, documents = vector_store
        
        if len(documents) == 0:
            logger.info("Empty vector store, no results to return")
            return []
            
        # Convert to numpy array and normalize
        query_vector = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_vector)
        
        # Search the index
        similarities, indices = index.search(query_vector, min(k, len(documents)))
        
        # Extract and return the relevant documents
        result = []
        
        for i, idx in enumerate(indices[0]):
            # Break if we encounter a negative index or similarity is too low
            if idx < 0 or similarities[0][i] < 0.5:
                break
                
            # Avoid index out of bounds
            if idx < len(documents):
                result.append(documents[idx])
                
        return result
        
    except Exception as e:
        logger.error(f"Error searching vector store: {str(e)}")
        return []

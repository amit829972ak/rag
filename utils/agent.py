import time
from utils.db_utils import add_message_to_db, get_conversation_messages
from utils.gemini_utils import get_ai_response as gemini_get_response, analyze_image_content as gemini_analyze_image, get_embedding as gemini_get_embedding
from utils.openai_utils import get_ai_response as openai_get_response, analyze_image_content as openai_analyze_image, get_embedding as openai_get_embedding
from utils.gemini_langchain_utils import get_rag_response as gemini_get_rag_response, get_multimodal_response as gemini_get_multimodal_response
from utils.langchain_utils import get_rag_response as openai_get_rag_response, get_multimodal_response as openai_get_multimodal_response
from utils.vector_store import search_vector_store
from utils.document_utils import get_document_summary

class Agent:
    """
    Agent class to handle conversation interactions with various LLMs and multimodal inputs.
    """
    
    def __init__(self):
        """Initialize the Agent with default values."""
        self.conversation_id = None
        self.last_message_id = None
        self.selected_model = "gemini"  # default model
        self.model_version = None  # specific version of the model (e.g., "gpt-4o" or "gemini-1.0-pro")
        self.api_key = None
    
    def set_conversation_id(self, conversation_id):
        """Set the conversation ID for this agent instance."""
        self.conversation_id = conversation_id
        
    def set_model(self, model_name, api_key=None, model_version=None):
        """
        Set which AI model to use.
        
        Args:
            model_name (str): 'gemini' or 'openai'
            api_key (str, optional): API key for the selected model
            model_version (str, optional): Specific version of the model to use
        """
        self.selected_model = model_name
        self.api_key = api_key
        self.model_version = model_version
    
    def get_conversation_history(self, limit=20):
        """
        Get the conversation history.
        
        Args:
            limit (int): Maximum number of messages to retrieve
            
        Returns:
            list: List of message dictionaries
        """
        if not self.conversation_id:
            return []
        
        return get_conversation_messages(self.conversation_id, limit)
    
    def get_chatbot_format_history(self, history=None):
        """
        Format the conversation history for Streamlit chat display.
        
        Args:
            history (list, optional): List of message dictionaries from the database
            
        Returns:
            list: List of (role, content) tuples for Streamlit chat messages
        """
        if history is None:
            history = self.get_conversation_history()
        
        # Format for Streamlit chat
        formatted_messages = []
        
        if history:
            for msg in history:
                role = "user" if msg["role"] == "user" else "assistant"
                content = msg["content"]
                
                # Add the message content
                formatted_messages.append((role, content))
            
        return formatted_messages
    
    def process_query(self, query, vector_store=None, image=None, document_content=None):
        """
        Process a user query and get an AI response.
        
        Args:
            query (str): The user's text query
            vector_store (tuple, optional): FAISS vector store for RAG
            image (Image, optional): PIL Image if an image was uploaded
            document_content (str, optional): Extracted text content from a document
            
        Returns:
            str: The AI's response
        """
        # Check for API key first
        if not self.api_key:
            api_key_missing_message = f"Please enter a valid API key for the {self.selected_model.capitalize()} model in the sidebar settings. You need an API key to interact with the AI models."
            # Save messages to DB for continuity
            add_message_to_db(self.conversation_id, "user", query)
            add_message_to_db(self.conversation_id, "assistant", api_key_missing_message)
            return api_key_missing_message
            
        # Save user message to DB
        add_message_to_db(self.conversation_id, "user", query)
        
        # Determine which model's functions to use
        if self.selected_model == "gemini":
            get_response = gemini_get_response
            analyze_image = gemini_analyze_image
            get_embedding = gemini_get_embedding
            get_rag_response = gemini_get_rag_response
            get_multimodal_response = gemini_get_multimodal_response
        else:  # OpenAI
            get_response = openai_get_response
            analyze_image = openai_analyze_image
            get_embedding = openai_get_embedding
            get_rag_response = openai_get_rag_response
            get_multimodal_response = openai_get_multimodal_response
        
        try:
            # Handle different query types based on context
            response_text = ""
            
            # If both image and document are provided, prioritize image
            if image:
                # Multimodal query with image
                try:
                    # Analyze image
                    image_analysis = analyze_image(image, api_key=self.api_key, model_version=self.model_version)
                    
                    # Get response that incorporates both query and image analysis
                    response_text = get_multimodal_response(query, image_analysis, api_key=self.api_key, model_version=self.model_version)
                except Exception as e:
                    response_text = f"Error processing image: {str(e)}"
            
            elif document_content:
                # Document query
                try:
                    # Create a summary of the document
                    doc_summary = get_document_summary(document_content)
                    
                    # Prepare a context for RAG with the document content
                    context = [{
                        "content": f"Document Analysis: {doc_summary}\n\n{document_content[:5000]}"  # Limit to first 5000 chars
                    }]
                    
                    # Get RAG-enhanced response
                    response_text = get_rag_response(query, context, api_key=self.api_key, model_version=self.model_version)
                except Exception as e:
                    response_text = f"Error processing document query: {str(e)}"
            
            elif vector_store:
                # RAG-enhanced query
                try:
                    # Get embedding for the query
                    query_embedding = get_embedding(query, api_key=self.api_key, model_version=self.model_version)
                    
                    # Search vector store for relevant context
                    search_results = search_vector_store(vector_store, query_embedding, k=3)
                    
                    if search_results:
                        # Get RAG-enhanced response with context
                        response_text = get_rag_response(query, search_results, api_key=self.api_key, model_version=self.model_version)
                    else:
                        # Fallback to regular response if no context found
                        response_text = get_response(query, api_key=self.api_key, model_version=self.model_version)
                except Exception as e:
                    # Fallback to regular response on error
                    print(f"Error in RAG query processing: {e}")
                    response_text = get_response(query, api_key=self.api_key, model_version=self.model_version)
            
            else:
                # Regular text query
                response_text = get_response(query, api_key=self.api_key, model_version=self.model_version)
            
            # Save assistant's response to DB
            message_id = add_message_to_db(self.conversation_id, "assistant", response_text)
            self.last_message_id = message_id
            
            return response_text
            
        except Exception as e:
            error_message = f"Error: {str(e)}"
            # Save error message as assistant response
            add_message_to_db(self.conversation_id, "assistant", error_message)
            return error_message

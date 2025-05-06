import logging
from utils.gemini_utils import get_ai_response as gemini_get_response, analyze_image_content as gemini_analyze_image
from utils.openai_utils import get_ai_response as openai_get_response, analyze_image_content as openai_analyze_image
from utils.db_utils import get_conversation_messages, add_message_to_db
from utils.vector_store import search_vector_store
from utils.gemini_utils import get_embedding as gemini_get_embedding
from utils.openai_utils import get_embedding as openai_get_embedding

class Agent:
    """
    Agent class to handle conversation interactions with various LLMs and multimodal inputs.
    """
    
    def __init__(self):
        """Initialize the Agent with default values."""
        self.conversation_id = None
        self.model_name = "gemini"  # Default model
        self.api_key = None
        self.model_version = None
        self.history = []
        
    def set_conversation_id(self, conversation_id):
        """Set the conversation ID for this agent instance."""
        self.conversation_id = conversation_id
        self.load_history_from_db()
        
    def set_model(self, model_name, api_key=None, model_version=None):
        """
        Set which AI model to use.
        
        Args:
            model_name (str): 'gemini' or 'openai'
            api_key (str, optional): API key for the selected model
            model_version (str, optional): Specific version of the model to use
        """
        self.model_name = model_name
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
            
        chat_history = []
        for msg in history:
            role = msg.role
            content = msg.content
            chat_history.append((role, content))
            
        return chat_history
    
    def load_history_from_db(self):
        """Load conversation history from the database."""
        self.history = self.get_conversation_history()
        
    def add_to_history(self, role, content):
        """
        Add a message to the conversation history and database.
        
        Args:
            role (str): The role of the message sender ('user' or 'assistant')
            content (str): The message content
        """
        if self.conversation_id:
            add_message_to_db(self.conversation_id, role, content)
            
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
        try:
            # Add user query to conversation history
            self.add_to_history('user', query)
            
            # Determine which model functions to use
            get_response = gemini_get_response if self.model_name == "gemini" else openai_get_response
            analyze_image = gemini_analyze_image if self.model_name == "gemini" else openai_analyze_image
            get_embedding = gemini_get_embedding if self.model_name == "gemini" else openai_get_embedding
            
            # Variables to store additional context for the prompt
            image_analysis = None
            relevant_information = None
            
            # Process image if provided
            if image:
                logging.info("Processing image analysis")
                image_analysis = analyze_image(image, self.api_key, self.model_version)
            
            # Retrieve relevant information if there's a vector store
            if vector_store and (query or document_content):
                logging.info("Searching vector store for relevant information")
                text_to_embed = query
                
                # If there's document content, use that as additional context
                if document_content:
                    if len(document_content) > 1000:
                        # For long documents, use the query to find relevant parts
                        text_to_embed = query
                    else:
                        # For shorter documents, use the document content itself
                        text_to_embed = document_content
                        
                # Generate embedding and search vector store
                query_embedding = get_embedding(text_to_embed, self.api_key)
                
                if query_embedding:
                    relevant_items = search_vector_store(vector_store, query_embedding)
                    if relevant_items:
                        relevant_information = "\n\n".join([item for item in relevant_items if item])
            
            # Prepare the system prompt based on the available inputs
            system_prompt = """You are a helpful AI assistant that provides accurate, relevant information. 
            Be concise yet thorough in your answers."""
            
            # Add additional context based on inputs
            message_parts = []
            
            # If there's an image analysis, include it in the context
            if image_analysis:
                message_parts.append(f"Image Analysis:\n{image_analysis}")
                system_prompt += "\nWhen responding to queries about images, refer specifically to visual details you observe."
                
            # If there's document content, include it in the context
            if document_content:
                # Only include a preview of long documents
                doc_preview = document_content
                if len(document_content) > 2000:
                    doc_preview = document_content[:2000] + "... [document continues]"
                
                message_parts.append(f"Document Content:\n{doc_preview}")
                system_prompt += "\nWhen responding to queries about documents, reference specific content from the document."
                
            # If there's relevant information from the vector store, include it
            if relevant_information:
                message_parts.append(f"Relevant Information:\n{relevant_information}")
                system_prompt += "\nUtilize the relevant information provided when answering the query."
                
            # Construct the enhanced prompt with all the context
            enhanced_prompt = query
            if message_parts:
                context = "\n\n".join(message_parts)
                enhanced_prompt = f"{query}\n\nContext:\n{context}"
                
            # Get the AI response with the assembled context
            response = get_response(
                enhanced_prompt, 
                system_prompt=system_prompt, 
                api_key=self.api_key,
                model_version=self.model_version
            )
            
            # Add the response to the conversation history
            self.add_to_history('assistant', response)
            
            return response
            
        except Exception as e:
            error_message = f"Error processing query: {str(e)}"
            logging.error(error_message)
            
            # Add the error message to the conversation history
            self.add_to_history('assistant', f"⚠️ {error_message}")
            
            return f"⚠️ {error_message}\n\nPlease try again or check your API key."

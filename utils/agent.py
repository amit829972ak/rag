from utils.gemini_utils import get_ai_response as gemini_get_ai_response
from utils.openai_utils import get_ai_response as openai_get_ai_response
from utils.gemini_langchain_utils import get_rag_response as gemini_get_rag_response
from utils.gemini_langchain_utils import get_multimodal_response as gemini_get_multimodal_response
from utils.langchain_utils import get_rag_response as openai_get_rag_response
from utils.langchain_utils import get_multimodal_response as openai_get_multimodal_response
from utils.db_utils import add_message_to_db, get_conversation_messages

class Agent:
    """
    Agent class to manage conversation flow and determine the appropriate response strategy.
    """
    
    def __init__(self):
        """
        Initialize the agent with empty conversation history.
        """
        self.conversation_id = None
        self.history = []
    
    def set_conversation_id(self, conversation_id):
        """
        Set the active conversation ID and load its history.
        
        Args:
            conversation_id (int): The conversation ID to set.
        """
        self.conversation_id = conversation_id
        self.load_history_from_db()
    
    def load_history_from_db(self):
        """
        Load conversation history from the database.
        """
        if not self.conversation_id:
            return
        
        # Get messages from database
        messages = get_conversation_messages(self.conversation_id)
        
        # Convert to agent history format
        self.history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
        ]
    
    def add_to_history(self, role, content):
        """
        Add a message to the conversation history and database.
        
        Args:
            role (str): The role of the message sender ('user' or 'assistant').
            content (str): The message content.
        """
        # Add to memory
        self.history.append({"role": role, "content": content})
        
        # Add to database if conversation_id is set
        if self.conversation_id:
            add_message_to_db(self.conversation_id, role, content)
    
    def process_query(self, query=None, image_analysis=None, document_content=None, relevant_info=None, api_key=None, model_name="gemini"):
        """
        Process a user query and determine the appropriate response strategy.
        
        Args:
            query (str, optional): The user's text query.
            image_analysis (str, optional): Analysis of an uploaded image.
            document_content (str, optional): Content of an uploaded document.
            relevant_info (list, optional): Relevant information from the vector store.
            api_key (str, optional): API key for the selected model.
            model_name (str, optional): The model to use ("gemini" or "openai"). Defaults to "gemini".
            
        Returns:
            str: The agent's response.
        """
        # Skip processing if no inputs are provided
        if not query and not image_analysis and not document_content:
            return "I'm not sure what you're asking. Could you please provide a question or upload an image or document?"
        
        # Add user message to history (text, image, or document description)
        if query:
            user_message = query
            
            # Add image context if available
            if image_analysis:
                user_message += f"\n\n[Image context: {image_analysis}]"
                
            # Add document context info if available
            if document_content:
                # Just add a note that document was provided - full content is used in prompt
                user_message += f"\n\n[Document provided]"
            
            self.add_to_history("user", user_message)
        elif image_analysis:
            # Case where only an image was uploaded with no text
            self.add_to_history("user", f"[Image uploaded: {image_analysis}]")
        
        # Select the appropriate functions based on model_name
        if model_name == "gemini":
            get_ai_response = gemini_get_ai_response
            get_rag_response = gemini_get_rag_response
            get_multimodal_response = gemini_get_multimodal_response
        else:  # openai
            get_ai_response = openai_get_ai_response
            get_rag_response = openai_get_rag_response
            get_multimodal_response = openai_get_multimodal_response
            
        # Determine response type based on inputs
        if query and document_content:
            # Document-based query - construct a special prompt with document content
            system_prompt = f"""
            You are a helpful assistant. A user has uploaded a document and asked a question about it.
            
            Document content:
            {document_content}
            
            Based on the document content above, please answer the user's question or help them understand the document.
            Only use information from the document to answer. If the document doesn't contain relevant information,
            acknowledge this and provide a general response.
            """
            
            response = get_ai_response(query, system_prompt=system_prompt, api_key=api_key)
            
        elif query and image_analysis:
            # Both text and image - multimodal response
            response = get_multimodal_response(query, image_analysis, api_key=api_key)
            
        elif query and relevant_info:
            # Text with relevant knowledge - RAG response
            response = get_rag_response(query, relevant_info, api_key=api_key)
            
        else:
            # Simple query or image-only - standard response
            if query:
                prompt = query
            else:
                prompt = f"Describe this image in detail: {image_analysis}"
            
            response = get_ai_response(
                prompt=prompt,
                context=self.history[-5:] if len(self.history) > 0 else None,
                api_key=api_key
            )
        
        # Add agent's response to history
        self.add_to_history("assistant", response)
        
        return response
    
    def get_conversation_context(self):
        """
        Get the current conversation context.
        
        Returns:
            list: The conversation history.
        """
        return self.history
    
    def reset(self):
        """
        Reset the agent's state.
        """
        self.history = []
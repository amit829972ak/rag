import os
import base64
import json
from io import BytesIO
import logging
import google.generativeai as genai

# Gemini model options with expanded set of models
GEMINI_MODELS = {
    "gemini-1.0-pro": {
        "name": "Gemini 1.0 Pro", 
        "description": "Free tier capable model, good balance of capability and cost",
        "api_name": "gemini-pro"
    },
    "gemini-1.0-pro-vision": {
        "name": "Gemini 1.0 Pro Vision", 
        "description": "Free tier model with image understanding",
        "api_name": "gemini-pro-vision"
    },
    "gemini-1.5-pro": {
        "name": "Gemini 1.5 Pro", 
        "description": "Advanced model with improved capabilities",
        "api_name": "gemini-1.5-pro"
    },
    "gemini-1.5-pro-latest": {
        "name": "Gemini 1.5 Pro (Latest)", 
        "description": "Latest version with the most recent improvements",
        "api_name": "gemini-1.5-pro-latest"
    },
    "gemini-1.5-flash": {
        "name": "Gemini 1.5 Flash", 
        "description": "Faster model, good for quick responses",
        "api_name": "gemini-1.5-flash"
    },
    "gemini-1.5-flash-latest": {
        "name": "Gemini 1.5 Flash (Latest)", 
        "description": "Latest version of the faster model",
        "api_name": "gemini-1.5-flash-latest"
    },
    "gemini-2.0-pro": {
        "name": "Gemini 2.0 Pro", 
        "description": "Next-generation model with enhanced reasoning",
        "api_name": "gemini-2.0-pro"
    },
    "gemini-2.0-pro-vision": {
        "name": "Gemini 2.0 Pro Vision", 
        "description": "Advanced vision capabilities with improved understanding",
        "api_name": "gemini-2.0-pro-vision"
    },
    "gemini-2.5-pro": {
        "name": "Gemini 2.5 Pro", 
        "description": "Latest model with state-of-the-art capabilities",
        "api_name": "gemini-2.5-pro"
    },
    "gemini-2.5-flash": {
        "name": "Gemini 2.5 Flash", 
        "description": "Fastest next-gen model for efficient responses",
        "api_name": "gemini-2.5-flash"
    }
}
# Set the default model
DEFAULT_TEXT_MODEL = "gemini-1.0-pro"
GEMINI_VISION_MODELS = [
    "gemini-1.0-pro-vision", 
    "gemini-1.5-pro", 
    "gemini-1.5-flash", 
    "gemini-1.5-pro-latest", 
    "gemini-1.5-flash-latest",
    "gemini-2.0-pro-vision",
    "gemini-2.0-pro",
    "gemini-2.5-pro",
    "gemini-2.5-flash"
]

def get_gemini_client(api_key=None):
    """
    Configure the Gemini API with the provided API key.
    
    Args:
        api_key (str, optional): Google Gemini API key
        
    Returns:
        None: Configuration is set globally
        
    Raises:
        Exception: If API key is invalid or missing
    """
    try:
        # Check for API key in parameters or environment variables
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        
        if not api_key:
            logging.warning("No Google API key provided. Some features may not work.")
            return False
        
        # Configure the Gemini API with the provided key
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        logging.error(f"Error configuring Gemini API: {str(e)}")
        return False

def get_ai_response(prompt, system_prompt=None, context=None, api_key=None, model_version=None):
    """
    Get a response from the Google Gemini API.
    
    Args:
        prompt (str): The user's prompt.
        system_prompt (str, optional): System instructions to guide the AI.
        context (list, optional): Previous conversation context.
        api_key (str, optional): Google Gemini API key.
        model_version (str, optional): The specific model version to use.
        
    Returns:
        str: The AI's response.
        
    Raises:
        Exception: If there's an API error
    """
    try:
        # Configure the client if needed
        client_configured = get_gemini_client(api_key)
        
        if not client_configured:
            return "⚠️ Unable to generate a response: Missing or invalid Google API key. Please provide a valid API key in the sidebar."
        
        # Get the appropriate model
        if not model_version or model_version not in GEMINI_MODELS:
            model_version = DEFAULT_TEXT_MODEL
            
        model_name = GEMINI_MODELS[model_version]["api_name"]
        
        # Create a GenerativeModel object
        model = genai.GenerativeModel(model_name)
        
        # Prepare the chat session
        if context:
            chat = model.start_chat(history=context)
            response = chat.send_message(prompt)
        else:
            # For a single prompt without context
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
            
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
            
            # Construct the full prompt with system instruction if provided
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            response = model.generate_content(
                full_prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
        # Extract and return the text response
        if hasattr(response, 'text'):
            return response.text
        else:
            return response.candidates[0].content.parts[0].text
            
    except Exception as e:
        logging.error(f"Error calling Gemini API: {str(e)}")
        return f"⚠️ Unable to generate a response: {str(e)}\n\nPlease check your API key and try again."

def encode_image_to_base64(image):
    """
    Encode an image to base64 for API transmission.
    
    Args:
        image (PIL.Image): The image to encode.
        
    Returns:
        str: Base64-encoded image string.
    """
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    return encoded_image

def analyze_image_content(image, api_key=None, model_version=None):
    """
    Analyze an image using Google Gemini's vision capabilities.
    
    Args:
        image (PIL.Image): The image to analyze.
        api_key (str, optional): Google Gemini API key.
        model_version (str, optional): The specific model version to use.
        
    Returns:
        str: Analysis of the image content.
    """
    try:
        # Configure the client if needed
        client_configured = get_gemini_client(api_key)
        
        if not client_configured:
            return "⚠️ Unable to analyze image: Missing or invalid Google API key. Please provide a valid API key in the sidebar."
        
        # Get the appropriate model for vision tasks
        vision_model = "gemini-pro-vision"  # Default vision model
        
        # Create a GenerativeModel object
        model = genai.GenerativeModel(vision_model)
        
        # Prepare the prompt for image analysis
        prompt = """
        Analyze this image and describe what you see in detail. 
        Include:
        1. Main subjects or objects
        2. Actions or activities shown
        3. Setting or background
        4. Any text visible in the image
        5. Notable details that might be relevant
        
        Provide a thorough and objective description without making assumptions beyond what is clearly visible.
        """
        
        # Create the request with the image
        image_data = BytesIO()
        image.save(image_data, format="JPEG")
        image_data = image_data.getvalue()
        
        response = model.generate_content([prompt, image_data])
        
        # Extract and return the text response
        if hasattr(response, 'text'):
            return response.text
        else:
            return response.candidates[0].content.parts[0].text
            
    except Exception as e:
        logging.error(f"Error analyzing image with Gemini: {str(e)}")
        return f"⚠️ Unable to analyze image: {str(e)}\n\nPlease check your API key and try again."

def get_embedding(text, api_key=None):
    """
    Get an embedding vector for the given text using Google's embedding model.
    
    Args:
        text (str): The text to embed.
        api_key (str, optional): Google API key.
        
    Returns:
        list: The embedding vector.
    """
    try:
        # Configure the client if needed
        client_configured = get_gemini_client(api_key)
        
        if not client_configured:
            logging.warning("Missing Google API key for embedding generation.")
            return None
            
        # Use Google's embedding model
        embedding_model = "embedding-001"
        
        # Get the embedding
        embedding = genai.embed_content(
            model=embedding_model,
            content=text,
            task_type="retrieval_document"
        )
        
        # Return the embedding values
        if hasattr(embedding, "embedding"):
            return embedding.embedding
        else:
            return embedding.embedding_values
            
    except Exception as e:
        logging.error(f"Error generating embedding with Google API: {str(e)}")
        return None

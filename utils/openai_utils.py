import os
import base64
import json
from io import BytesIO
import logging
import openai

# Define model options with clear descriptions
OPENAI_MODELS = {
    "gpt-3.5-turbo": {
        "name": "GPT-3.5 Turbo",
        "description": "Affordable, fast model (free tier)",
        "max_tokens": 2048
    },
    "gpt-4o": {
        "name": "GPT-4o",
        "description": "Latest and most capable model",
        "max_tokens": 4096
    },
    "gpt-4-turbo": {
        "name": "GPT-4 Turbo",
        "description": "Powerful model with good efficiency",
        "max_tokens": 4096
    }
}

# Set the default model
DEFAULT_MODEL = "gpt-3.5-turbo"

def get_openai_client(api_key=None):
    """
    Get or create an OpenAI client with the provided API key.
    If no key is provided, try to use the environment variable.
    
    Args:
        api_key (str, optional): OpenAI API key
        
    Returns:
        OpenAI: OpenAI client
    """
    try:
        # Check for API key in parameters or environment variables
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        if not api_key:
            logging.warning("No OpenAI API key provided. Some features may not work.")
            return None
            
        # Create a client with the provided API key
        client = openai.OpenAI(api_key=api_key)
        
        # Test with a simple call to ensure the API key works
        client.models.list(limit=1)
        
        return client
    except Exception as e:
        logging.error(f"Error creating OpenAI client: {str(e)}")
        return None

def get_ai_response(prompt, system_prompt=None, context=None, api_key=None, model_version=None):
    """
    Get a response from the OpenAI API.
    
    Args:
        prompt (str): The user's prompt.
        system_prompt (str, optional): System instructions to guide the AI.
        context (list, optional): Previous conversation context.
        api_key (str, optional): OpenAI API key.
        model_version (str, optional): The specific model version to use.
        
    Returns:
        str: The AI's response.
        
    Raises:
        Exception: If there's an API error (rate limit, authentication, etc.)
    """
    try:
        # Get or create a client
        client = get_openai_client(api_key)
        
        if not client:
            return "⚠️ Unable to generate a response: Missing or invalid OpenAI API key. Please provide a valid API key in the sidebar."
        
        # Get the appropriate model
        if not model_version or model_version not in OPENAI_MODELS:
            model_version = DEFAULT_MODEL
        
        # Prepare the messages for the conversation
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        # Add conversation context if provided
        if context:
            for msg in context:
                messages.append(msg)
        
        # Add the current user prompt
        messages.append({"role": "user", "content": prompt})
        
        # Make the API call
        response = client.chat.completions.create(
            model=model_version,
            messages=messages,
            temperature=0.7,
            max_tokens=OPENAI_MODELS[model_version]["max_tokens"],
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0
        )
        
        # Extract and return the response text
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {str(e)}")
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
    Analyze an image using OpenAI's vision capabilities.
    
    Args:
        image (PIL.Image): The image to analyze.
        api_key (str, optional): OpenAI API key.
        model_version (str, optional): The specific model version to use (not used for vision tasks).
        
    Returns:
        str: Analysis of the image content.
    """
    try:
        # Get or create a client
        client = get_openai_client(api_key)
        
        if not client:
            return "⚠️ Unable to analyze image: Missing or invalid OpenAI API key. Please provide a valid API key in the sidebar."
        
        # Encode the image to base64
        base64_image = encode_image_to_base64(image)
        
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
        
        # Make the API call for vision analysis
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",  # Currently, the vision model is separate
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        # Extract and return the response text
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error analyzing image with OpenAI: {str(e)}")
        return f"⚠️ Unable to analyze image: {str(e)}\n\nPlease check your API key and try again."

def get_embedding(text, api_key=None):
    """
    Get an embedding vector for the given text.
    
    Args:
        text (str): The text to embed.
        api_key (str, optional): OpenAI API key.
        
    Returns:
        list: The embedding vector.
    """
    try:
        # Get or create a client
        client = get_openai_client(api_key)
        
        if not client:
            logging.warning("Missing OpenAI API key for embedding generation.")
            return None
        
        # Get the embedding from OpenAI
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        
        # Return the embedding
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Error generating embedding with OpenAI: {str(e)}")
        return None

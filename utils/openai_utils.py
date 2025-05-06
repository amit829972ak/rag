import os
import io
import base64
import json
from openai import OpenAI

# OpenAI model options
OPENAI_MODELS = {
    "gpt-3.5-turbo": {
        "name": "GPT-3.5 Turbo", 
        "description": "Free tier capable model, good balance of capability and cost",
        "api_name": "gpt-3.5-turbo"
    },
    "gpt-4-turbo": {
        "name": "GPT-4 Turbo", 
        "description": "Cutting-edge reasoning, follows instructions better",
        "api_name": "gpt-4-turbo"
    },
    "gpt-4o": {
        "name": "GPT-4o", 
        "description": "Latest model with improved speed and capabilities",
        "api_name": "gpt-4o"
    }
}

# Models for specific capabilities
VISION_MODELS = ["gpt-4-vision-preview", "gpt-4o"]
EMBEDDING_MODEL = "text-embedding-ada-002"

# Default model
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
    # Require API key - no fallback to environment variable
    if not api_key:
        raise Exception("No API key provided. Please provide an OpenAI API key.")
    
    # Add basic format validation (OpenAI keys start with "sk-" and are ~50+ chars)
    if not api_key.startswith("sk-") or len(api_key) < 20:
        raise Exception("Invalid API key format. OpenAI API keys should start with 'sk-' and be at least 20 characters long.")
    
    # Just create the client without making test calls
    return OpenAI(api_key=api_key)

def get_ai_response(prompt, system_prompt=None, context=None, api_key=None, model_version=None):
    """
    Get a response from the OpenAI API.
    
    Args:
        prompt (str): The user's prompt.
        system_prompt (str, optional): System instructions to guide the AI.
        context (list, optional): Previous conversation context.
        api_key (str, optional): OpenAI API key.
        model_version (str, optional): The specific OpenAI model to use.
        
    Returns:
        str: The AI's response.
        
    Raises:
        Exception: If there's an API error (rate limit, authentication, etc.)
    """
    messages = []
    
    # Add system message if provided
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # Add context messages if provided
    if context:
        messages.extend(context)
    
    # Add user prompt
    messages.append({"role": "user", "content": prompt})
    
    # Use the specified model or the default
    selected_model = model_version if model_version and model_version in OPENAI_MODELS else DEFAULT_MODEL
    
    # Get the API name for the selected model
    model = OPENAI_MODELS[selected_model]["api_name"]
    
    try:
        client = get_openai_client(api_key)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        error_msg = str(e)
        if "API key" in error_msg:
            raise Exception("Invalid OpenAI API key. Please check your API key settings and try again.")
        elif "quota" in error_msg.lower() or "rate" in error_msg.lower() or "insufficient_quota" in error_msg:
            raise Exception(f"OpenAI API quota or rate limit reached: {error_msg}")
        else:
            raise Exception(f"Error generating response: {error_msg}")

def encode_image_to_base64(image):
    """
    Encode an image to base64 for API transmission.
    
    Args:
        image (PIL.Image): The image to encode.
        
    Returns:
        str: Base64-encoded image string.
    """
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def analyze_image_content(image, api_key=None, model_version=None):
    """
    Analyze an image using OpenAI's vision capabilities.
    
    Args:
        image (PIL.Image): The image to analyze.
        api_key (str, optional): OpenAI API key.
        model_version (str, optional): The specific OpenAI model to use.
        
    Returns:
        str: Analysis of the image content.
    """
    base64_image = encode_image_to_base64(image)
    
    try:
        # Use a vision-capable model
        # If the selected model supports vision, use it; otherwise use the default vision model
        vision_model = "gpt-4o"  # Newest model with vision capabilities
        
        # If a specific model was requested and it's in our vision models list, use it
        if model_version and model_version in VISION_MODELS:
            # Get the API name for the selected model
            vision_model = OPENAI_MODELS[model_version]["api_name"]
            
        client = get_openai_client(api_key)
        response = client.chat.completions.create(
            model=vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this image in detail. Describe what you see and provide any relevant information about the content."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        error_msg = str(e)
        if "API key" in error_msg:
            raise Exception("Invalid OpenAI API key. Please check your API key settings and try again.")
        elif "quota" in error_msg.lower() or "rate" in error_msg.lower() or "insufficient_quota" in error_msg:
            raise Exception(f"OpenAI API quota or rate limit reached: {error_msg}")
        elif "content" in error_msg.lower() and ("policy" in error_msg.lower() or "filter" in error_msg.lower()):
            raise Exception("The image appears to contain content that violates OpenAI's content policy. Please try a different image.")
        else:
            raise Exception(f"Error analyzing image: {error_msg}")

def get_embedding(text, api_key=None, model_version=None):
    """
    Get an embedding vector for the given text.
    
    Args:
        text (str): The text to embed.
        api_key (str, optional): OpenAI API key.
        model_version (str, optional): Not used for embeddings, but included for API consistency.
        
    Returns:
        list: The embedding vector.
    """
    try:
        client = get_openai_client(api_key)
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"  # We use a specific embedding model regardless of model_version
        )
        return response.data[0].embedding
    except Exception as e:
        error_msg = str(e)
        if "API key" in error_msg:
            raise Exception("Invalid OpenAI API key. Please check your API key settings and try again.")
        elif "quota" in error_msg.lower() or "rate" in error_msg.lower():
            raise Exception(f"OpenAI API quota or rate limit reached: {error_msg}")
        else:
            raise Exception(f"Error generating embedding: {error_msg}")

from PIL import Image
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_image(image):
    """
    Process and optimize an image for analysis.
    
    Args:
        image (PIL.Image): The image to process.
        
    Returns:
        PIL.Image: The processed image.
    """
    try:
        # Resize large images to reduce API costs and improve performance
        max_dim = 1200  # Maximum dimension (width or height)
        
        width, height = image.size
        
        # Only resize if the image is larger than the maximum dimension
        if width > max_dim or height > max_dim:
            # Calculate the scaling factor
            scaling_factor = min(max_dim / width, max_dim / height)
            
            # Calculate new dimensions
            new_width = int(width * scaling_factor)
            new_height = int(height * scaling_factor)
            
            # Resize the image
            image = image.resize((new_width, new_height), Image.LANCZOS)
            
        # Ensure the image is in RGB mode (not RGBA, grayscale, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return image  # Return the original image if processing fails

def convert_image_to_bytes(image):
    """
    Convert a PIL Image to bytes.
    
    Args:
        image (PIL.Image): The image to convert.
        
    Returns:
        bytes: The image as bytes.
    """
    if not image:
        return None
        
    try:
        img_byte_array = io.BytesIO()
        image.save(img_byte_array, format='JPEG')
        return img_byte_array.getvalue()
        
    except Exception as e:
        logger.error(f"Error converting image to bytes: {str(e)}")
        return None

def bytes_to_image(image_bytes):
    """
    Convert bytes to a PIL Image.
    
    Args:
        image_bytes (bytes): The image bytes.
        
    Returns:
        PIL.Image: The reconstructed image.
    """
    if not image_bytes:
        return None
        
    try:
        return Image.open(io.BytesIO(image_bytes))
        
    except Exception as e:
        logger.error(f"Error converting bytes to image: {str(e)}")
        return None

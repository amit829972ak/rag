from PIL import Image, ImageOps
import io

def process_image(image):
    """
    Process and optimize an image for analysis.
    
    Args:
        image (PIL.Image): The image to process.
        
    Returns:
        PIL.Image: The processed image.
    """
    # Ensure image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize image if it's too large (maintain aspect ratio)
    max_dimension = 1024
    if max(image.size) > max_dimension:
        ratio = max_dimension / max(image.size)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.LANCZOS)
    
    return image

def convert_image_to_bytes(image):
    """
    Convert a PIL Image to bytes.
    
    Args:
        image (PIL.Image): The image to convert.
        
    Returns:
        bytes: The image as bytes.
    """
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return buffered.getvalue()

def bytes_to_image(image_bytes):
    """
    Convert bytes to a PIL Image.
    
    Args:
        image_bytes (bytes): The image bytes.
        
    Returns:
        PIL.Image: The reconstructed image.
    """
    if image_bytes is None:
        return None
    
    try:
        return Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        print(f"Error converting bytes to image: {e}")
        return None
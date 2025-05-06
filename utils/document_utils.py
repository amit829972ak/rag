import re
import pandas as pd
import io
from typing import Tuple, Optional, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_document(file, file_type: str) -> Tuple[str, Optional[pd.DataFrame]]:
    """
    Process various document types and extract their content.
    
    Args:
        file: The uploaded file object
        file_type (str): The file extension/type
        
    Returns:
        Tuple[str, Optional[pd.DataFrame]]: Tuple containing extracted text content and dataframe (if applicable)
    """
    try:
        # CSV or TSV processing
        if file_type.lower() in ['.csv', '.tsv']:
            sep = ',' if file_type.lower() == '.csv' else '\t'
            df = pd.read_csv(file, sep=sep)
            
            # Get a string representation for the AI
            content = f"This is a {file_type[1:].upper()} file with {len(df)} rows and {len(df.columns)} columns.\n\n"
            content += f"Column names: {', '.join(df.columns.tolist())}\n\n"
            
            # Add sample data (first few rows)
            sample_size = min(5, len(df))
            if sample_size > 0:
                content += f"First {sample_size} rows:\n"
                content += convert_df_to_csv_string(df.head(sample_size))
                
            return content, df
            
        # Excel processing
        elif file_type.lower() == '.xlsx':
            df = pd.read_excel(file)
            
            # Get a string representation for the AI
            content = f"This is an Excel file with {len(df)} rows and {len(df.columns)} columns.\n\n"
            content += f"Column names: {', '.join(df.columns.tolist())}\n\n"
            
            # Add sample data (first few rows)
            sample_size = min(5, len(df))
            if sample_size > 0:
                content += f"First {sample_size} rows:\n"
                content += convert_df_to_csv_string(df.head(sample_size))
                
            return content, df
            
        # PDF processing
        elif file_type.lower() == '.pdf':
            try:
                # Try to use PyPDF2 for PDF extraction
                from PyPDF2 import PdfReader
                
                bytes_data = file.getvalue()
                reader = PdfReader(io.BytesIO(bytes_data))
                
                content = ""
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    content += page.extract_text() + "\n\n"
                    
                if not content.strip():
                    content = "The PDF appears to contain no extractable text content. It might be scanned or image-based."
                
                return content, None
                
            except Exception as e:
                logger.error(f"Error extracting PDF content: {str(e)}")
                return f"Error processing PDF: {str(e)}", None
                
        # Plain text processing
        elif file_type.lower() == '.txt':
            content = file.getvalue().decode('utf-8')
            return content, None
            
        # Word document processing
        elif file_type.lower() in ['.docx', '.doc']:
            try:
                from docx import Document
                
                bytes_data = file.getvalue()
                doc = Document(io.BytesIO(bytes_data))
                
                content = ""
                for paragraph in doc.paragraphs:
                    content += paragraph.text + "\n"
                    
                return content, None
                
            except Exception as e:
                logger.error(f"Error extracting Word document content: {str(e)}")
                return f"Error processing Word document: {str(e)}", None
                
        else:
            return f"Unsupported file type: {file_type}", None
            
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return f"Error processing document: {str(e)}", None

def get_file_extension(filename: str) -> str:
    """
    Get the lowercase file extension including the dot.
    
    Args:
        filename (str): The filename
        
    Returns:
        str: Lowercase file extension with dot
    """
    pattern = r'(\.[a-zA-Z0-9]+)$'
    match = re.search(pattern, filename.lower())
    return match.group(1) if match else ''

def convert_df_to_csv_string(df: pd.DataFrame) -> str:
    """
    Convert a DataFrame to a CSV string.
    
    Args:
        df (pd.DataFrame): DataFrame to convert
        
    Returns:
        str: CSV string representation
    """
    return df.to_csv(index=False)

def convert_df_to_json_string(df: pd.DataFrame) -> str:
    """
    Convert a DataFrame to a JSON string.
    
    Args:
        df (pd.DataFrame): DataFrame to convert
        
    Returns:
        str: JSON string representation
    """
    return df.to_json(orient='records', indent=2)

def get_document_summary(document_content: str) -> str:
    """
    Create a brief summary of the document content.
    
    Args:
        document_content (str): The document content
        
    Returns:
        str: A brief summary
    """
    if not document_content:
        return "No content available to summarize."
        
    # Get the document size in words and characters
    words = document_content.split()
    word_count = len(words)
    char_count = len(document_content)
    
    # Extract the beginning of the document (first few sentences)
    sentences = document_content.split('.')
    first_sentences = '.'.join(sentences[:3]) + '.'
    
    # Create a brief summary
    summary = f"Document with {word_count} words ({char_count} characters).\n\n"
    summary += f"Beginning: {first_sentences}\n\n"
    
    # Add structure information if it's a structured document
    if "Column names:" in document_content and "rows:" in document_content:
        # Extract column names
        match = re.search(r"Column names: ([^\n]+)", document_content)
        if match:
            columns = match.group(1)
            summary += f"Contains a table with columns: {columns}\n"
    
    return summary

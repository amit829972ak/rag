from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from utils.openai_utils import get_ai_response

def create_rag_chain(api_key=None):
    """
    Create a RAG chain using LangChain.
    
    Args:
        api_key (str, optional): OpenAI API key.
        
    Returns:
        LLMChain: A LangChain chain for RAG responses.
    """
    # Create a custom prompt template for RAG
    rag_template = """
    Answer the query based on the provided context.
    
    Context:
    {context}
    
    Query: {query}
    
    Answer:
    """
    
    # Initialize the prompt with the template
    prompt = PromptTemplate(
        input_variables=["context", "query"],
        template=rag_template
    )
    
    # Initialize the language model
    try:
        # Use the provided API key if available
        if api_key:
            llm = ChatOpenAI(temperature=0.7, max_tokens=800, openai_api_key=api_key)
        else:
            llm = ChatOpenAI(temperature=0.7, max_tokens=800)
        
        # Create the chain
        chain = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=False
        )
        
        return chain
    except Exception as e:
        print(f"Error creating RAG chain: {e}")
        return None

def get_rag_response(query, context, api_key=None):
    """
    Get a RAG-enhanced response using the provided context.
    
    Args:
        query (str): The user's query.
        context (list): List of relevant information from the vector store.
        api_key (str, optional): OpenAI API key.
        
    Returns:
        str: The RAG-enhanced response.
    """
    try:
        # Create the chain with API key
        chain = create_rag_chain(api_key=api_key)
        
        if chain:
            # Format context as a string
            context_str = "\n\n".join([f"- {item['content']}" for item in context])
            
            # Get response from chain
            response = chain.run(context=context_str, query=query)
            return response
        else:
            # Fallback to direct API call
            context_str = "\n\n".join([f"- {item['content']}" for item in context])
            system_prompt = f"""
            You are a helpful assistant. Answer the user's question based on this context:
            
            {context_str}
            
            Only use information from the context to answer. If the context doesn't contain 
            relevant information, acknowledge this and provide a general response.
            """
            
            return get_ai_response(query, system_prompt=system_prompt, api_key=api_key)
    except Exception as e:
        print(f"Error in RAG response: {e}")
        return f"I encountered an issue while retrieving information: {str(e)}. Let me try to answer more generally."

def get_multimodal_response(query, image_analysis, api_key=None):
    """
    Get a response that incorporates both text query and image analysis.
    
    Args:
        query (str): The user's text query.
        image_analysis (str): Analysis of the uploaded image.
        api_key (str, optional): OpenAI API key.
        
    Returns:
        str: Response that considers both the text and image.
    """
    system_prompt = f"""
    You are a helpful assistant with the ability to see images. A user has uploaded an image 
    and asked a question or made a comment about it.
    
    The following is an analysis of the image content:
    {image_analysis}
    
    Based on this image analysis and the user's question, provide a thoughtful and relevant response.
    """
    
    return get_ai_response(query, system_prompt=system_prompt, api_key=api_key)
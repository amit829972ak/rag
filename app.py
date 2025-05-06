import streamlit as st
from PIL import Image
import io
import os
import pandas as pd

from utils.gemini_utils import analyze_image_content as gemini_analyze_image, get_embedding as gemini_get_embedding, get_gemini_client
from utils.openai_utils import analyze_image_content as openai_analyze_image, get_embedding as openai_get_embedding, get_openai_client
from utils.document_utils import process_document, get_file_extension, get_document_summary
from utils.image_utils import process_image, convert_image_to_bytes, bytes_to_image
from utils.agent import Agent
from utils.vector_store import initialize_vector_store, search_vector_store
from utils.db_utils import initialize_database, get_or_create_user, get_or_create_conversation

# Initialize the database
initialize_database()

# Initialize session state
if "agent" not in st.session_state:
    st.session_state.agent = Agent()
    
if "vector_store" not in st.session_state:
    st.session_state.vector_store = initialize_vector_store()
    
if "conversation_id" not in st.session_state:
    # Get or create default user
    user = get_or_create_user()
    # Get or create a conversation
    conversation = get_or_create_conversation(user.id)
    st.session_state.conversation_id = conversation.id
    # Set the conversation ID for the agent
    st.session_state.agent.set_conversation_id(conversation.id)

# App title and description
st.title("Multimodal RAG Chatbot")
st.markdown("Chat with an AI assistant that can see images, understand documents, and access relevant knowledge using either Google Gemini or OpenAI models.")

# Welcome message and instructions
if "visited" not in st.session_state:
    st.session_state.visited = True
    
    welcome_container = st.container()
    with welcome_container:
        st.info("👋 Welcome to the Multimodal RAG Chatbot!")
        
        with st.expander("How to use this chatbot", expanded=True):
            st.markdown("""
            This chatbot can interact with you through text, images, and documents. Here's how to use it:
            
            1. **Select Model**: Choose between Google Gemini or OpenAI models in the sidebar.
            
            2. **API Key**: Enter the appropriate API key for your selected model.
            
            3. **Text Queries**: Simply type your question in the chat input at the bottom of the page.
            
            4. **Image Analysis**: Upload an image from the sidebar, then ask a question about it.
               - Supported formats: jpg, jpeg, png
               
            5. **Document Analysis**: Upload a document from the sidebar, then ask questions about its content.
               - Supported formats: CSV, TSV, XLSX, PDF, TXT, DOCX
               
            6. **Start Over**: Use the "New Conversation" button in the sidebar to start fresh.
            
            Try asking questions about uploaded images or documents, or any general knowledge queries!
            """)
            
        st.markdown("---")

# Sidebar with options
with st.sidebar:
    st.header("Settings")
    
    # Model selection - new feature
    st.subheader("🤖 Select AI Model")
    
    # Initialize model selection in session state if not already there
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "gemini"
    
    # Model selection dropdown
    selected_model = st.radio(
        "Choose which AI model to use:",
        options=["gemini", "openai"],
        format_func=lambda x: "Google Gemini" if x == "gemini" else "OpenAI GPT",
        index=0 if st.session_state.selected_model == "gemini" else 1,
        help="Select which AI provider you want to use for processing your queries."
    )
    
    # Update the session state with the selection
    st.session_state.selected_model = selected_model
    
    # API Key section - with conditional display based on model selection
    if selected_model == "gemini":
        st.subheader("🔑 Google Gemini API Key")
        
        # Import model options from gemini_utils.py
        from utils.gemini_utils import GEMINI_MODELS, DEFAULT_TEXT_MODEL
        
        # Model version selection dropdown for Gemini
        st.subheader("🤖 Select Gemini Model Version")
        
        # Initialize model version in session state if not already there
        if "gemini_model_version" not in st.session_state:
            st.session_state.gemini_model_version = DEFAULT_TEXT_MODEL
        
        # Create a list of model options with formatted display names
        gemini_model_options = []
        model_descriptions = {}
        
        for model_id, model_info in GEMINI_MODELS.items():
            display_name = f"{model_info['name']}"
            gemini_model_options.append(model_id)
            model_descriptions[model_id] = model_info['description']
        
        # Model selection dropdown
        selected_gemini_model = st.selectbox(
            "Choose Gemini model version:",
            options=gemini_model_options,
            format_func=lambda x: f"{GEMINI_MODELS[x]['name']} - {GEMINI_MODELS[x]['description']}",
            index=gemini_model_options.index(st.session_state.gemini_model_version) if st.session_state.gemini_model_version in gemini_model_options else 0,
            help="Select which Gemini model version to use. The free tier model is Gemini 1.0 Pro."
        )
        
        # Update the session state with the selection
        st.session_state.gemini_model_version = selected_gemini_model
        
        # Initialize API key in session state if not already there
        if "google_api_key" not in st.session_state:
            st.session_state.google_api_key = ""
        
        # Get API key with key value from session state as default (no environment variable)
        api_key = st.text_input(
            "Enter your Google Gemini API key", 
            value=st.session_state.google_api_key,
            type="password", 
            help="You must enter a valid API key to use this model.",
            key="gemini_api_key_input"
        )
        
        # Always update the session state with whatever is in the input
        st.session_state.google_api_key = api_key
        
        # Basic client-side validation of API key format
        if api_key:
            if len(api_key) < 20:
                st.warning("⚠️ Google Gemini API key appears to be too short. Keys are typically longer.")
                st.session_state.google_api_key = api_key  # Keep what the user entered
            else:
                st.success("✅ Google Gemini API key format looks valid!")
                st.session_state.google_api_key = api_key  # Keep what the user entered
        else:
            st.warning("⚠️ Please enter your Google Gemini API key to use this model.")
            st.session_state.google_api_key = None
                
        # API key information
        with st.expander("How to get a Google Gemini API key"):
            st.markdown("""
            1. Go to [makersuite.google.com](https://makersuite.google.com)
            2. Sign up or log in to your Google account
            3. Navigate to the API Keys section
            4. Create a new API key
            5. Copy and paste it into the field above
            
            **Note:** Google Gemini API has a free tier with Gemini 1.0 Pro. Check Google's documentation for more details.
            """)
    else:  # OpenAI model selected
        st.subheader("🔑 OpenAI API Key")
        
        # Import model options from openai_utils.py
        from utils.openai_utils import OPENAI_MODELS, DEFAULT_MODEL
        
        # Model version selection dropdown for OpenAI
        st.subheader("🤖 Select OpenAI Model Version")
        
        # Initialize model version in session state if not already there
        if "openai_model_version" not in st.session_state:
            st.session_state.openai_model_version = DEFAULT_MODEL
        
        # Create a list of model options with formatted display names
        openai_model_options = []
        model_descriptions = {}
        
        for model_id, model_info in OPENAI_MODELS.items():
            display_name = f"{model_info['name']}"
            openai_model_options.append(model_id)
            model_descriptions[model_id] = model_info['description']
        
        # Model selection dropdown
        selected_openai_model = st.selectbox(
            "Choose OpenAI model version:",
            options=openai_model_options,
            format_func=lambda x: f"{OPENAI_MODELS[x]['name']} - {OPENAI_MODELS[x]['description']}",
            index=openai_model_options.index(st.session_state.openai_model_version) if st.session_state.openai_model_version in openai_model_options else 0,
            help="Select which OpenAI model version to use. The free tier model is GPT-3.5 Turbo."
        )
        
        # Update the session state with the selection
        st.session_state.openai_model_version = selected_openai_model
        
        # Initialize OpenAI API key in session state if not already there
        if "openai_api_key" not in st.session_state:
            st.session_state.openai_api_key = ""
        
        # Get API key with key value from session state as default (no environment variable)
        api_key = st.text_input(
            "Enter your OpenAI API key", 
            value=st.session_state.openai_api_key,
            type="password", 
            help="You must enter a valid API key to use this model.",
            key="openai_api_key_input"
        )
        
        # Always update the session state with whatever is in the input
        st.session_state.openai_api_key = api_key
        
        # Basic client-side validation of API key format
        if api_key:
            if not api_key.startswith("sk-") or len(api_key) < 20:
                st.warning("⚠️ Invalid API key format. OpenAI API keys should start with 'sk-' and be longer.")
                st.session_state.openai_api_key = api_key  # Keep what the user entered
            else:
                st.success("✅ OpenAI API key format looks valid!")
                st.session_state.openai_api_key = api_key  # Keep what the user entered
        else:
            st.warning("⚠️ Please enter your OpenAI API key to use this model.")
            st.session_state.openai_api_key = None
        
        # API key information
        with st.expander("How to get an OpenAI API key"):
            st.markdown("""
            1. Go to [platform.openai.com](https://platform.openai.com)
            2. Sign up or log in to your OpenAI account
            3. Navigate to API keys in your account settings
            4. Create a new secret key
            5. Copy and paste it into the field above
            
            **Note:** OpenAI offers a free tier with $5 credit for new users. GPT-3.5 Turbo is the most affordable model.
            """)
    
    st.header("Conversation")
    
    if st.button("New Conversation"):
        # Get or create default user
        user = get_or_create_user()
        # Create a new conversation
        conversation = get_or_create_conversation(user.id, "New Conversation")
        st.session_state.conversation_id = conversation.id
        st.session_state.agent.set_conversation_id(conversation.id)
        st.rerun()
    
    st.markdown("---")
    
    # File upload section with tabs for different types
    upload_tab1, upload_tab2 = st.tabs(["Upload Image", "Upload Document"])
    
    with upload_tab1:
        st.markdown("### Upload an Image")
        image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="image_uploader")
        
        if image_file is not None:
            try:
                # Display the uploaded image in the sidebar
                image = Image.open(image_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Process and store the image in session state
                processed_image = process_image(image)
                st.session_state.image = processed_image
                st.session_state.image_bytes = convert_image_to_bytes(processed_image)
                
                # Clear any document data
                st.session_state.document_content = None
                st.session_state.document_df = None
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.warning("Please try uploading a different image file.")
                st.session_state.image = None
                st.session_state.image_bytes = None
    
    with upload_tab2:
        st.markdown("### Upload a Document")
        document_file = st.file_uploader(
            "Choose a document...", 
            type=["csv", "tsv", "xlsx", "pdf", "txt", "docx", "doc"], 
            key="document_uploader"
        )
        
        if document_file is not None:
            try:
                file_extension = get_file_extension(document_file.name)
                supported_extensions = ['.csv', '.tsv', '.xlsx', '.pdf', '.txt', '.docx', '.doc']
                
                if file_extension not in supported_extensions:
                    st.error(f"Unsupported file type: {file_extension}")
                    st.info(f"Supported file types: {', '.join(supported_extensions)}")
                    st.session_state.document_content = None
                    st.session_state.document_df = None
                else:
                    with st.spinner(f"Processing {file_extension} file..."):
                        # Process the document and get content and dataframe (if applicable)
                        document_content, document_df = process_document(document_file, file_extension)
                        
                        if not document_content:
                            st.warning(f"No content could be extracted from {document_file.name}")
                            st.session_state.document_content = None
                            st.session_state.document_df = None
                        else:
                            # Store in session state
                            st.session_state.document_content = document_content
                            st.session_state.document_df = document_df
                            
                            # Display a preview
                            preview_text = document_content[:2000]
                            if len(document_content) > 2000:
                                preview_text += f"...\n\n[Preview showing first 2,000 characters. Full content ({len(document_content):,} characters) will be analyzed.]"
                            st.text_area("Document Preview", preview_text, height=200)
                            
                            # Display info about document size
                            content_size = len(document_content)
                            words = document_content.split()
                            word_count = len(words)
                            st.info(f"Document extracted successfully: {word_count:,} words ({content_size/1000:.1f}KB). Full document content will be analyzed.")
                            
                            # Clear any image data
                            st.session_state.image = None
                            st.session_state.image_bytes = None
                            
                            st.success(f"✅ {document_file.name} processed successfully!")
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
                st.warning("Please try uploading a different document file.")
                st.session_state.document_content = None
                st.session_state.document_df = None

# Update the agent with the selected model, model version, and API key
if st.session_state.selected_model == "gemini":
    st.session_state.agent.set_model(
        "gemini", 
        st.session_state.google_api_key, 
        st.session_state.gemini_model_version if "gemini_model_version" in st.session_state else None
    )
else:
    st.session_state.agent.set_model(
        "openai", 
        st.session_state.openai_api_key, 
        st.session_state.openai_model_version if "openai_model_version" in st.session_state else None
    )

# Display messages from the conversation history
message_history = st.session_state.agent.get_conversation_history()
chat_history = st.session_state.agent.get_chatbot_format_history(message_history)

# Display the chat messages using the Streamlit chat elements
for role, content in chat_history:
    with st.chat_message(role):
        st.markdown(content)

# Chat input for questions - always enabled now (agent handles API key validation)
query = st.chat_input("Ask a question...")

# If a query is submitted, process it
if query:
    # Display the user's message
    with st.chat_message("user"):
        st.markdown(query)
        
    # Display a spinner while processing the response
    with st.spinner("Thinking..."):
        # Process the query through the agent
        response = st.session_state.agent.process_query(
            query=query, 
            vector_store=st.session_state.vector_store,
            image=st.session_state.image if "image" in st.session_state and st.session_state.image else None,
            document_content=st.session_state.document_content if "document_content" in st.session_state and st.session_state.document_content else None
        )
    
    # Display the assistant's response
    with st.chat_message("assistant"):
        st.markdown(response)

# Footer with app information
st.markdown("---")
st.markdown("**Multimodal RAG Chatbot** | A demonstration of retrieval-augmented generation with multimodal capabilities")

# Key capability indicators at the bottom
capability_col1, capability_col2, capability_col3 = st.columns(3)
with capability_col1:
    st.info("💬 Text Processing", icon="💬")
with capability_col2:
    st.info("🖼️ Image Analysis", icon="🖼️")
with capability_col3:
    st.info("📄 Document Processing", icon="📄")

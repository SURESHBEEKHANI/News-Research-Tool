# Import necessary libraries
import os  # This library helps us work with files and folders on the computer.
import streamlit as st  # Streamlit is a framework that helps us create interactive web apps.
from dotenv import load_dotenv  # This library loads environment variables from a .env file, used for configuration.
from langchain.chains import RetrievalQAWithSourcesChain  # A component that allows us to answer questions based on retrieved documents.
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Used to split large texts into smaller, manageable parts.
from langchain.document_loaders import WebBaseLoader  # This is used to fetch text from web pages.
from langchain_groq import ChatGroq  # A specific chat model for generating responses.
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # This creates embeddings for text to help with search and retrieval.
from langchain.vectorstores import FAISS  # A library for storing and searching vector representations of documents.
from langchain_core.prompts import ChatPromptTemplate  # For crafting prompts for chat interactions.
from langchain_core.messages import AIMessage, HumanMessage  # To manage messages exchanged in the chat interface.
import logging  # This library helps track errors and important events in the application.

# Setup logging to capture important information and errors
logging.basicConfig(level=logging.INFO)  # Set the logging level to INFO to record informative messages.

# Load environment variables from the .env file
load_dotenv()  # Load environment variables, which may include sensitive data like API keys.
api_key = os.getenv("GROQ_API_KEY")  # Retrieve the API key for the ChatGroq model.
api_key_google = os.getenv("GOOGLE_API_KEY")  # Retrieve the Google API key (if needed).

# Function to initialize the ChatGroq model
def initialize_model(api_key: str) -> ChatGroq:
    """This function sets up the ChatGroq model using the secret key."""
    return ChatGroq(
        model="mixtral-8x7b-32768",  # Specify the model name for ChatGroq.
        temperature=0,  # A temperature of 0 means responses are more deterministic (less random).
        max_tokens=None,  # No limit on the number of tokens in the response.
        timeout=None,  # No timeout for responses.
        max_retries=5,  # The model will retry up to 5 times if it fails to generate a response.
        api_key=api_key  # Use the provided API key for authentication.
    )

# Function to load data from a given URL
def load_url_data(url: str) -> list:
    """This function gets the text from a web page at the given URL."""
    loader = WebBaseLoader([url])  # Create a loader to fetch text from the provided URL.
    return loader.load()  # Load and return the content from that web page.

# Function to split long documents into smaller chunks
def split_documents(data: list) -> list:
    """This function breaks long pieces of text into smaller, manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Each chunk will have a maximum size of 1500 characters.
        separators=["\n", "\n\n", " ", ""],  # Various ways to split the text.
        chunk_overlap=20  # Allow for 20 characters to overlap between chunks for context.
    )
    return text_splitter.split_documents(data)  # Split the text and return the smaller parts.

# Function to create embeddings and save them in a FAISS index
def create_faiss_index(docs: list, index_path: str) -> None:
    """This function makes embeddings from the text and saves them in a FAISS index."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Create embeddings using the specified model.
    vectorstore = FAISS.from_documents(docs, embeddings)  # Create a searchable index from the documents and their embeddings.
    vectorstore.save_local(index_path)  # Save this index to a local file.
    return vectorstore  # Return the created vectorstore for later use.

# Function to generate a response from the AI based on user input
def get_response(user_query: str, vectorstore, chat_history, llm) -> str:
    """Generates a response from the AI model based on the user query."""
    # Create a template for the chat prompt using the user query
    prompt_template = ChatPromptTemplate(
        messages=[HumanMessage(content=user_query)]  # Include the user's query in the message template.
    )
    
    # Set up the RetrievalQAWithSourcesChain to find answers based on the query
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(k=5))  # Use the vectorstore to retrieve relevant documents.
    result = chain({"question": user_query}, return_only_outputs=True)  # Get the answer from the chain.

    return result["answer"]  # Return the generated answer.

# Main application logic
def main() -> None:
    """This is where our main app runs and does all the work."""
    
    # Custom CSS for modifying the appearance of the app
    st.markdown(
        """
        <style>
        .stTitle { font-size: 40px; }  # Set the font size for the title.
        .stSidebar { font-size: 20px; }  # Set the font size for sidebar elements.
        .stTextInput, .stButton { font-size: 18px; }  # Set font size for input fields and buttons.
        .stTextArea { font-size: 16px; }  # Set font size for text areas.
        </style>
        """,
        unsafe_allow_html=True  # Allow HTML for custom styling.
    )

    st.title("News Research Tool 📈")  # Set the main title of the application.
    st.sidebar.title("News Article URL")  # Title for the sidebar input.

    url = st.sidebar.text_input("Enter the news article URL 📰")  # Prompt user to enter a news article URL.
    process_url_clicked = st.sidebar.button("Process URL 🚀")  # Button to trigger URL processing.
    index_path = "faiss_index"  # Filename for saving the FAISS index.
    main_placeholder = st.empty()  # Create a placeholder for dynamic content in the main area.

    # Initialize chat history and vectorstore in session state
    if "chat_history" not in st.session_state:  # Check if chat history already exists.
        st.session_state.chat_history = []  # If not, create a new list to store chat history.
    if "vectorstore" not in st.session_state:  # Check if vectorstore is initialized.
        st.session_state.vectorstore = None  # Initialize vectorstore to None.

    llm = initialize_model(api_key)  # Initialize the chat model using the provided API key.

    if process_url_clicked:  # If the user clicks the button to process the URL:
        if not url:  # Check if the user did not enter a URL.
            st.sidebar.error("Please provide a URL 😔")  # Show an error message.
        elif not (url.startswith("http://") or url.startswith("https://")):  # Check if the URL format is correct.
            st.sidebar.error("Invalid URL format 🚫. Please include 'http://' or 'https://' 😕.")  # Show an error if the URL is invalid.
        else:  # If the URL is valid:
            try:
                with st.spinner("Loading data from URL... 🤔"):  # Show a loading spinner while data is loading.
                    data = load_url_data(url)  # Load data from the URL.

                with st.spinner("Splitting text into chunks... ✂️"):  # Indicate that the text is being split.
                    docs = split_documents(data)  # Split the text into smaller chunks.

                with st.spinner("Creating embeddings... 🔍"):  # Indicate that embeddings are being created.
                    vectorstore = create_faiss_index(docs, index_path)  # Create embeddings and save the index.
                    st.session_state.vectorstore = vectorstore  # Store the vectorstore in session state.

                st.success("Data processed and index saved! You can now ask questions 😃")  # Notify the user that processing is complete.

            except Exception as e:  # If an error occurs during the process:
                logging.error(f"Error processing URL: {e}")  # Log the error for debugging.
                st.sidebar.error(f"Error processing URL: {e} 😞")  # Show an error message in the sidebar.
                st.error("Error occurred while processing the URL. Please try again. 😓")  # Show a general error message.

    # Display the chat history (both AI and user messages)
    for message in st.session_state.chat_history:  # Iterate over the stored chat messages.
        if isinstance(message, AIMessage):  # Check if the message is from the AI.
            with st.chat_message("AI"):  # Display AI messages in the chat interface.
                st.markdown(message.content)  # Render the AI message content.
        elif isinstance(message, HumanMessage):  # Check if the message is from the user.
            with st.chat_message("Human"):  # Display human messages in the chat interface.
                st.markdown(message.content)  # Render the user message content.

    # Input field for the user to type their message
    user_query = st.chat_input("Type a message...")  # Input field for the user to enter their query.
    if user_query and user_query.strip():  # If the user entered a valid, non-empty query:
        st.session_state.chat_history.append(HumanMessage(content=user_query))  # Append the user query to chat history.
        
        with st.chat_message("Human"):  # Display the user's message in the chat interface.
            st.markdown(user_query)  # Render the user's message.
            
        with st.chat_message("AI"):  # Generate and display the AI response.
            if st.session_state.vectorstore:  # Ensure that the vectorstore has been initialized.
                response = get_response(user_query, st.session_state.vectorstore, st.session_state.chat_history, llm)  # Get the AI's response.
            else:
                response = "Please process a URL first before asking questions."  # Inform the user to process a URL first.

            st.markdown(response)  # Show the AI's response.

        st.session_state.chat_history.append(AIMessage(content=response))  # Add the AI's response to chat history.

# Run the main application function
if __name__ == "__main__":  # Check if this script is being run as the main program.
    main()  # Start the application.

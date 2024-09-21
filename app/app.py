# Import necessary libraries
import os  # This library helps us work with files and folders on the computer.
import streamlit as st  # This library helps us create a web app easily.
from dotenv import load_dotenv  # This library loads secret information from a special file.
from langchain.chains import RetrievalQAWithSourcesChain  # This helps us answer questions and show where the answers come from.
from langchain.text_splitter import RecursiveCharacterTextSplitter  # This helps us break long texts into smaller parts.
from langchain.document_loaders import WebBaseLoader  # This helps us grab text from web pages.
from langchain_groq import ChatGroq  # This is a special model that can chat and answer questions.
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # This creates special math objects called "embeddings" to help find answers.
from langchain.vectorstores import FAISS  # This helps us store and search through our text easily.
import logging  # This helps us keep track of any problems that happen in our code.

# Setup logging
logging.basicConfig(level=logging.INFO)  # We set up logging to show important messages.

# Load environment variables from the .env file
load_dotenv()  # This loads secret information, like API keys, from a special file called .env.
api_key = os.getenv("GROQ_API_KEY")  # We get the secret key we need to use the ChatGroq model.

# Initialize the ChatGroq model
def initialize_model(api_key: str) -> ChatGroq:
    """This function sets up the ChatGroq model using the secret key."""
    return ChatGroq(
        model="mixtral-8x7b-32768",  # This is the name of the chat model we want to use.
        temperature=0,  # This controls how random the responses are (0 means more predictable).
        max_tokens=None,  # This allows us to have as many words as we want in the response.
        timeout=None,  # This means we won't set a time limit for how long to wait for an answer.
        max_retries=5,  # This allows the program to try getting an answer up to 5 times if it fails.
        api_key=api_key  # We use our secret key to access the model.
    )

# Load and process the URL
def load_url_data(url: str) -> list:
    """This function gets the text from a web page at the given URL."""
    loader = WebBaseLoader([url])  # We create a loader to grab text from the URL.
    return loader.load()  # We load and return the content from that web page.

# Split documents into chunks
def split_documents(data: list) -> list:
    """This function breaks long pieces of text into smaller, manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Each piece of text will be no longer than 1500 characters.
        separators=["\n", "\n\n", " ", ""],  # These are the ways we can split the text.
        chunk_overlap=20  # Each piece will overlap with the next by 20 characters.
    )
    return text_splitter.split_documents(data)  # We split the text and return the smaller pieces.

# Create embeddings and save FAISS index
def create_faiss_index(docs: list, index_path: str) -> None:
    """This function makes math objects (embeddings) from the text and saves them."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # We create embeddings using a specific model.
    vectorstore = FAISS.from_documents(docs, embeddings)  # We make a searchable index from the documents and embeddings.
    vectorstore.save_local(index_path)  # We save this index on our computer.

# Main application logic
def main() -> None:
    """This is where our main app runs and does all the work."""
    
    # Custom CSS for font size
    st.markdown(
        """
        <style>
        .stTitle { font-size: 40px; }
        .stSidebar { font-size: 20px; }
        .stTextInput, .stButton { font-size: 18px; }
        .stTextArea { font-size: 16px; }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("News Research Tool 📈")  # We set the title of the app.
    #st.sidebar.image("img/news.png", width=150)  # We show a picture in the sidebar.
    st.sidebar.title("News Article URL")  # We set a title in the sidebar for the URL input.

    url = st.sidebar.text_input("Enter the news article URL 📰")  # We ask the user to input a web page URL.
    process_url_clicked = st.sidebar.button("Process URL 🚀")  # We add a button to start processing the URL.
    index_path = "faiss_index"  # This is the name of the file where we will save our index.
    main_placeholder = st.empty()  # This creates a space where we can show different messages later.

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:  # Check if we already have chat history saved.
        st.session_state.chat_history = []  # If not, we create a new list to store it.

    llm = initialize_model(api_key)  # We set up our chat model using the secret key.

    if process_url_clicked:  # If the button to process the URL is clicked:
        if not url:  # Check if the user didn't enter a URL.
            st.sidebar.error("Please provide a URL 😔")  # Show an error message.
        elif not (url.startswith("http://") or url.startswith("https://")):  # Check if the URL is in the right format.
            st.sidebar.error("Invalid URL format 🚫. Please include 'http://' or 'https://' 😕.")  # Show an error if not valid.
        else:  # If the URL is valid:
            try:
                with st.spinner("Loading data from URL... 🤔"):  # Show a loading message while we wait.
                    data = load_url_data(url)  # Get the text from the web page.

                with st.spinner("Splitting text into chunks... ✂️"):  # Show another loading message.
                    docs = split_documents(data)  # Break the text into smaller pieces.

                with st.spinner("Creating embeddings... 🔍"):  # Show another loading message.
                    create_faiss_index(docs, index_path)  # Create embeddings and save the index.

                st.success("Data processed and index saved! You can now ask questions 😃")  # Show a success message.

            except Exception as e:  # If something goes wrong:
                logging.error(f"Error processing URL: {e}")  # Log the error for us to check later.
                st.sidebar.error(f"Error processing URL: {e} 😞")  # Show an error in the sidebar.
                st.error("Error occurred while processing the URL. Please try again. 😓")  # Show a general error message.

    # Asking questions based on processed data
    query = main_placeholder.chat_input("Ask a question from your news URL:")  # Ask the user for a question about the news.

    if query:  # If the user has asked a question:
        try:
            if os.path.exists(index_path):  # Check if our saved index exists.
                with st.spinner("Loading the FAISS index... 🗂️"):  # Show a loading message.
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Set up embeddings again.
                    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)  # Load our saved index.

                with st.spinner("Retrieving answer... 🤖💡"):  # Show a loading message.
                    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(k=5))  # Set up a question-answering system.
                    result = chain({"question": query}, return_only_outputs=True)  # Get the answer to the user's question.

                # Store the question and answer in chat history
                st.session_state.chat_history.append({"question": query, "answer": result["answer"]})  # Save the question and answer in chat history.

                for entry in st.session_state.chat_history:  # Go through each entry in chat history:
                    st.write(f"**Q:** {entry['question']}")  # Show the question.
                    st.write(f"**A:** {entry['answer']}")  # Show the answer.

                # Display sources if available
                sources = result.get("sources", "")  # Get the sources for the answer.
                if sources:  # If there are sources:
                    st.subheader("Sources 📝:")  # Show a subheader for sources.
                    sources_list = sources.split("\n")  # Split the sources by new lines.
                    for source in sources_list:  # Go through each source:
                        st.write(source)  # Show each source.

            else:
                st.error("No index found. Please process a URL first 😔.")  # Show an error if the index doesn't exist.

        except Exception as e:  # If something goes wrong while retrieving the answer:
            logging.error(f"Error retrieving answer: {e}")  # Log the error.
            st.error(f"Error retrieving answer: {e} 😢")  # Show an error message.

# Run the main application function
if __name__ == "__main__":  # Check if this file is being run directly:
    main()  # Run the main function to start the app.

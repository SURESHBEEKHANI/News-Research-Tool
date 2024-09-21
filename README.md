# News Research Tool

A Streamlit application that allows users to process news articles from URLs, extract content, and ask questions based on the articles. The app utilizes LangChain for document processing, Google’s AI for embeddings, and the ChatGroq model for efficient question answering.

## Features

- Input a news article URL to retrieve and process content.
- Automatically splits the article into manageable text chunks.
- Creates embeddings to enable question-answering capabilities.
- Users can ask questions and receive answers with sourced references.
- Customizable UI with CSS for enhanced user experience.


##Technologies Used
Streamlit: For building the web interface.
LangChain: For document processing and embeddings.
FAISS: For efficient similarity searches.
Google Generative AI: For creating embeddings.
ChatGroq: For providing answers to user queries.
model use : mixtral-8x7b-32768

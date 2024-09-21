# News Research Tool

This Streamlit application enables users to process news articles from URLs, extract content, and ask questions related to the articles. It leverages LangChain for document processing, Google’s AI for creating embeddings, and the ChatGroq model (mixtral-8x7b-32768) for efficient question answering.

## Features

- Input a news article URL to retrieve and process content.
- Automatically splits articles into manageable text chunks.
- Generates embeddings to facilitate question-answering.
- Users can ask questions and receive answers with sourced references.
- Customizable UI using CSS for an enhanced user experience.

## Technologies Used

- **Streamlit**: For building the web interface.
- **LangChain**: For document processing and embeddings.
- **FAISS**: For efficient similarity searches.
- **Google Generative AI**: For creating embeddings.
- **ChatGroq**: For answering user queries using the model mixtral-8x7b-32768.

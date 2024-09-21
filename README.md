# News Research Tool

A Streamlit application that allows users to process news articles from URLs, extract content, and ask questions based on the articles. The app utilizes LangChain for document processing, Google’s AI for embeddings, and the ChatGroq model for efficient question answering.

## Features

- Input a news article URL to retrieve and process content.
- Automatically splits the article into manageable text chunks.
- Creates embeddings to enable question-answering capabilities.
- Users can ask questions and receive answers with sourced references.
- Customizable UI with CSS for enhanced user experience.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/news-research-tool.git
   cd news-research-tool
Install the required packages:

bash
Copy code
pip install -r requirements.txt
Set up your environment variables:

Create a .env file in the project directory.
Add your API keys (e.g., GROQ_API_KEY=your_api_key).
Usage
Run the app:

bash
Copy code
streamlit run app.py
Open your web browser and go to http://localhost:8501.

Enter a news article URL and click "Process URL" to start.

Ask questions related to the processed article.

Technologies Used
Streamlit: For building the web interface.
LangChain: For document processing and embeddings.
FAISS: For efficient similarity searches.
Google Generative AI: For creating embeddings.
ChatGroq: For providing answers to user queries.

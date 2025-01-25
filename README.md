# LangChain Chatbot with OpenAI and Chroma DB

This is an advanced chatbot implementation using LangChain, OpenAI models, and Chroma DB. The chatbot can answer questions based on the LangChain documentation and other embedded knowledge, stored efficiently in a persistent vector database.

## Features

- Uses OpenAI's GPT-3.5-turbo model for generating responses
- Uses OpenAI's text-embedding-ada-002 for creating embeddings
- Loads and processes LangChain documentation automatically
- Persists embeddings in Chroma DB for efficient retrieval
- Implements conversation memory for context-aware responses
- Shows source documents for transparency
- Handles large datasets through efficient text splitting and chunking
- Provides robust error handling and graceful degradation

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

3. Initialize the document store (this will download and process the documentation):
```bash
python data_loader.py
```

4. Run the chatbot:
```bash
python chatbot.py
```

## Usage

1. Start the chatbot by running `python chatbot.py`
2. Type your questions about LangChain, vector stores, embeddings, etc.
3. The bot will provide answers along with relevant source documents
4. Type 'quit' to exit the chatbot

## Data Sources

The chatbot's knowledge comes from:
- LangChain's official documentation
- Custom curated information about:
  - Vector stores and embeddings
  - OpenAI capabilities
  - Chroma DB
  - Document processing
  - LangChain framework

## Models Used
- LLM: gpt-3.5-turbo (OpenAI)
- Embeddings: text-embedding-ada-002 (OpenAI)

## Project Structure
- `chatbot.py`: Main chatbot implementation
- `data_loader.py`: Handles document loading and vector store management
- `requirements.txt`: Project dependencies
- `chroma_db/`: Directory containing the persistent vector store

## Note
The first run will take longer as it needs to download the documentation and create embeddings. Subsequent runs will be faster as they use the persisted vector store.

## Cost Consideration
This implementation uses OpenAI's API, which is a paid service. The costs are associated with:
- Generating embeddings for documents (one-time cost during initialization)
- Creating embeddings for user queries (per query)
- Using the GPT-3.5-turbo model for responses (per query)

Make sure you have a valid OpenAI API key with sufficient credits.
>>>>>>> master

import os
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY in your environment variables")

class DocumentLoader:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model="text-embedding-ada-002"  # OpenAI's most capable embedding model
        )
        
    def load_langchain_docs(self):
        """Load LangChain documentation from key URLs"""
        urls = [
            "https://python.langchain.com/docs/get_started/introduction",
            "https://python.langchain.com/docs/concepts/chat_models/",
            "https://python.langchain.com/docs/tutorials/rag/",
            "https://python.langchain.com/docs/versions/migrating_chains/",
            "https://python.langchain.com/docs/how_to/#agents",
            "https://python.langchain.com/docs/how_to/chatbots_memory/",
            "https://python.langchain.com/docs/versions/v0_3/"
        ]
        
        loader = WebBaseLoader(urls)
        documents = loader.load()
        # Log document content for debugging
        if not documents:
            print("No documents were loaded from the URLs.")
        else:
            for i, doc in enumerate(documents):
                print(f"Document {i + 1}: {doc.page_content[:200]}...")  
                # Print the first 200 characters
    

        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        splits = text_splitter.split_documents(documents)
        print(f"Split into {len(splits)} chunks.")

        
        return splits
    
    def add_local_data(self):
        """Add custom local data"""
        local_data = [
            "LangChain is a framework for developing applications powered by language models. It provides tools and abstractions for working with LLMs.",
            "Vector stores are databases optimized for storing and retrieving vector embeddings. They are essential for semantic search and similarity matching.",
            "OpenAI provides state-of-the-art language models and embedding models for various NLP tasks.",
            "Embeddings are numerical representations of text that capture semantic meaning. They allow for efficient similarity comparisons.",
            "Chroma DB is a vector database designed for storing and retrieving embeddings efficiently. It's open-source and easy to use.",
            "Text splitting is crucial for processing large documents. It involves breaking text into smaller chunks while maintaining context.",
            "Document loaders in LangChain help import various data sources like web pages, PDFs, and databases into a standard format.",
        ]
        
        # Split local data into documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        splits = text_splitter.create_documents(local_data)
        
        return splits
    
    def create_or_load_vectorstore(self):
        """Create a new vector store or load existing one"""
        # Check if vector store exists
        if os.path.exists(self.persist_directory):
            print("Loading existing vector store...")
            vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            print("Creating new vector store...")
            # Load and combine all data
            langchain_docs = self.load_langchain_docs()
            local_data = self.add_local_data()
            print(f"LangChain docs: {len(langchain_docs)} documents loaded.")
            print(f"Local data: {len(local_data)} chunks added.")

            all_splits = langchain_docs + local_data
            print(f"Total splits to add to vector store: {len(all_splits)}")
            
            # Create and persist the vector store
            vectorstore = Chroma.from_documents(
                documents=all_splits,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            vectorstore.persist()
            
        return vectorstore

if __name__ == "__main__":
    # Initialize and run the document loader
    loader = DocumentLoader()
    vectorstore = loader.create_or_load_vectorstore()
    print(f"Vector store created/loaded with {vectorstore._collection.count()} documents")

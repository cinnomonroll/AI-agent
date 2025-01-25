import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from data_loader import DocumentLoader

# Load environment variables
load_dotenv()

# Load OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY in your environment variables")

class Chatbot:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            openai_api_key=OPENAI_API_KEY
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"  # Match this with the chain's output key
        )
        self.qa_chain = None

    def initialize(self):
        """Initialize or load the vector store and create the QA chain."""
        print("Loading document store...")
        loader = DocumentLoader(self.persist_directory)
        vectorstore = loader.create_or_load_vectorstore()
        
        # Check if vector store is empty
        if vectorstore._collection.count() == 0:
            print("Warning: Vector store is empty. Please add documents for better chatbot performance.")

        print("Creating QA chain...")
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
            memory=self.memory,
            return_source_documents=True,
            verbose=True,
            chain_type="stuff",  # Explicitly set chain type
            return_generated_question=True,  # For debugging
        )

    def get_response(self, query):
        """Get a response from the chatbot."""
        if not self.qa_chain:
            raise ValueError("Chatbot not initialized. Call initialize() first.")

        try:
            # Get result from the QA chain
            result = self.qa_chain({"question": query})
            
            # Debug output
            print("\nDebug Information:")
            print(f"Generated Question: {result.get('generated_question', 'N/A')}")
            print(f"Number of source documents: {len(result.get('source_documents', []))}")
            
            # Extract answer and sources
            answer = result.get("answer", "I'm sorry, I couldn't find an answer.")
            sources = result.get("source_documents", [])
            
            # Format sources, including metadata if available
            formatted_sources = []
            for i, doc in enumerate(sources, 1):
                source_text = doc.page_content[:200] + "..."
                if hasattr(doc, 'metadata'):
                    source_text = f"[Source: {doc.metadata.get('source', 'Unknown')}] {source_text}"
                formatted_sources.append(f"{i}. {source_text}")

            if not formatted_sources:
                formatted_sources = ["No sources available."]

            return {
                "answer": answer,
                "sources": formatted_sources
            }

        except Exception as e:
            print(f"\nError in QA Chain: {str(e)}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            return {
                "answer": "I'm sorry, something went wrong. Please try again later.",
                "sources": []
            }

def main():
    """Main function to interact with the chatbot."""
    print("Initializing chatbot...")
    chatbot = Chatbot()

    try:
        chatbot.initialize()
        print("\nChatbot is ready! Type 'quit' to exit.")
        print("You can ask questions about LangChain, vector stores, embeddings, and more.")
    except Exception as e:
        print(f"Failed to initialize chatbot: {str(e)}")
        return

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break

        try:
            response = chatbot.get_response(user_input)
            print(f"\nBot: {response['answer']}")
            
            # Display sources
            if response['sources'] and response['sources'][0] != "No sources available.":
                print("\nSources used:")
                for source in response['sources']:
                    print(source)

        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try asking your question in a different way.")

if __name__ == "__main__":
    main()

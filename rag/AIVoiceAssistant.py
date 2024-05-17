#creating rag system in the Qdrant vector database
from qdrant_client import QdrantClient
from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import ServiceContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext

import warnings
warnings.filterwarnings("ignore")

class AIVoiceAssistant:
    def __init__(self):
        self._qdrant_url = "http://localhost:6333"
        self._client = QdrantClient(url=self._qdrant_url, prefer_grpc=False)
        self._llm = Ollama(model="mistral", request_timeout=150.0)
        #use the local llm embedding and use it to store info in the vector db
        self._service_context = ServiceContext.from_defaults(llm=self._llm, embed_model="local")
        self._index = None
        self._create_kb()
        self._create_chat_engine()

    
    #store the users chat history, information exchanged by the bot and the user is used
    def _create_chat_engine(self):
        memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
        self._chat_engine = self._index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            system_prompt=self._prompt,
        )

    #create knowledge base, use the information, convert it to embeddings and store it in the vector db
    #passing the information about the restaurant menu and it will be used by the LLM to perform similarity search
    def _create_kb(self):
        try:
            reader = SimpleDirectoryReader(
                input_files=[r"C:\Users\gunja\Desktop\LLM_project\rag\restaurant_file_2.txt"]
            )
                #input_files=["/mnt/c/Users/gunja/Desktop/LLM_project/rag/restaurant_file.txt"]
            #)
            documents = reader.load_data()
            vector_store = QdrantVectorStore(client=self._client, collection_name="restaurant_menu")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self._index = VectorStoreIndex.from_documents(
                documents, service_context=self._service_context, storage_context=storage_context
            )
            print("Knowledgebase created successfully!")
        except Exception as e:
            print(f"Error while creating knowledgebase: {e}")

    #use the LLM, the context and the prompt to generate the answer to the users prompts
    def interact_with_llm(self, user_query):
        AgentChatResponse = self._chat_engine.chat(user_query)
        answer = AgentChatResponse.response
        return answer
    

    #system message to better the guide the LLM to perform the task
    @property
    def _prompt(self):
        return """
            You are a professional AI Assistant receptionist working in California's one of the best restaurant called Spice & Syrup.
            Ask these questions always at the beginning [Name and Contact number, what they want to order]. DON'T ASK THESE QUESTIONS 
            IN ONE go and keep the conversation engaging ! Always ask questions one after another!
            
            If you don't know the answer, just say that you don't know. Don't try to make up an answer.
            Provide concise and short answers not more than 10 words, and DON'T CHAT WITH YOURSELF!

            End the conversation with a greeting.
            """
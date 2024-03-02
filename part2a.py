"""
Venturenix LLM workshop 2024-03
Part 2a) Using Google Docs to build a RAG using chromadb
"""
from google.oauth2 import service_account
from llama_index.llms import ChatMessage, MessageRole
from llama_index.callbacks.base import CallbackManager
from llama_index.llms.vertex import Vertex
from llama_index import ServiceContext, StorageContext, VectorStoreIndex, set_global_service_context
from llama_index.vector_stores import ChromaVectorStore
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.readers import GoogleDocsReader
import chromadb
from chromadb.config import Settings
import chainlit as cl

SERVICE_ACCOUNT = 'vtxworkshop.serviceaccount.json'
VERTEX_LLM_MODEL = 'chat-bison'
# HUGGINGFACE_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
# Classic embedding model
HUGGINGFACE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT)
llm = Vertex(model=VERTEX_LLM_MODEL, project=credentials.project_id, credentials=credentials)
embed_model = HuggingFaceEmbedding(model_name=HUGGINGFACE_EMBEDDING_MODEL)

# disable telemetry
db = chromadb.PersistentClient(path="./chroma_db", settings=Settings(anonymized_telemetry=False, allow_reset=True))
# Remove all previous data if exists, remove this if you want to use a previous built index
db.reset()
chroma_collection = db.get_or_create_collection("vtx")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Get the oauth 2.0 json here:
# https://console.cloud.google.com/apis/api/docs.googleapis.com/metrics?
# Save as credentials.json
document_ids = ["1w3JC_VbRydo4ATUozPDcedMLoP4LaWSjb3OEfZnl8ho"]
documents = GoogleDocsReader().load_data(document_ids=document_ids)

@cl.on_chat_start
async def factory():
    service_context = ServiceContext.from_defaults(
        embed_model=embed_model,
        llm=llm,
        num_output=1024,
        callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()])
        )
    set_global_service_context(service_context)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    query_engine = index.as_chat_engine()
    cl.user_session.set("query_engine", query_engine)

@cl.on_message
async def main(message: cl.Message):
    llm = cl.user_session.get("query_engine")
    query_engine = cl.user_session.get("query_engine")
    response = query_engine.chat(message.content)
    response_message = cl.Message(content="")
    response_message.content = response.response
    await response_message.send()
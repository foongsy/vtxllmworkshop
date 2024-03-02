"""
Venturenix LLM workshop 2024-03
Part 2) Using local documents (docx) to build a simple in memory RAG
"""
from google.oauth2 import service_account
from llama_index.llms import ChatMessage, MessageRole
from llama_index.callbacks.base import CallbackManager
from llama_index.llms.vertex import Vertex
from llama_index import ServiceContext, VectorStoreIndex, set_global_service_context
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.readers import SimpleDirectoryReader
import chainlit as cl

SERVICE_ACCOUNT = 'vtxworkshop.serviceaccount.json'
VERTEX_LLM_MODEL = 'chat-bison'
HUGGINGFACE_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT)
llm = Vertex(model=VERTEX_LLM_MODEL, project=credentials.project_id, credentials=credentials)
embed_model = HuggingFaceEmbedding(model_name=HUGGINGFACE_EMBEDDING_MODEL)

@cl.on_chat_start
async def factory():
    service_context = ServiceContext.from_defaults(
        embed_model=embed_model,
        llm=llm,
        num_output=1024,
        callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()])
        )
    set_global_service_context(service_context)
    documents = SimpleDirectoryReader("./data").load_data()
    index = VectorStoreIndex.from_documents(documents)
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
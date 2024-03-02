"""
Venturenix Lab LLM workshop 2024-03
Part 1) Directly use a LLM from GCP without any RAG
"""
from google.oauth2 import service_account
from llama_index.llms import ChatMessage, MessageRole
from llama_index.callbacks.base import CallbackManager
from llama_index.llms.vertex import Vertex
from llama_index import ServiceContext, set_global_service_context
from llama_index.embeddings import HuggingFaceEmbedding

import chainlit as cl

SERVICE_ACCOUNT = 'vtxworkshop.serviceaccount.json'
VERTEX_LLM_MODEL = 'chat-bison'
HUGGINGFACE_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT)
llm = Vertex(model=VERTEX_LLM_MODEL, project=credentials.project_id, credentials=credentials)
# Although embedding model is specify here, it was never used. It's defined to override the default openAI setting
embed_model = HuggingFaceEmbedding(model_name=HUGGINGFACE_EMBEDDING_MODEL)

messages = [
    ChatMessage(role=MessageRole.SYSTEM, content="Reply everything in Traditional Chinese"),
    ChatMessage(role=MessageRole.USER, content="Hello"),
]

@cl.on_chat_start
async def factory():
    service_context = ServiceContext.from_defaults(
    embed_model=embed_model,
    llm=llm,
    chunk_size=512,
    callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]),
    )
    set_global_service_context(service_context)
    cl.user_session.set("query_engine", llm)

@cl.on_message
async def main(message: cl.Message):
    llm = cl.user_session.get("query_engine")  # type: QueryComponent
    in_msg = [
        ChatMessage(role=MessageRole.SYSTEM, content="Reply everything in Traditional Chinese"),
        ChatMessage(role=MessageRole.USER, content=message.content),
    ]
    response = await llm.achat(messages=in_msg)

    response_message = cl.Message(content="")

    response_message.content = response.message.content

    await response_message.send()
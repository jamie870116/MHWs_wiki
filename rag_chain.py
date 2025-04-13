import getpass
import os
from dotenv import load_dotenv

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain import hub
from langchain_community.document_loaders import BSHTMLLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph, MessagesState, END
from typing_extensions import List, TypedDict
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_core.runnables import RunnableConfig
from langchain_core.stores import BaseStore
import uuid
import bs4

load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

llm = init_chat_model("gpt-4o-mini", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)

# === Step 1: Load HTML documents ===
html_dir = "wiki_pages_html"
docs = []

for filename in os.listdir(html_dir):
    if filename.endswith(".html"):
        filepath = os.path.join(html_dir, filename)
        loader = BSHTMLLoader(filepath)
        docs.extend(loader.load())

print(f"✅ Loaded {len(docs)} HTML documents")

# === Step 2: Split into chunks ===
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
print(f"✅ Split into {len(all_splits)} chunks")

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Define application steps
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

def query_or_respond(state: MessagesState):
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tools = ToolNode([retrieve])

def generate(state: MessagesState):
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages
    response = llm.invoke(prompt)
    return {"messages": [response]}

def update_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = (user_id, "memories")
    last_user_msg = next((m.content for m in reversed(state["messages"]) if m.type == "human"), "unknown")
    memory = f"User asked: {last_user_msg}"
    memory_id = str(uuid.uuid4())
    store.put(namespace, memory_id, {"memory": memory})
    return state

graph_builder = StateGraph(MessagesState)
graph_builder.add_node("query_or_respond", query_or_respond)
graph_builder.add_node("tools", tools)
graph_builder.add_node("generate", generate)
graph_builder.add_node("update_memory", update_memory)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges("query_or_respond", tools_condition, {END: END, "tools": "tools"})
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", "update_memory")
graph_builder.add_edge("update_memory", END)

chat_history = InMemorySaver()
in_memory_store = InMemoryStore()
graph = graph_builder.compile(checkpointer=chat_history, store=in_memory_store)

user_id = "1"
config = {"configurable": {"thread_id": "1", "user_id": user_id}}

def call_model(config):
    print("\nType 'exit' to end the conversation.")
    while True:
        user_input = input("\n🧑 You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Exiting.")
            break
        for update in graph.stream({"messages": [{"role": "user", "content": user_input}]}, config, stream_mode="updates"):
            for node_name, output in update.items():
                if isinstance(output, dict) and "messages" in output:
                    print("🤖 AI:", output["messages"][-1].content)

call_model(config)

import os
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAI
from operator import itemgetter
from dotenv import load_dotenv
import bs4
from bs4 import SoupStrainer
loader = WebBaseLoader('https://en.wikipedia.org/wiki/Elon_Musk',
                       bs_kwargs=dict(parse_only=SoupStrainer(class_=('mw-content-ltr mw-parser-output'))))
documents = loader.load()

# Split documents into chunks
recursive = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
chunks = recursive.split_documents(documents)

# Initialize embedding and Qdrant
embed = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')

# Qdrant setup
api_key = os.getenv('qdrant_api_key')
url = 'https://1328bf7c-9693-4c14-a04c-f342030f3b52.us-east4-0.gcp.cloud.qdrant.io:6333'
doc_store = QdrantVectorStore.from_existing_collection(
    embedding=embed,
    url=url,
    api_key=api_key,
    prefer_grpc=True,
    collection_name="Elon Muske"

)
chat_container=st.container()

# Initialize Google LLM
google_api = os.getenv('google_api_key')
llm = GoogleGenerativeAI(model="gemini-1.5-flash-002", google_api_key=google_api)

# Setup
history = []
num_chunks = 5
retriever = doc_store.as_retriever(search_type="mmr", search_kwargs={"k": num_chunks})
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
prompt_str = """
You are a highly knowledgeable and conversational chatbot specializing in providing accurate and insightful information about Elon Musk.
Answer all questions as if you are an expert on his life, career, companies, and achievements.

Context: {context}

Question: {question}
"""
_prompt = ChatPromptTemplate.from_template(prompt_str)

# Chain setup
query_fetcher = itemgetter("question")
history_fetcher = itemgetter("chat_history")
setup = {"question": query_fetcher, "context": query_fetcher | retriever|format_docs}
_chain = setup | _prompt | llm | StrOutputParser()

# Streamlit UI
st.title("Ask Anything About Elon Musk")
# Input field for the user to type a query
query = st.text_input("Please enter a query")
# Only invoke the chain if a query is entered
if query:
    with st.spinner('Processing....'):
      response = _chain.invoke({'question': query})
    with chat_container:
      st.chat_message('user').write(query)
      st.chat_message('ai').write(response)
    # Process the query and get the response
    # Update chat history
    query = f"user_question: {query}"
    response = f"ai_response: {response}"
    history.append((query, response))
else:
    st.write("Please enter a query to interact with the chatbot.")



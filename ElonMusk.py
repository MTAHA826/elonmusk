import os
import streamlit as st
import sounddevice as sd
import torch
import io
import librosa
import numpy as np
import tempfile
import wave
from transformers import pipeline
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

# Load environment variables
load_dotenv()

# Document loader (same as before)
loader = WebBaseLoader(
    'https://en.wikipedia.org/wiki/Elon_Musk',
    bs_kwargs=dict(parse_only=SoupStrainer(class_=('mw-content-ltr mw-parser-output')))
)
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

# Initialize Google LLM
google_api = os.getenv('google_api_key')
llm = GoogleGenerativeAI(model="gemini-1.5-flash-002", google_api_key=google_api)

# Setup retriever and chain
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
setup = {"question": query_fetcher, "context": query_fetcher | retriever | format_docs}
_chain = setup | _prompt | llm | StrOutputParser()

# Audio conversion and transcription
def convert_bytes_to_array(audio_bytes):
    audio_bytes = io.BytesIO(audio_bytes)
    audio, sample_rate = librosa.load(audio_bytes, sr=None)
    return audio

def transcribe_audio(audio_bytes):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    pipe = pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-small",
        chunk_length_s=30,
        device=device,
    )
    audio_array = convert_bytes_to_array(audio_bytes)
    prediction = pipe(audio_array, batch_size=1)["text"]
    return prediction

# Streamlit UI
st.title("Ask Anything About Elon Musk")

# Chat container to display conversation
chat_container = st.container()

# Input Section: Text, Voice, and Upload
col1, col2, col3 = st.columns([4, 1, 1])  # Adjust column widths

with col1:
    query = st.text_input("Please enter a query", label_visibility="collapsed")  # Hides label for cleaner look
with col2:
    send_button = st.button("Send")  # Send button to process text input
with col3:
    record_button = st.button("Record Voice")  # Button to trigger audio recording

uploaded_audio = st.file_uploader("Upload an audio file for transcription", type=["wav", "mp3"])

# Process Text Query
if send_button and query:
    with st.spinner("Processing your text query..."):
        response = _chain.invoke({'question': query})  # Generate response
    with chat_container:
        st.chat_message('user').write(query)
        st.chat_message('ai').write(response)

# Process Uploaded Audio
if uploaded_audio:
    with st.spinner("Transcribing uploaded audio..."):
        transcribed_audio = transcribe_audio(uploaded_audio.getvalue())  # Transcribe uploaded audio
        st.write(f"Transcription: {transcribed_audio}")
    with st.spinner("Processing your transcribed query..."):
        response = _chain.invoke({'question': transcribed_audio})  # Generate response
    with chat_container:
        st.chat_message('user').write(f"(Audio) {transcribed_audio}")
        st.chat_message('ai').write(response)

# Process Voice Recording
if record_button:
    def record_audio(duration=5, samplerate=16000):
        st.info(f"Recording for {duration} seconds...")
        audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype="int16")
        sd.wait()  # Wait until recording is finished
        return audio, samplerate

    def save_audio(audio, samplerate):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            with wave.open(tmpfile.name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(samplerate)
                wf.writeframes(audio.tobytes())
            return tmpfile.name

    audio, samplerate = record_audio()  # Record audio
    audio_path = save_audio(audio, samplerate)  # Save as WAV
    with st.spinner("Transcribing your voice..."):
        transcribed_audio = transcribe_audio(open(audio_path, "rb").read())  # Transcribe audio
        st.write(f"Transcription: {transcribed_audio}")
    with st.spinner("Processing your voice query..."):
        response = _chain.invoke({'question': transcribed_audio})  # Generate response
    with chat_container:
        st.chat_message('user').write(f"(Voice) {transcribed_audio}")
        st.chat_message('ai').write(response)

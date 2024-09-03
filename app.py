import streamlit as st
import os
import pickle
from pathlib import Path
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack import Pipeline
from typing import List

# Define paths
preprocessed_data_path = 'preprocessed_data.pkl'

# Check API key setup
if "api_key_set" not in st.session_state:
    api_key = os.environ.get("OPENAI_API_KEY")
    st.session_state.api_key_set = api_key is not None
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

if not st.session_state.api_key_set:
    openai_api_key = st.text_input("Please enter your OpenAI API key:", type="password")
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        st.session_state.api_key_set = True
        st.success("API key set successfully!")
        st.rerun()
    else:
        st.warning("OpenAI API key is required to run this application.")
        st.stop()

# Load preprocessed data
def load_preprocessed_data(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)

# Initialize document store and components
document_store = InMemoryDocumentStore()

# Load preprocessed documents with embeddings
docs_with_embeddings = load_preprocessed_data(preprocessed_data_path)
document_store.write_documents(docs_with_embeddings["documents"])

text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
retriever = InMemoryEmbeddingRetriever(document_store)
template = """
Consider the context given in the question, please generate a wild conspiracy theory in two paragraphs that could only occur based on the following news articles of events that occurred in San Francisco in the last year (and at the end of the answer, give me the part of the theory where you used the documents and the title of the documents you used):

The context for the question are the following news articles of events that occurred in San Francisco in the last year, consider them as highly important to properly answer the question:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""
prompt_builder = PromptBuilder(template=template)
basic_rag_pipeline = Pipeline()
basic_rag_pipeline.add_component("text_embedder", text_embedder)
basic_rag_pipeline.add_component("retriever", retriever)
basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
basic_rag_pipeline.add_component("llm", OpenAIGenerator(model="gpt-3.5-turbo"))
basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
basic_rag_pipeline.connect("retriever", "prompt_builder.documents")
basic_rag_pipeline.connect("prompt_builder", "llm")

# Streamlit code to interact with the pipeline
st.title("👁️‍🗨️🌉 SF Conspiracy Theory Generator ")
st.write(
    "This is a chatbot powered by OpenAI's GPT-3.5-Turbo, orchestrated by Haystack 2.0 to generate conspiracy theories about the city of San Francisco."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    result = basic_rag_pipeline.run({
        "text_embedder": {"text": prompt},
        "prompt_builder": {"question": prompt}
    })

    documents = result.get("llm")
    if documents:
        with st.chat_message("assistant"):
            response = documents["replies"][0]
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.error("No documents found. Please check the pipeline configuration.")
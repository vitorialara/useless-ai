import streamlit as st
import os
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack import Pipeline

# Set the OpenAI API key directly
os.environ["OPENAI_API_KEY"] = "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"

# Import documents from local files.
document_store = InMemoryDocumentStore()

# Path to the directory containing the .txt files
folder_path = 'scraped_articles'

# List to store the Document objects
docs = []

# Loop through each file in the directory
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        
        # Read the contents of the file
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            title = lines[0].strip()  # Title is on the first line
            summary = lines[2].strip()  # Summary is on the third line
            
            # Create a Document object
            doc = Document(content=summary, meta={"title": title})
            docs.append(doc)

# Initialize document embedder
doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
doc_embedder.warm_up()

# Embed documents and write to the document store
docs_with_embeddings = doc_embedder.run(docs)
document_store.write_documents(docs_with_embeddings["documents"])

# Initialize retriever and text embedder
text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
retriever = InMemoryEmbeddingRetriever(document_store)

# Define custom prompt
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

# Set up the pipeline
basic_rag_pipeline = Pipeline()
basic_rag_pipeline.add_component("text_embedder", text_embedder)
basic_rag_pipeline.add_component("retriever", retriever)
basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
basic_rag_pipeline.add_component("llm", OpenAIGenerator(model="gpt-3.5-turbo"))

# Connect the components
basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
basic_rag_pipeline.connect("retriever", "prompt_builder.documents")
basic_rag_pipeline.connect("prompt_builder", "llm")

# Streamlit code to interact with the pipeline
st.title("👁️‍🗨️🌉 SF Conspiracy Theory Generator ")
st.write(
    "This is a chatbot powered by OpenAI's GPT-3.5-Turbo, orchestrated by Haystack 2.0 to generate conspiracy theories about the city of San Francisco."
)

# Create a session state variable to store the chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the existing chat messages via `st.chat_message`
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Create a chat input field to allow the user to enter a message
if prompt := st.chat_input("What is up?"):

    # Store and display the current prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run the pipeline with correctly formatted input
    result = basic_rag_pipeline.run({
        "text_embedder": {"text": prompt},
        "prompt_builder": {"question": prompt}
    })

    documents = result.get("llm")

    if documents:
        with st.chat_message("assistant"):
            response = documents["replies"][0]  # Assuming you want the first reply
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.error("No documents found. Please check the pipeline configuration.")
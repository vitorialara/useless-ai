import streamlit as st
import os
import pickle
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack import Pipeline
from dotenv import load_dotenv
import openai
import requests
from io import BytesIO
from PIL import Image

# Load environment variables from .env file
load_dotenv()

# Fetch OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Check if the OpenAI API key is set
if not OPENAI_API_KEY:
    st.error("OpenAI API key is not set. Please set it in the .env file.")
    st.stop()

# Set the OpenAI API key
openai.api_key = OPENAI_API_KEY

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

# Initialize text embedder (for sentence-level embeddings)
text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

# Initialize retriever, without embedding_model, as it is not supported in your version
retriever = InMemoryEmbeddingRetriever(document_store=document_store)

# Template for conspiracy theory generation
template = """
Using the context provided, generate a creative and outlandish conspiracy theory in a paragraph (Your total response (including title and references) MUST be 800 characters or less). 
Your theory should be based on the following news articles about events in San Francisco from the last year.
Make sure the theory is wild but still incorporates specific details news articles.
At the beginning of your answer, provide a catchy title for the theory in 7 or fewer words.
Bold this title using Markdown.
Don't use the word "Conspiracy" in your title.
In addition, center the title and have it on its own unique line.
Don't have a colon at the end of the title
At the end of your answer, provide the titles of the articles you used in a bulleted list.

Context:
Below are the key news articles that occurred in San Francisco over the last year. 
Use them to inform your answer:
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

    # Step 1: Embed the query using the text embedder
    result = basic_rag_pipeline.run({
        "text_embedder": {"text": prompt},  # Embed the input prompt
        "prompt_builder": {"question": prompt}
    })

    documents = result.get("llm")
    if documents:
        with st.chat_message("assistant"):
            response = documents["replies"][0]
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Function to generate an image using DALL-E API based on the conspiracy theory
            def generate_image(prompt: str) -> str:
                try:
                    # Correct call to openai.images.generate
                    response = openai.images.generate(
                        prompt=prompt,  # Use the conspiracy theory text as a prompt
                        n=1,
                        size="512x512"
                    )
                    # Access the first image URL correctly from the response object
                    image_url = response.data[0].url  # Access as an object, not as a dict
                    return image_url
                except Exception as e:
                    st.error(f"Error generating image: {e}")
                    return None

            # Function to display the image from a URL
            def display_image(image_url: str, file_path: str, save: bool = False):
                try:
                    response = requests.get(image_url)
                    img = Image.open(BytesIO(response.content))

                    if save:
                        if not os.path.exists(file_path):
                            os.makedirs(file_path)
                        img.save(os.path.join(file_path, "generated_image.png"))
                    
                    st.image(img, caption="Generated by DALL-E", use_column_width=True)
                except Exception as e:
                    st.error(f"Error displaying image: {e}")

            # Now generate an image based on the generated conspiracy theory text
            image_url = generate_image(response)  # Use the conspiracy theory text as the image generation prompt
            if image_url:
                display_image(image_url, file_path="images", save=True)
            else:
                st.error("Failed to generate image.")

    else:
        st.error("No documents found. Please check the pipeline configuration.")
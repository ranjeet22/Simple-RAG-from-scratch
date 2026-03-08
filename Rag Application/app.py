import streamlit as st
import ollama
import numpy as np

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

def cosine_similarity(a, b):
    dot_product = sum([x*y for x,y in zip(a,b)])
    norm_a = sum([x**2 for x in a]) ** 0.5
    norm_b = sum([x**2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)


def load_dataset(file):
    dataset = file.read().decode("utf-8").splitlines()
    return dataset


def build_vector_db(dataset):
    vector_db = []
    progress = st.progress(0)

    for i, chunk in enumerate(dataset):
        if chunk.strip() == "":
            continue

        embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
        vector_db.append((chunk, embedding))
        progress.progress((i+1)/len(dataset))

    return vector_db


def retrieve(query, top_n=3):
  query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]

  # temporary list to store (chunk, similarity) pairs
  similarities = []
  for chunk, embedding in st.session_state.vector_db:
    similarity = cosine_similarity(query_embedding, embedding)
    similarities.append((chunk, similarity))

  # sort by similarity in descending order, because higher similarity means more relevant chunks
  similarities.sort(key=lambda x: x[1], reverse=True)
  
  # finally, return the top N most relevant chunks
  return similarities[:top_n]


# STREAMLIT UI

st.set_page_config(page_title="RAG Chatbot")
st.title("Simple RAG Chatbot 🤖")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (.txt)", type="txt")

if uploaded_file and "vector_db" not in st.session_state:
    with st.spinner("Uploading your dataset and building knowledge base..."):
        dataset = load_dataset(uploaded_file)
        st.session_state.vector_db = build_vector_db(dataset)

    st.success("Dataset uploaded successfully! You can now ask questions.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if "vector_db" in st.session_state:
    if input_query := st.chat_input("Ask something about the dataset..."):
        st.session_state.messages.append({"role":"user","content":input_query})
        with st.chat_message("user"):
            st.markdown(input_query)

        retrieved_knowledge = retrieve(input_query)

        instruction_prompt = f'''You are a helpful chatbot.
        Use only the following pieces of context to answer the question. Don't make up any new information: {'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}'''

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            stream = ollama.chat(
                model=LANGUAGE_MODEL,
                messages=[
                    {'role': 'system', 'content': instruction_prompt},
                    {'role': 'user', 'content': input_query},
                ],
                stream=True,
            )

            for chunk in stream:
                full_response += chunk["message"]["content"]
                response_placeholder.markdown(full_response)

        st.session_state.messages.append(
            {"role":"assistant","content":full_response}
        )

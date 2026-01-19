import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

st.set_page_config(page_title="Simple RAG Bot", page_icon="🤖")
st.title("🤖 Simple RAG Bot (Free / Local Embeddings)")
st.write("This version retrieves answers from your notes without OpenAI quota.")

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=emb
)

q = st.text_input("Ask something from your notes:")

if st.button("Ask") and q:
    docs = db.similarity_search(q, k=3)

    st.subheader("Most relevant notes (retrieved)")
    for i, d in enumerate(docs, 1):
        st.markdown(f"**Chunk {i}:**")
        st.write(d.page_content)

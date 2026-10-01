import streamlit as st
import main

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Document Q&A Assistant")
st.caption("Ask questions grounded strictly in your document embeddings.")

# Initialize chat history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar options
with st.sidebar:
    st.header("Configuration")
    st.write("**Model:** `openai/gpt-oss-120b`")
    st.write("**Embeddings:** `BAAI/bge-small-en-v1.5`")
    st.write("**Search:** MMR (`k=4`, `fetch_k=10`)")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Display prior conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("View Retrieved Context Chunks"):
                for idx, chunk in enumerate(message["sources"], 1):
                    st.markdown(f"**Chunk {idx}:**\n{chunk}")

# Capture user query
query = st.chat_input("Ask a question about your documents...")

if query:
    # 1. Show user message immediately
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 2. Query your pipeline from main.py
    with st.chat_message("assistant"):
        with st.spinner("Searching document & generating response..."):
            answer, sources = main.get_answer(query)
            st.markdown(answer)
            
            if sources:
                with st.expander("View Retrieved Context Chunks"):
                    for idx, chunk in enumerate(sources, 1):
                        st.markdown(f"**Chunk {idx}:**\n{chunk}")

    # 3. Save assistant message and sources
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
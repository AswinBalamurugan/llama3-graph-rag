import os
import streamlit as st
from scripts.query_graph import query_graph_rag, load_graph
from scripts.build_graph import load_docs, split_docs, create_graph, save_graph, GRAPH_PATH

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title('Graph-Based RAG Chatbot')

# Upload files
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    # Save uploaded files
    DATA_PATH = 'data'
    os.makedirs(DATA_PATH, exist_ok=True)
    for uploaded_file in uploaded_files:
        file_path = os.path.join(DATA_PATH, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Saved file {uploaded_file.name}")
    
    # Display placeholder message
    with st.spinner("Uploading file(s) to database..."):
        # Build and save graph
        docs = load_docs(DATA_PATH)
        chunks = split_docs(docs)
        graph = create_graph(chunks)
        save_graph(graph, GRAPH_PATH)
    
    st.success("Knowledge graph built and saved!", icon="✅")

if os.path.exists(GRAPH_PATH):
    G = load_graph(GRAPH_PATH)

    docs = set([G.nodes[node]['source'] for node in G.nodes])
    if docs:
        st.write("Documents available in the graph database:")
        for i,name in enumerate(docs):
            st.write(f'{i+1}. {name}')
    else:
        st.warning('The database is empty!', icon="⚠️")

# Display chat history
for i, (query, response, sources) in enumerate(st.session_state.chat_history):
    st.markdown(f'### Query {i+1}:')
    st.write(query)

    st.markdown(f"### Response {i+1}:")
    st.write(response)
    
    st.markdown(f"### Sources for Query {i+1}:")
    st.table(sources)
    
    st.markdown("---")

# Query input
query_text = st.text_input("Ask your question!")

if query_text:
    response, sources = query_graph_rag(query_text)

    # Add new interaction to chat history
    st.session_state.chat_history.append((query_text, response, sources))
    
    st.markdown('### Query:')
    st.write(query_text)

    st.markdown("### Response:")
    st.write(response)
    
    st.markdown("### Context:")
    st.table(sources)

    st.markdown("---")

# Add a button to clear chat history
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()

# Clear graph and data
if st.button("Clear Data and Graph"):
    if os.path.exists(GRAPH_PATH):
        os.remove(GRAPH_PATH)
    if os.path.exists('data'):
        os.system('rm -rf data')
    st.success("Graph and data cleared!", icon="✅")
    st.rerun()

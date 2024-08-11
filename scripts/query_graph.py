import json
import numpy as np
import pandas as pd
import networkx as nx
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate

GRAPH_PATH = 'graphs/graph_data.json'

PROMPT_TEMPLATE = [
    ("system", "You are a helpful assistant. Please answer the question using the context provided."),
    ("user", "Context: {context}\n---\nQuestion: {question}")
]

def load_graph(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return nx.node_link_graph(data)

def query_graph(graph, query_text):
    """
    Query the knowledge graph to find the most relevant context for the query using Ollama embeddings.
    """
    # Initialize the Ollama embedding function
    embedding_function = OllamaEmbeddings(model='mxbai-embed-large')

    # Generate the embedding for the query
    query_embedding = embedding_function.embed_query(query_text)

    # Compare the query embedding with each node's embedding in the graph
    relevant_nodes = []
    for node in graph.nodes:
        node_embedding = graph.nodes[node]['embedding']
        similarity = np.dot(query_embedding, node_embedding) \
                        / (np.linalg.norm(query_embedding) * np.linalg.norm(node_embedding))
        relevant_nodes.append((node, similarity))

    # Sort nodes by similarity
    relevant_nodes = sorted(relevant_nodes, key=lambda x: x[1], reverse=True)

    # Extract the most relevant context from the top nodes and their edges
    source_data = []
    context = []
    for node, similarity in relevant_nodes[:5]:
        # Get all edges (relationships) for this node
        edges = graph.edges(node, data=True)
        
        # Build the context from the edges
        for edge in edges:
            source, target, data = edge
            relation = data.get('relation', 'related to')
            context.append(f"{source} {relation} {target}")

            # Get the sources for the relationship in context
            node_source_1 = graph.nodes[source]['source']
            page_1 = graph.nodes[source]['page']+1
            node_source_2 = graph.nodes[target]['source']
            page_2 = graph.nodes[target]['page']+1

            if (node_source_1, page_1) not in source_data:
                source_data.append((node_source_1, page_1))
            
            if (node_source_2, page_2) not in source_data:
                source_data.append((node_source_2, page_2))

    context_text = "\n\n---\n\n".join(context) if context else "No relevant context found."
    
    return context_text, pd.DataFrame(source_data, columns=['Document', 'Page Number'])


def generate_response(context, query_text):
    prompt_template = ChatPromptTemplate.from_messages(PROMPT_TEMPLATE)
    model = Ollama(model="llama3")

    chain = prompt_template | model
    response = chain.invoke({"context": context, "question": query_text})
    
    return response

def query_graph_rag(query_text):
    graph = load_graph(GRAPH_PATH)
    context, source_data = query_graph(graph, query_text)
    response = generate_response(context, query_text)
    
    return response, source_data

if __name__ == "__main__":
    query_text = input("Enter your query: ")
    response, source_data = query_graph_rag(query_text)
    print(f"Response:\n{response}")
    print(f"Context:\n{source_data}")

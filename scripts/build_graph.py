import os, re
import json
import networkx as nx
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_PATH = 'data'
GRAPH_PATH = 'graphs/graph_data.json'

PROMPT_TEMPLATE = [
    ("system", "You are a helpful assistant specializing in extracting entities and relationships."),
    ("user", "Text: {text}\n---\nPlease extract as many entities and relationships as you can\
        in the following format:\nEntity1 -> relation -> Entity2\n\
        The response must only contain the realtionships and nothing else.\n\n")
]

def load_docs(path: str):
    loader = PyPDFDirectoryLoader(path)
    return loader.load()

def split_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    return text_splitter.split_documents(docs)

def extract_relationships(text):
    """
    Use the LLM to extract entities and relationships from the text.
    """
    prompt_template = ChatPromptTemplate.from_messages(PROMPT_TEMPLATE)
    model = Ollama(model="llama3")

    chain = prompt_template | model
    response = chain.invoke({"text": text})
    
    cleaned_response = re.sub(r'\*', '', response)

    entities, relationships = [], []
    lines = cleaned_response.strip().split('\n')
    
    for line in lines:
        if "->" in line:
            parts = line.split("->")
            if len(parts) == 3:
                source, relation, target = parts
                entities.extend([source.strip(), target.strip()])
                relationships.append((source.strip(), relation.strip(), target.strip()))
            elif len(parts) == 2:
                source, target = parts
                entities.extend([source.strip(), target.strip()])
                relationships.append((source.strip(), "", target.strip()))
    
    return entities, relationships

def create_graph(chunks):
    G = nx.Graph()
    # Initialize the Ollama embedding function
    embedding_function = OllamaEmbeddings(model='mxbai-embed-large')
    for chunk in chunks:
        entities, relationships = extract_relationships(chunk.page_content)
        
        # Add entities as nodes
        for entity in entities:
            # Generate the embedding for the enitity
            entity_embedding = embedding_function.embed_query(entity)
            if entity not in G.nodes:
                # get the file name
                source_file = chunk.metadata.get('source').split(os.sep)[-1]
                G.add_node(entity, source=source_file, page=chunk.metadata['page'], embedding=entity_embedding)
        
        # Add relationships as edges
        for relationship in relationships:
            source, relation, target = relationship
            G.add_edge(source, target, relation=relation)
    
    return G

def save_graph(graph, path):
    data = nx.node_link_data(graph)
    with open(path, 'w') as f:
        json.dump(data, f)

def main():
    docs = load_docs(DATA_PATH)
    chunks = split_docs(docs)
    graph = create_graph(chunks)
    save_graph(graph, GRAPH_PATH)

if __name__ == "__main__":
    main()

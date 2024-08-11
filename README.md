Please refer: [My Project Collection](https://github.com/AswinBalamurugan/Machine_Learning_Projects/blob/main/README.md)

# Graph-Based RAG Chatbot

## Aim
The aim of this project is to build a Graph-based Retrieval-Augmented Generation (RAG) system that extracts entities and relationships from PDF documents, creates a knowledge graph, and allows users to query the graph using natural language.

## Objective
* **Entity & Relationship Extraction**: Use a Language Model (LLM) to extract entities and relationships from documents.
* **Graph Construction**: Build a knowledge graph where nodes represent entities and edges represent relationships between them.
* **Query & Response**: Enable users to query the graph in natural language and get contextual responses based on the graph structure.

## Setup
To set up the project, follow these steps:

1. Clone the GitHub repository:
```bash
git clone <repository_url>
cd <repository_directory>
```
2. Create a Python environment using Conda:
```bash
conda create -n graph_rag_env python=3.9
conda activate graph_rag_env
```
3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Use the Streamlit UI
1. **Download the ollama package**: After downloading the ollama package, use the below commands to download and run the required LLM models.
```bash
ollama pull llama3 
ollama pull mxbai-embed-large 
```
The above commands have to be executed only once to download the models. The below command can be executed run the models.
```bash
ollama serve
```

2. **Run the chatbot**: Start the chatbot server and interact with it using the user interface by running the below command in the same directory where the `app.py` python file exists. 
```bash
streamlit run app.py
```

3. **Upload PDF documents**: Use the Streamlit UI to upload PDF files. The application will automatically build the knowledge graph from the uploaded documents.

4. **Ask Questions**: Once the graph is built, you can input your queries into the Streamlit UI. The system will retrieve relevant context from the knowledge graph and generate a response.  

Use the `graph-rag-example.pdf` to try out the sample question in the screenshots sections.

## Screenshots to compare graph-RAG and simple RAG
Comparing the results with the simple RAG implementation from my [A Chatbot with Chroma Database and Llama3 Model](https://github.com/AswinBalamurugan/llama3-local-rag.git) project.

|Simple RAG| Graph RAG|
|-----|-----|
|![rag](https://github.com/AswinBalamurugan/llama3-graph-rag.git/blob/main/images/rag.png)|![graph](https://github.com/AswinBalamurugan/llama3-graph-rag.git/blob/main/images/graph-rag.png)|

The graph-RAG response is superior due to its comprehensive analysis, which integrates contextual depth and strategic considerations, offering a nuanced comparison beyond mere physical attributes. This approach provides a more holistic understanding of the characters' abilities and potential interactions.

## Conclusion
This project leverages graph-based structures and LLMs to create a dynamic knowledge retrieval system. By following the setup and usage instructions, you can explore the potential of RAG systems in document analysis and query-based information retrieval.

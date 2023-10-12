import os
os.environ['OPENAI_API_KEY']="<API_KEY_HERE>"

from flask import Flask, request, jsonify
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from googleDriveLoader import GoogleDriveLoader
import json

app = Flask(__name__)


def load_and_create_vector_store(embedding, persist_directory, loaders=None, documents=None):
    docs = []

    if loaders is not None:
        for loader in loaders:
            docs.extend(loader.load())

    if documents is not None:
        docs.extend(documents)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)

    vectordb = Chroma.from_documents(documents=splits, embedding=embedding, persist_directory=persist_directory)
    return vectordb


@app.route('/load_pdfs', methods=['POST'])
def load_pdfs():
    pdf_paths = request.json['pdf_paths']
    loaders = [PyPDFLoader(pdf_path) for pdf_path in pdf_paths]
    embedding = OpenAIEmbeddings()
    persist_directory = 'docs/chroma/'
    vectordb = load_and_create_vector_store(embedding, persist_directory, loaders=loaders)
    return jsonify({"message": "PDFs loaded successfully"})


@app.route('/similarity_search', methods=['POST'])
def similarity_search():
    global vectordb
    question = request.json['question']
    k = request.json.get('k', 5)

    if vectordb is not None:
        docs = vectordb.similarity_search(question, k=k)
        results = [{"document": doc.page_content} for doc in docs]
        return jsonify({"results": results})
    else:
        return jsonify({"error": "Vector database not initialized."})


@app.route('/similarity_search_with_score', methods=['POST'])
def similarity_search_with_score():
    global vectordb
    question = request.json['question']
    k = request.json.get('k', 5)

    if vectordb is not None:
        docs = vectordb.similarity_search_with_score(question, k=k)
        results = [{"document": doc[0].page_content, "score": doc[1]} for doc in docs]
        return jsonify({"results": results})
    else:
        return jsonify({"error": "Vector database not initialized."})


@app.route('/similarity_search_best_score', methods=['POST'])
def similarity_search_best_score():
    global vectordb
    question = request.json['question']
    k = request.json.get('k', 5)

    if vectordb is not None:
        docs = vectordb.similarity_search_with_score(question, k=k)
        min_score_result = min(docs, key=lambda x: x[1])
        result_item = {
            "document": min_score_result[0].page_content,
            "score": min_score_result[1]
        }
        return jsonify({"result": result_item})
    else:
        return jsonify({"error": "Vector database not initialized."})


@app.route('/load_gdrive', methods=['POST'])
def load_gdrive():
    global vectordb
    gdl_instance = GoogleDriveLoader()
    username = request.json['username']
    docs = gdl_instance.load(username)
    embedding = OpenAIEmbeddings()
    persist_directory = 'docs/chroma/'
    vectordb = load_and_create_vector_store(embedding, persist_directory, documents=docs)
    return jsonify({"results": "SUCCESS"})

@app.route('/get_shortlisted_doc', methods=['POST'])
def get_shortlisted_doc():
    try:
        # global vectordb

        # Get the JSON data from the request
        data = request.get_json()
        # Ensure that the data contains the expected keys
        if 'question' not in data or 'username' not in data or 'files' not in data:
            return jsonify(
                {'error': 'Invalid JSON data format. Expecting "question," "username," and "files" keys'}), 400

        # Extract the values from the JSON data
        question = data['question']
        username = data['username']
        files = data['files']

        gdl_instance = GoogleDriveLoader()
        docs = gdl_instance.load_documents_from_list(json.loads(files), username)
        embedding = OpenAIEmbeddings()
        persist_directory = 'docs/chroma/'+username+"/"
        vectordb_internal = load_and_create_vector_store(embedding, persist_directory, documents=docs)

        docs = vectordb_internal.similarity_search_with_score(question, k=5)
        min_score_result = min(docs, key=lambda x: x[1])
        result_item = {
            "document": min_score_result[0].page_content,
            "score": min_score_result[1]
        }
        return jsonify({"result": result_item})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_documents', methods=['GET'])
def get_documents():
    global vectordb

    if vectordb is not None:
        documents = vectordb.get()
        return jsonify({"documents": documents})
    else:
        return jsonify({"error": "Vector database not initialized."})


if __name__ == '__main__':
    vectordb = None
    app.run(host='0.0.0.0', port=5000)
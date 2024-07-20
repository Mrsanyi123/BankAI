from flask import Flask, request, jsonify
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from embedding import get_embedding_function
import logging

app = Flask(__name__)

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

logging.basicConfig(level=logging.INFO)

def query_rag(query_text: str):
    # Prepare the DB
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB
    results = db.similarity_search_with_score(query_text, k=1)
    logging.info(f"Results: {results}")

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)
    logging.info(f"Response: {response_text}")
    return response_text

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    query_text = data['query_text']
    response_text = query_rag(query_text)
    return jsonify({'response_text': response_text})

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
import openai
from embedding import get_embedding_function
import logging
from dotenv import load_dotenv
import os

app = Flask(__name__)

CHROMA_PATH = "chroma"

# Modify the prompt template to remove "based on the provided context"
PROMPT_TEMPLATE = """
Context:

{context}

---

Question: {question}

Answer:
"""

logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from the environment
openai.api_key = os.getenv('OPENAI_API_KEY')

def query_rag(query_text: str):
    # Prepare the DB
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB
    results = db.similarity_search_with_score(query_text, k=1)
    logging.info(f"Results: {results}")

    if not results:
        return "No relevant information found in the documents."

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Use OpenAI GPT for response generation
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    response_text = response['choices'][0]['message']['content']
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

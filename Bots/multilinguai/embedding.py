import openai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from the environment
openai.api_key = os.getenv('OPENAI_API_KEY')


class OpenAIEmbedding:
  def __init__(self, model="text-embedding-ada-002"):
    self.model = model

  def embed_query(self, text):
    response = openai.Embedding.create(input=[text], model=self.model)
    return response['data'][0]['embedding']

def get_embedding_function():
  return OpenAIEmbedding()

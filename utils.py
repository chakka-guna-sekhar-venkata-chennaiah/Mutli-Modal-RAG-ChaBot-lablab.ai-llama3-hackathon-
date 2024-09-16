from openai import OpenAI
from langchain_community.vectorstores import FAISS  # Updated import
from langchain_core.llms.base import LLM  # Import the base LLM class
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from pydantic import BaseModel, Field
import streamlit as st

client = OpenAI(
    api_key=st.secrets['api_key'],
    base_url="https://llm.mdb.ai/"
)

class MDBEmbeddings(Embeddings):
    def __init__(self, client):
        super().__init__()
        self.client = client

    def embed_query(self, text):
        response = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def __call__(self, text):
        return self.embed_query(text)

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

# Define the LLM response function using Groq
def get_llm_response(prompt):
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


# Instantiate the embeddings and LLM classes
embeddings = MDBEmbeddings(client=client)


# Load the FAISS index with custom embeddings
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
db1 = FAISS.load_local("faiss_index_audio", embeddings, allow_dangerous_deserialization=True)

# Create a wrapper class for the Groq LLM
class GroqLLM(LLM):
    def __init__(self, client):
        self.client = client

    def _call(self, prompt, **kwargs):
        return self.client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}]
        ).choices[0].message.content

# Instantiate the Groq LLM
groq_llm = GroqLLM(client)

# Define the prompt template for the LLMChain
prompt_template = """
You are an assistant tasked with summarizing tables and text.
Give a concise summary of the table or text.
Answer the question based only on the following context, which can include text, images, and tables:
{context}
Question: {question}
Don't answer if you are not sure and decline to answer and say "Sorry, I don't have much information about it."
Just return the helpful answer in as much detail as possible.
Answer:
"""

# Setup the LLMChain with the custom chat model
qa_chain = LLMChain(llm=groq_llm, prompt=PromptTemplate.from_template(prompt_template))

# Define the answer function to handle queries
def answer(question):
    relevant_docs = db.similarity_search(question)
    context = ""
    relevant_images = []
    for d in relevant_docs:
        if d.metadata['type'] == 'text':
            context += '[text]' + d.metadata['original_content']
        elif d.metadata['type'] == 'table':
            context += '[table]' + d.metadata['original_content']
        elif d.metadata['type'] == 'image':
            context += '[image]' + d.page_content
            relevant_images.append(d.metadata['original_content'])
    result = qa_chain.run({'context': context, 'question': question})
    return result, relevant_images


# Query the vectorstore
def answer1(question):
    relevant_docs = db1.similarity_search(question)
    context = ""
    relevant_images = []
    for d in relevant_docs:
        if d.metadata['type'] == 'text':
            context += '[text]' + d.metadata['original_content']
        elif d.metadata['type'] == 'image':
            context += '[image]' + d.page_content
            relevant_images.append(d.metadata['original_content'])
    result = qa_chain.run({'context': context, 'question': question})
    return result, relevant_images



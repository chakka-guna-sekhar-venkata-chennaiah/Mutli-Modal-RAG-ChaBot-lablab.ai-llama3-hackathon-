from openai import OpenAI
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from pydantic import BaseModel, Field
import streamlit as st

# Update the client for Groq
client = OpenAI(
    base_url='https://api.groq.com/openai/v1',
    api_key=st.secrets['api_key']
)

# Define the embedding function
def embed_query(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Update the LLM call in the answer function
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
    
    # Call the LLM using Groq
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
        ]
    )
    result = response.choices[0].message.content
    return result, relevant_images

# Load the FAISS index with embeddings using the embed_query function
# Note: You may need to create embeddings for your documents before loading the FAISS index
# Example: embeddings = [embed_query(doc) for doc in documents]  # Create embeddings for your documents

db = FAISS.load_local("faiss_index", embed_query, allow_dangerous_deserialization=True)
db1 = FAISS.load_local("faiss_index_audio", embed_query, allow_dangerous_deserialization=True)

# Setup the LLMChain with the Groq LLM function
def get_llm_response(prompt):
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

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

# Update the qa_chain to use the get_llm_response function
qa_chain = LLMChain(llm=get_llm_response, prompt=PromptTemplate.from_template(prompt_template))

# Define the answer function to handle queries
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



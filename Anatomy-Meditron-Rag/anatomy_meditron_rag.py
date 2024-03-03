# -*- coding: utf-8 -*-
"""Anatomy-Meditron-RAG.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1R1q8f3u9M-k0W25fueWtYsg6Qwl4eAXS
"""

# !pip install -q -r requirements.txt

# torch
# sentence-transformers==2.2.2
# transformers
# langchain
# fastapi
# uvicorn
# pypdf
# PyPDF2
# jinja2
# qdrant-client
# ctransformers
# python-multipart
# aiofiles
# pdfquery
# chromadb
# python-dotenv

# Imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, PyPDFLoader , TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pdfquery import PDFQuery
import warnings
import os
from dotenv import load_dotenv, dotenv_values

# !mkdir data
# !cd data && mkdir ChromaDB
# !touch requirements.txt
# !touch .env

#Create the embedding model
embeddings = SentenceTransformerEmbeddings(model_name='NeuML/pubmedbert-base-embeddings')

# document loader
loader = DirectoryLoader('./data/', glob='**/*.pdf', show_progress=True, loader_cls=PyPDFLoader)

documents = loader.load();

#Split the loaded docs into text

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1024,
    chunk_overlap = 100,
)

texts = text_splitter.split_documents(documents=documents)

"""Create the Vector DB\
Supplying a persist_directory stores the embedding on disk
"""

persist_directory = './data/ChromaDB'

chromadb = Chroma.from_documents(
    texts,
    embeddings,
    persist_directory=persist_directory
)

"""Make a Retriever"""

retriever = chromadb.as_retriever(search_kwargs={'k':5})

"""Make a chain"""

local_llm = 'TheBloke/meditron-7B-chat-GGUF' ## LLM to use.

config = {
    'max_new tokens':2000,
    'context_length':2048,
    'repetition_penalty':1.1,
    'temperature': 0.1,
    'top_k':50,
    'top_p':0.9,
    'stream':True,
    # 'threads':int(os.cpu_count()/2)
}

llm = CTransformers(
    model=local_llm,
    model_type='llama',
    **config,

)

print(llm('AI is going to'))

prompt_template = """
  Use the following pieces of information to answer the user's questions.
  If the question is related to anatomy and physiology, answer the question appropriately using the knowledge you have.
  If the question is is not anatomy or physiology related, just say you don't know the answer.
  Do not try to make up answers in case you do not know the answer.

  Context: {context}
  Question: {question}

  Return the helpful answer below, nothing else.
  Helpful answer:
"""


QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)# Run chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

"""**Test the qa_chain**"""

warnings.filterwarnings("ignore", message="Number of tokens \(\d+\) exceeded maximum context length \(512\)", category=UserWarning)
question = "Central nervous system?"
result = qa_chain({"query": question});
# Check the result of the query
result["result"];
# Check the source document from where we
result["source_documents"][0]


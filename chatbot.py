from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.chains import RetrievalQA
from langchain_community.llms import CTransformers
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

app = Flask(__name__)
CORS(app)

# Initialize Pinecone
pc = Pinecone(api_key="38693106-7022-47cd-be26-a64298125bcc")
index_name = 'medical-chatbot'
index = pc.Index(index_name)

def load_pdf(data):
    loader = DirectoryLoader(data, glob="*.pdf")
    return loader.load()

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return text_splitter.split_documents(extracted_data)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = PineconeVectorStore(index, embedding)

prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
helpful answer:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

llm = CTransformers(model="C:\\Users\\USER\\OneDrive\\Desktop\\database chatbot\\llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens': 512, 'temperature': 0.8})

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query', '')

    result = qa.invoke({"query": query})

    return jsonify({"result": result["result"]})

if __name__ == '__main__':
    app.run(port=5000)

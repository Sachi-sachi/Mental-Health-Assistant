from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings
import logging
import cassio
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Cassandra
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os

# Flask setup
app = Flask(__name__)
CORS(app)

# Suppress warnings & logging
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger('cassandra.protocol').setLevel(logging.ERROR)


# Connect to AstraDB
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# Load Hugging Face Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize ChatGroq LLM
llm = ChatGroq(model="llama3-8b-8192", temperature=0, max_tokens=4000)

# Create vector store in AstraDB
astra_vector_store = Cassandra(embedding=embeddings, session=None, table_name="Projekt", keyspace=None)

# Set up memory for conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define RAG-based chatbot chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=astra_vector_store.as_retriever(search_kwargs={"k": 3}),
    memory=memory
)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("question", "").strip()

    if not user_input:
        return jsonify({"error": "Invalid input"}), 400

    try:
        response = qa_chain.invoke({"question": user_input})
        return jsonify({"response": response["answer"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

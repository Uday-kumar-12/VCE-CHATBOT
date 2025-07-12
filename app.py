from flask import Flask, render_template, request, jsonify
import os
import warnings
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module='langchain')

app = Flask(__name__)
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as short as possible from the provided context, make sure to provide all the details. 
    If the answer is not in the provided context, and if the context refers to daily routine or wishes, give the output; 
    otherwise, say "answer is not available in the context". Do not provide a wrong answer.

    Context:\n {context}\n
    Question: {question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

@app.route("/")
def index():
    return render_template("index1.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_question = request.form["question"]
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    index_path = "faiss_index"
    try:
        # Load FAISS index and search for relevant docs
        new_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        if docs:
            chain = get_conversational_chain()
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            output = response["output_text"]
        else:
            # Fallback model if no relevant docs are found
            general_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
            output = general_model(user_question=user_question).text

        return jsonify({"response": output})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
app = Flask(__name__)

load_dotenv()

CORS(app)


app.debug = True

from langchain.retrievers import WikipediaRetriever
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

openai_api_key = os.getenv('OPENAI_API_KEY')

os.environ["OPENAI_API_KEY"] = openai_api_key


retriever = WikipediaRetriever()

model = ChatOpenAI(model_name="gpt-4") 
qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get("question")

    if question:
        result = qa({"question": question, "chat_history": []})
        answer = result['answer']
        return jsonify({"answer": answer})
    else:
        return jsonify({"error": "Missing 'question' parameter"}), 400

if __name__ == '__main__':
    app.run()
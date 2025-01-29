import logging
import os
import getpass
from langchain_openai import ChatOpenAI
from flask import Flask, request, jsonify
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPEN_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

app = Flask(__name__)

chat_model = ChatOpenAI(model="gpt-4", api_key=os.environ["OPENAI_API_KEY"])


@app.route("/talking_ia", methods=["POST"])
def text_ai():
    user_text = request.json.get("text")

    if not user_text:
        return jsonify({"error": "Text not provided"}), 400

    result = chat_model.invoke(input=user_text)

    return jsonify({"response": str(result.content)})


if __name__ == "__main__":
    app.run(debug=True)

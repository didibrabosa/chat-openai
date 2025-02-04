import logging
import os
import getpass
from langchain_openai import ChatOpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPEN_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

app = FastAPI()

chat_model = ChatOpenAI(model="gpt-4", api_key=os.environ["OPENAI_API_KEY"])


class QuestionRequest(BaseModel):
    text: str


@app.post("/talking_ai")
def text_ai(request: QuestionRequest):
    try:
        user_text = request.text

        if not user_text:
            return HTTPException(status_code=400, detail="Text not provided")

        result = chat_model.invoke(input=user_text)

        return ({"response": str(result.content)})
    except Exception as ex:
        logger.error(f"Error whike asking question: {ex}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

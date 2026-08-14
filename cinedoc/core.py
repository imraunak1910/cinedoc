from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import ChatMistralAI

model=ChatMistralAI(model="mistral-small-2603")

response=model.invoke("what is an LLM?")

print(response.content)


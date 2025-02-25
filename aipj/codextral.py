import os
from mistralai import Mistral
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

client = Mistral(api_key=api_key)

model = "codestral-mamba-latest"

message = [
    {
        "role": "user", 
        "content": "Write a function for fibonacci"
    }
]

chat_response = client.chat.complete(
    model=model,
    messages=message
)
print(chat_response.choices[0].message.content)
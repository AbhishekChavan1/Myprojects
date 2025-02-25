import os
from mistralai import Mistral

api_key = os.getenv("MISTRAL_API_KEY")
model = "mistral-embed"

client = Mistral(api_key=api_key)

embeddings_response = client.embeddings.create(
    model=model,
    inputs=["Embed this sentence.", "As well as this one."]
)

print(embeddings_response)
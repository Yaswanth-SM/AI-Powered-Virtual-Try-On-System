import os
from dotenv import load_dotenv
from huggingface_hub import whoami

load_dotenv()

token = os.getenv("HF_TOKEN")

if not token:
    print("Token not found. Check your .env file.")
else:
    try:
        user = whoami(token=token)
        print("Hugging Face login successful!")
        print("Username:", user["name"])
    except Exception as error:
        print("Token check failed:", error)
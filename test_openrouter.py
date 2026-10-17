
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    print("OpenRouter API key not found in .env")
    raise SystemExit(1)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

try:
    
    response = client.chat.completions.create(
        model="google/gemini-3.1-flash-lite",
        messages=[
        {
            "role": "user",
            "content": "Give one short suggestion to improve phone battery life."
        }
    ],
    max_tokens=100,
    temperature=0.3
)


    print("OpenRouter is working!")
    print(response.choices[0].message.content)

except Exception as error:
    print("Request failed:", error)

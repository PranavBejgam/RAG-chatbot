import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.responses.create(
    model="gpt-4o-mini",
    input="Write a creative short story of boy going to office",
    temperature=1.5,
    max_output_tokens=50
)

print(response.output_text)

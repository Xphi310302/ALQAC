from openai import OpenAI
from dotenv import load_dotenv
import json
import os

# Construct the path to the .env file relative to this script's location
# to ensure it's found regardless of the current working directory.
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)


class FPTApi:
    def __init__(self, model_name: str = "Qwen2.5-7B-Instruct"):
        api_key = os.getenv("FPT_API_KEY")
        base_url = os.getenv("FPT_API_URL")

        if not api_key:
            raise ValueError(
                "FPT_API_KEY not found. "
                "Please ensure it is set in the llm_fpt/.env file."
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model_name = model_name

    def complete(self, system_prompt: str = "", text: str = "") -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": text},
            ],
            max_tokens=1000,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

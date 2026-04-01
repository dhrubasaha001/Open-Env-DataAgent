import os
import re
from openai import OpenAI
from env import DataAnalysisEnv
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)


def extract_number(text):
    if not text:
        return None
    match = re.search(r"\d+.?\d*", text)
    return float(match.group()) if match else None


def run_agent():
    env = DataAnalysisEnv("data/sample.csv")
    state = env.reset()

    print("Initial State:", state)

    prompt = f"""
You are a strict data analyst.

Dataset:
Columns: {state['columns']}
Rows: {state['rows']}

Task: Calculate the mean of 'price'.

RULES:
* Return ONLY a number
* No explanation
* No text
* Example: 5500.0

Answer:
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Return only numeric answers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=30
        )

        output = response.choices[0].message.content
        print("LLM Output:", output)

    except Exception as e:
        print("⚠️ LLM Error:", e)
        output = None

    # 🔥 Robust parsing
    predicted_value = extract_number(output)

    # 🔥 Fallback if LLM fails
    if predicted_value is None:
        print("Using fallback value")
        predicted_value = float(env.data["price"].mean())

    action = {
        "task": "mean",
        "column": "price",
        "value": predicted_value
    }

    obs, reward, done, _ = env.step(action)

    print("Observation:", obs)
    print("Final Score:", reward)


if __name__ == "__main__":
    run_agent()
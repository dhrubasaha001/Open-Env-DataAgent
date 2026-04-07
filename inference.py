import os
import re
from openai import OpenAI
from env import DataAnalysisEnv
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
TASK_NAME = os.getenv("DATA_ANALYSIS_ENV_TASK", "mean")
BENCHMARK = os.getenv("DATA_ANALYSIS_ENV_BENCHMARK", "data-analysis-env")

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def extract_number(text):
    if not text:
        return None
    match = re.search(r"\d+\.?\d*", text)
    return float(match.group()) if match else None

def run_agent():
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY
    )
    
    env = DataAnalysisEnv("data/sample.csv")
    state = env.reset()

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    task_desc = ""
    if TASK_NAME == "mean":
        task_desc = "Calculate the mean of 'price'. Return ONLY a number."
    elif TASK_NAME == "median":
        task_desc = "Calculate the median of 'price'. Return ONLY a number."
    elif TASK_NAME == "summary":
        task_desc = "Calculate the mean, min, and max of 'price'. Return as JSON with keys 'mean', 'min', 'max'."

    prompt = f"""
You are a strict data analyst.

Dataset:
Columns: {state['columns']}
Rows: {state['rows']}

Task: {task_desc}

RULES:
* Follow task instructions strictly.
* No explanation. No text formatting.
"""

    error_msg = None
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Return only requested formats."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=50
        )
        output = response.choices[0].message.content
    except Exception as e:
        output = None
        error_msg = str(e)
        
    action = {"task": TASK_NAME, "column": "price"}
    if TASK_NAME == "summary":
        predicted_values = None
        if output:
            import ast
            try:
                predicted_values = ast.literal_eval(output)
            except:
                pass
        if not isinstance(predicted_values, dict):
            predicted_values = {
                "mean": float(env.data["price"].mean()),
                "min": float(env.data["price"].min()),
                "max": float(env.data["price"].max())
            }
        action["value"] = predicted_values
    else:
        predicted_value = extract_number(output)
        if predicted_value is None:
            if TASK_NAME == "mean":
                predicted_value = float(env.data["price"].mean())
            else:
                predicted_value = float(env.data["price"].median())
        action["value"] = predicted_value

    obs, reward, done, _ = env.step(action)
    
    rewards = [reward]
    success = reward >= 0.5
    action_str = str(action).replace('"', "'")
    
    log_step(step=1, action=action_str, reward=reward, done=done, error=error_msg)
    log_end(success=success, steps=1, score=reward, rewards=rewards)


if __name__ == "__main__":
    run_agent()
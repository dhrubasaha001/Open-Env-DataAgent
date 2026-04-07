from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from env import DataAnalysisEnv
import uvicorn

app = FastAPI()

env = DataAnalysisEnv("data/sample.csv")

class DataAction(BaseModel):
    task: str
    column: str
    value: Optional[Any] = None

class DataObservation(BaseModel):
    task: Optional[str] = None
    column: Optional[str] = None
    true_value: Optional[float] = None
    predicted_value: Optional[float] = None
    true_values: Optional[Dict[str, float]] = None
    predicted_values: Optional[Dict[str, Any]] = None
    score: Optional[float] = None
    result: Optional[float] = None
    mean: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    error: Optional[str] = None

class StepResponse(BaseModel):
    observation: DataObservation
    reward: float
    done: bool
    info: Dict[str, Any]

class StateResponse(BaseModel):
    step: int
    columns: List[str]
    rows: int

@app.post("/reset", response_model=StateResponse)
def reset():
    return env.reset()

@app.post("/step", response_model=StepResponse)
def step(action: DataAction):
    obs, reward, done, info = env.step(action.dict())
    return {
        "observation": obs,
        "reward": float(reward),
        "done": done,
        "info": info
    }

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()

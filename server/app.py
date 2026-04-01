from fastapi import FastAPI
from env import DataAnalysisEnv
import uvicorn

app = FastAPI()

env = DataAnalysisEnv("data/sample.csv")

@app.post("/reset")
def reset():
    return env.reset()

@app.post("/step")
def step(action: dict):
    obs, reward, done, info = env.step(action)
    return {
    "observation": obs,
    "reward": reward,
    "done": done,
    "info": info
    }



def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if "__name___" == "__main__":
    main()

from env import DataAnalysisEnv

env = DataAnalysisEnv("data/sample.csv")

state = env.reset()
print("Initial State:", state)

action = {
    "task": "mean",
    "column": "price",
    "value": 5400  
}

obs, reward, done, _ = env.step(action)

print("Observation:", obs)
print("Final Score:", reward)
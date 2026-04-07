# 📊 Data Analysis OpenEnv Environment

## 💥 Overview

This project implements a **real-world OpenEnv environment** where an AI agent performs data analysis tasks on a dataset.

The environment follows the standard:

* `reset()`
* `step()`
* `state()`

API and provides **graded rewards (0.0 → 1.0)** based on prediction accuracy.

---

## ⚙️ Environment Description

The environment simulates a **data analysis workflow**:

* Agent receives dataset metadata (columns, rows)
* Agent performs analytical tasks
* Environment evaluates predictions using a scoring system

---

## 🧩 Tasks

### 🟢 Easy

* Mean calculation

### 🟡 Medium

* Median calculation

### 🔴 Hard

* Summary statistics (mean, min, max)

---

## 🧠 Reward System

Uses **relative error-based scoring**:

* Near correct → **1.0**
* Slight error → **0.8 / 0.5**
* Incorrect → **0.0**

Provides **partial reward signals**, not binary scoring.

---

## 🤖 Inference Script

Run agent:

```bash
python inference.py
```

* Uses OpenAI-compatible API
* Handles imperfect LLM output
* Includes fallback mechanism for robustness

---

## 📦 Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🐳 Docker

Build and run:

```bash
docker build -t openenv-test .
docker run -p 7860:7860 openenv-test
```

Test API:

```bash
curl.exe -X POST http://localhost:7860/reset
```

---

## 🔁 API Endpoints

### POST `/reset`

Returns initial state:

```json
{
  "step": 0,
  "columns": ["price"],
  "rows": 10
}
```

---

### POST `/step`

Input:

```json
{
  "task": "mean",
  "column": "price",
  "value": 5400
}
```

Output:

```json
{
  "observation": {...},
  "reward": 0.8,
  "done": true
}
```

---

## 📁 Project Structure

```
env.py
tasks.py
grader.py
baseline.py
inference.py
server/app.py
openenv.yaml
pyproject.toml
uv.lock
requirements.txt
Dockerfile
README.md
data/sample.csv
```

---

## 💡 Key Features

* Real-world data analysis simulation
* Modular environment design
* Multi-task difficulty levels
* Robust grading system
* Fault-tolerant inference (handles LLM failures)

---

## 🚀 Status

✅ OpenEnv compliant
✅ Multi-mode deployment ready
✅ Docker verified
✅ API tested

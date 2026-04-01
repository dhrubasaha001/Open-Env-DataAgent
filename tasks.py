from grader import grade_mean

def mean_task(data, column, predicted_value):
    if column not in data.columns:
        return {"error": f"Column '{column}' not found"}, -0.5, False

    true_value = float(data[column].mean())

    score = grade_mean(true_value, predicted_value)

    observation = {
        "task": "mean",
        "column": column,
        "true_value": true_value,
        "predicted_value": predicted_value,
        "score": score
    }

    return observation, score, True

def median_task(data, column):
    if column not in data.columns:
        return {"error": f"Column '{column}' not found"}, -0.5, False

    result = float(data[column].median())

    observation = {
        "task": "median",
        "column": column,
        "result": result
    }

    return observation, 1.0, True

def summary_task(data, column):
    if column not in data.columns:
        return {"error": f"Column '{column}' not found"}, -0.5, False

    mean_val = float(data[column].mean())
    min_val = float(data[column].min())
    max_val = float(data[column].max())

    observation = {
        "task": "summary",
        "column": column,
        "mean": mean_val,
        "min": min_val,
        "max": max_val
    }

    return observation, 1.0, True
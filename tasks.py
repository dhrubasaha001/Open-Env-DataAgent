from grader import grade_mean, grade_median, grade_summary

def mean_task(data, column, predicted_value):
    if column not in data.columns:
        return {"error": f"Column '{column}' not found"}, 0.01, False

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

def median_task(data, column, predicted_value):
    if column not in data.columns:
        return {"error": f"Column '{column}' not found"}, 0.01, False

    true_value = float(data[column].median())
    score = grade_median(true_value, predicted_value)

    observation = {
        "task": "median",
        "column": column,
        "true_value": true_value,
        "predicted_value": predicted_value,
        "score": score
    }

    return observation, score, True

def summary_task(data, column, predicted_values):
    if column not in data.columns:
        return {"error": f"Column '{column}' not found"}, 0.01, False

    true_values = {
        "mean": float(data[column].mean()),
        "min": float(data[column].min()),
        "max": float(data[column].max())
    }
    
    score = grade_summary(true_values, predicted_values)

    observation = {
        "task": "summary",
        "column": column,
        "true_values": true_values,
        "predicted_values": predicted_values,
        "score": score
    }

    return observation, score, True
def grade_mean(true_value, predicted_value):
    return grade_numeric(true_value, predicted_value)

def grade_median(true_value, predicted_value):
    return grade_numeric(true_value, predicted_value)

def grade_summary(true_values, predicted_values):
    if not isinstance(predicted_values, dict):
        return 0.0
    
    scores = []
    for k in ['mean', 'min', 'max']:
        if k in predicted_values:
            scores.append(grade_numeric(true_values[k], predicted_values[k]))
        else:
            scores.append(0.0)
            
    return sum(scores) / 3.0 if scores else 0.0

def grade_numeric(true_value, predicted_value):
    if true_value == 0:
        return 1.0 if predicted_value == 0 else 0.0

    try:
        error = abs(true_value - predicted_value)
        relative_error = error / abs(true_value)

        if relative_error < 0.01:      
            return 1.0
        elif relative_error < 0.05:
            return 0.8
        elif relative_error < 0.2:
            return 0.5
        else:
            return 0.0
    except (TypeError, ValueError):
        return 0.0
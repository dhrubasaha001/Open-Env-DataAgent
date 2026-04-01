def grade_mean(true_value, predicted_value):
    error = abs(true_value - predicted_value)

    if true_value == 0:
        return 0.0

    relative_error = error / abs(true_value)

    if relative_error < 0.01:      
        return 1.0
    elif relative_error < 0.05:
        return 0.8
    elif relative_error < 0.2:
        return 0.5
    else:
        return 0.0
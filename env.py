from tasks import mean_task, median_task, summary_task


class DataAnalysisEnv:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.current_step = 0
        self.done = False

    def reset(self):
        import pandas as pd

        self.data = pd.read_csv(self.data_path)
        self.data.columns = self.data.columns.str.strip()
        self.current_step = 0
        self.done = False

        return self.state()

    def state(self):
        return {
            "step": self.current_step,
            "columns": list(self.data.columns),
            "rows": len(self.data)
        }

    def step(self, action):
        self.current_step += 1

        task = action.get("task")
        column = action.get("column")
        value = action.get("value")

        if task == "mean":
            return (*mean_task(self.data, column, value), {})

        elif task == "median":
            return (*median_task(self.data, column), {})

        elif task == "summary":
            return (*summary_task(self.data, column), {})

        return {"error": "Invalid task"}, -0.2, False, {}

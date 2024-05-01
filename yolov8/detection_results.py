import ultralytics


class DetectionResults:
    """
    Class that converts Ultralytics results into a more usable format for the pool table detection system
    """

    def __init__(self, results: ultralytics.engine.results.Results):
        results = results.boxes
        self.results = {}
        if not hasattr(results, "id"):
            self.results = {}
        try:
            for i in range(len(results.id)):
                # Skip if it's not a ball
                if results.cls[i] != 0:
                    continue
                self.results[int(results.id[i])] = {
                    "x": float(results.xywh[i][0]),
                    "y": float(results.xywh[i][1]),
                    "w": float(results.xywh[i][2]),
                    "h": float(results.xywh[i][3]),
                }
        except Exception as e:
            print(e)
            self.results = {}

    def __getitem__(self, ball_id: int):
        return self.results[ball_id]

    def __iter__(self):
        return iter(self.results.values())

    def __len__(self):
        return len(self.results)

    def keys(self):
        return self.results.keys()

    def items(self):
        return self.results.items()

    def is_empty(self):
        return len(self.results) == 0

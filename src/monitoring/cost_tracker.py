"""Simple cost tracking hooks for model usage."""


class CostTracker:
    def __init__(self):
        self.total_cost = 0.0

    def add_cost(self, amount: float) -> float:
        self.total_cost += amount
        return self.total_cost

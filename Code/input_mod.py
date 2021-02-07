class InputNW:
    def __init__(self, value, weight):
        self.value = value
        self.weight = weight
        self.calculated_value = value * weight

    def getValue(self):
        return self.calculated_value

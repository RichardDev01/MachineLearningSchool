class InputNW:
    def __init__(self, value, weight):
        self.value = value
        self.weight = weight

    def getValue(self):
        calculated_value = self.value * self.weight
        return calculated_value

    def setWeight(self, weight):
        self.weight = weight

    def setValue(self, value):
        self.value = value

    def __str__(self):
        return f'This input had a value of {self.value} and a weight of {self.weight}'

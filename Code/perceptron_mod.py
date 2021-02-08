class InputNW:
    def __init__(self, inputlist: [[float]], bias: float = 0):
        # Input list contain as first value the raw input and as 2nd value the weight of that value
        self.inputlist = inputlist
        self.bias = bias

    def getValue(self):
        input_sum = 0
        for inp in self.inputlist:
            input_sum += inp[0] * inp[1]
        input_sum += self.bias
        return input_sum


    

    def __str__(self):
        return f'This input has {self.inputlist} as input' \
               f' and input has {self.bias} as bias\n '

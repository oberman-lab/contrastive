import numpy as np


def set_requires_grad(m, requires_grad):
    if hasattr(m, 'weight') and m.weight is not None:
        m.weight.requires_grad_(requires_grad)
    if hasattr(m, 'bias') and m.bias is not None:
        m.bias.requires_grad_(requires_grad)


class RunningAvg:
    def __init__(self):
        self.mean = 0
        self.N = 0

    def add(self, x):
        N = self.N
        self.mean = (self.mean * N + x) / (N + 1)
        self.N = N + 1

    def get(self):
        return self.mean

    def wipe(self):
        self.mean = 0
        self.N = 0

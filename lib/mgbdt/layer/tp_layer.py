import numpy as np


class TPLayer:
    """Target Porpopgation Layer
    """
    def __init__(self, F, G):
        self.F = F
        self.G = G
        self.input_size = F.input_size
        self.output_size = F.output_size

    def forward(self, input):
        return self.F(input)

    def backward(self, input, target):
        out = self.G(target)
        return out

    def fit_forward_mapping(self, input, target):
        self.F.fit(input, target)

    def fit_inverse_mapping(self, input, epsilon):
        F, G = self.F, self.G
        fit_targets = input
        if epsilon > 0:
            epsilon = np.random.randn(*fit_targets.shape) * epsilon
            fit_targets = fit_targets + epsilon
        fit_inputs = F(fit_targets)
        G.fit(fit_inputs, fit_targets)

    def __repr__(self):
        return "TPLayer(F={}, G={})".format(self.F, self.G)

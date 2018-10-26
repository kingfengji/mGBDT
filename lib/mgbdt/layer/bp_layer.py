class BPLayer:
    """Back Porpopgation Layer
    """
    def __init__(self, F):
        self.F = F
        self.input_size = F.input_size
        self.output_size = F.output_size

    def forward(self, input):
        return self.F(input)

    def backward(self, input, target, target_lr):
        gradient = self.F.gradient(input, target)
        out = input - target_lr * gradient
        return out

    def fit_forward_mapping(self, input, target):
        self.F.fit(input, target)

    def calc_loss(self, pred, target):
        return self.F.calc_loss(pred, target)

    def __repr__(self):
        return "BPLayer(F={})".format(self.F)

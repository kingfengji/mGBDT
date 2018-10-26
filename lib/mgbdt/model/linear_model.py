import torch
import torch.optim as optim

from ..utils.torch_utils import use_gpu, numpy_to_torch, torch_to_numpy, get_torch_loss


def _create_loss_fn(name, loss_params):
    if loss_params is None:
        loss_params = {}
    else:
        loss_params = loss_params.copy()
    if "size_average" in loss_params:
        loss_params.pop("size_average")
    return get_torch_loss(name, size_average=False, **loss_params)


class LinearModel:
    def __init__(self,
            input_size,
            output_size,
            bias=True,
            learning_rate=0.1,
            loss=None,
            loss_params=None,
            optimizer="SGD",
            activation=None):
        self.input_size = input_size
        self.output_size = output_size
        self.bias = bias
        if loss is None:
            self.loss_fn = _create_loss_fn("MSELoss", loss_params)
        else:
            self.loss_fn = _create_loss_fn(loss, loss_params)
        self.learning_rate = learning_rate

        self.F = torch.nn.Linear(input_size, output_size, bias=bias)
        if activation is not None:
            self.activation = getattr(torch.nn, activation)()
        else:
            self.activation = None
        if optimizer == "SGD":
            self.optimizer = optim.SGD(self.F.parameters(), lr=learning_rate, momentum=0, dampening=0, weight_decay=0, nesterov=False)
        elif optimizer == "RMSprop":
            self.optimizer = optim.RMSprop(self.F.parameters(), lr=learning_rate)
        elif optimizer == "Adam":
            self.optimizer = optim.Adam(self.F.parameters(), lr=learning_rate)
        else:
            raise ValueError()
        if use_gpu():
            self.F.cuda()

    def __repr__(self):
        return "LinearModel(input_size={}, output_size={}, learning_rate={:.3f}, loss_fn={}, activation={}, optimizer={})".format(
                self.input_size, self.output_size, self.learning_rate, self.loss_fn, self.activation, self.optimizer)

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def _predict(self, input):
        out = self.F(input)
        if self.activation is not None:
            out = self.activation(out)
        return out

    def predict(self, input):
        assert input.shape[1] == self.input_size, "input.shape[1] ({}) != self.input_size ({})".format(input.shape[1], self.input_size)
        input = numpy_to_torch(input)
        pred = self._predict(input)
        return torch_to_numpy(pred)

    def fit(self, input, target):
        assert input.shape[1] == self.input_size, "input.shape[1] ({}) != self.input_size ({})".format(input.shape[1], self.input_size)
        input = numpy_to_torch(input)
        target = numpy_to_torch(target)
        pred = self._predict(input)
        loss = self.loss_fn(pred, target) / len(pred)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def gradient(self, input, target):
        input = numpy_to_torch(input, requires_grad=True)
        target = numpy_to_torch(target)
        pred = self._predict(input)
        loss = self.loss_fn(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        return torch_to_numpy(input.grad)

    def calc_loss(self, pred, target):
        pred = numpy_to_torch(pred)
        target = numpy_to_torch(target)
        loss = self.loss_fn(pred, target) / len(pred)
        return torch_to_numpy(loss)[0]


def test_regression():
    # Test
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn import datasets
    np.random.seed(0)
    torch.manual_seed(0)

    boston = datasets.load_boston()
    X, y = boston.data, boston.target
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    X = (X - x_mean) / x_std
    print("X.shape={}, y.shape={}".format(X.shape, y.shape))
    model = LinearModel(input_size=13, output_size=1)
    for _ in range(10):
        model.fit(X, y)
    y_pred = model.predict(X)
    fig, ax = plt.subplots()
    ax.scatter(y, y_pred, edgecolors=(0, 0, 0))
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()


def test_classification():
    # Test
    import numpy as np
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    np.random.seed(0)
    torch.manual_seed(0)

    n_samples = 15000
    x_all, y_all = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.04, random_state=0)
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.3, random_state=0, stratify=y_all)
    print("x_train.shape={}, y_train.shape={}".format(x_train.shape, y_train.shape))

    model = LinearModel(input_size=2, output_size=2, learning_rate=0.1, loss="CrossEntropyLoss")
    for _ in range(50):
        model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred.argmax(axis=1))
    print("acc={:.2f}".format(acc * 100))

    import IPython
    IPython.embed()


if __name__ == "__main__":
    test_classification()

import numpy as np
import six
from torch.autograd import Variable
import torch
import torch.nn as nn


GPU_FLAG = False


def set_gpu_flag(flag):
    global GPU_FLAG
    GPU_FLAG = flag


def use_gpu():
    if not GPU_FLAG:
        return False
    return torch.cuda.is_available()


def numpy_to_torch(data, **kwargs):
    if data.dtype == np.int32 or data.dtype == np.int64:
        tensor = torch.LongTensor(data.astype(np.int64))
    else:
        tensor = torch.FloatTensor(data)
    if use_gpu():
        tensor = tensor.cuda()
    return Variable(tensor, **kwargs)


def torch_to_numpy(data):
    if use_gpu():
        return data.data.cpu().numpy()
    else:
        return data.data.numpy()


def get_torch_loss(loss, *args, **kwargs):
    if isinstance(loss, six.string_types):
        return getattr(nn, loss)(*args, **kwargs)
    return loss


class TorchLossWrapper:
    def __init__(self, loss, *args, **kwargs):
        self.loss_name = loss
        self.loss_fn = get_torch_loss(loss, *args, size_average=False, **kwargs)

    def forward(self, input, target):
        input = numpy_to_torch(input)
        target = numpy_to_torch(target)
        out = self.loss_fn(input, target)
        out = torch_to_numpy(out) / len(input)
        if out.shape == (1,):
            out = out[0]
        return out

    def backward(self, input, target):
        input = numpy_to_torch(input, requires_grad=True)
        target = numpy_to_torch(target)
        loss = self.loss_fn(input, target)
        loss.backward()
        out = torch_to_numpy(input.grad)
        return out

    def __repr__(self):
        return "TorchLossWrapper(loss_name={})".format(self.loss_name)

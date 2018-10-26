from mgbdt.utils.torch_utils import TorchLossWrapper


def get_loss(name, *args, **kwargs):
    return TorchLossWrapper(name, *args, **kwargs)

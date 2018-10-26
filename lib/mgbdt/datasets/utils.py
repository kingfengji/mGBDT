import os.path as osp


def get_dataset_base():
    return osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir, osp.pardir, "datasets"))

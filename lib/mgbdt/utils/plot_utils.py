import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import os.path as osp
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def check_dir(path):
    if not osp.exists(osp.dirname(path)):
        os.makedirs(osp.dirname(path))


def plot2d(X, color, save_path=None):
    plt.scatter(X[:, 0], X[:, 1], c=color, s=4, cmap=plt.cm.Spectral)
    if save_path is None:
        plt.show()
    else:
        print("[plot2d] Saving figure in {} ...".format(save_path))
        check_dir(save_path)
        plt.savefig(save_path)
        plt.clf()
        plt.close('all')


def plot3d(X, color, save_path=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    if save_path is None:
        plt.show()
    else:
        print("[plot3d] Saving figure in {} ...".format(save_path))
        check_dir(save_path)
        plt.savefig(save_path)
        plt.clf()
        plt.close('all')


def plot2d_pca(X, y, save_path=None, colors=None):
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(X)
    if colors is None:
        colors = np.asarray(['black', 'blue', 'red', 'purple', 'white', 'lime', 'cyan', 'orange', 'gray', 'yellow'])
    plt.scatter(x_pca[:, 0], x_pca[:, 1], s=5, color=colors[y[: x_pca.shape[0]]])
    if save_path is None:
        plt.show()
    else:
        print("[plot3d] Saving figure in {} ...".format(save_path))
        check_dir(save_path)
        plt.savefig(save_path)
        plt.clf()
        plt.close('all')
    return pca


def plot_img(img, save_path=None):
    if len(img.shape) == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    if len(img.shape) == 2:
        plt.imshow(img, cmap=plt.cm.Greys_r)
    else:
        plt.imshow(img)
    if save_path is None:
        plt.show()
    else:
        print("[plot3d] Saving figure in {} ...".format(save_path))
        check_dir(save_path)
        plt.savefig(save_path)
        plt.clf()
        plt.close('all')


def plot2d_pca_ext(X, y, save_path=None):
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(X)
    plt.scatter(x_pca[:, 0], x_pca[:, 1], s=5, c=y, cmap=plt.cm.jet)
    if save_path is None:
        plt.show()
    else:
        print("[plot3d] Saving figure in {} ...".format(save_path))
        check_dir(save_path)
        plt.savefig(save_path)
        plt.clf()
        plt.close('all')
    return pca


def plot2d_pca_tsne(X, y, save_path=None, pca_dimension=None, colors=None, subsamples=None):
    if subsamples is not None:
        X = X[subsamples]
        y = y[subsamples]
    if pca_dimension is not None:
        pca = PCA(n_components=pca_dimension)
        X = pca.fit_transform(X)
    if X.shape[1] == 2:
        x_enc = X
    else:
        tsne = TSNE(n_components=2, verbose=True)
        x_enc = tsne.fit_transform(X)
    if colors is None:
        plt.scatter(x_enc[:, 0], x_enc[:, 1], s=5, c=y, cmap=plt.cm.jet)
    else:
        colors = np.asarray(colors)
        plt.scatter(x_enc[:, 0], x_enc[:, 1], s=5, c=colors[y])
    if save_path is None:
        plt.show()
    else:
        print("[plot3d] Saving figure in {} ...".format(save_path))
        check_dir(save_path)
        plt.savefig(save_path)
        plt.clf()
        plt.close('all')


def plot_curve(df, train_col="train", test_col="test"):
    df[train_col].rename("train").plot(style="-", linewidth=1, color="red")
    df[test_col].rename("test").plot(style="--", linewidth=2, color="blue")
    plt.legend(fontsize="xx-large")
    plt.show()

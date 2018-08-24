import numpy as np

# - fun with spiral cluster dataset
def spiral_pts(k, npts):
    """Generate a spiral dataset

    Arguments:
    k: number of arms in the spiral set
    npts: number of points per arm

    Returns:
    - the dataset as a [(k * npts), 2] numpy array
    - the dataset as a 3d numpy array [npts, 2, k]
    - the labels as a one column matrix
    """
    sigma = np.array([[0.1, 0], [0, 1]])
    base = np.exp(5*np.random.rand(npts, 2, k)/2 - 2)
    def spiralize(x2d, offset=0):
        ls = np.sqrt(np.sum(x2d**2, axis=1))
        thetas = np.arctan(x2d[:, 1] / x2d[:, 0])
        xs = ls * np.cos(ls + thetas + offset)
        ys = ls * np.sin(ls + thetas + offset)
        return np.c_[xs, ys]
    dataset_split = np.zeros([npts, 2, k])
    dataset = np.empty([0, 2])
    labels = np.array([])
    for ik in range(k):
    	dataset_split[:, :, ik] = spiralize(np.dot(base[:, :, ik], sigma),
    										offset = ik * 2 * np.pi / k)
    	dataset = np.concatenate([dataset, dataset_split[:, :, ik]])
    	labels = np.append(labels, ik * np.ones(npts))
    return dataset, dataset_split, labels.reshape(-1, 1)

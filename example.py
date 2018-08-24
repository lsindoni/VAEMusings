import vaes
import numpy as np
from spiral import spiral_pts

# - Prepare dataset (no seeding, yet)
def prepare_data(dataset, labels, ratio=0.8):
    m = dataset.shape[0]
    perm = np.random.permutation(m)
    shuffled = dataset[perm]
    shuffled_labels = labels[perm]
    cut = int(np.ceil(m * ratio))
    train_set, test_set = shuffled[:cut, :], shuffled[cut:, :]
    train_labels, test_labels = shuffled_labels[:cut, :], shuffled_labels[cut:, :]
    return train_set, test_set, train_labels, test_labels

# - this is just for testing, nothing really happens
if __name__ == "__main__":
    # - A simple example with a five-armed spiral dataset
    original_dim = 2
    latent_dim = 2
    g_layers = [8, 16, 8] # - generative layers
    r_layers = [6, 12, 6] # - recognition layers
    # - generate data
    dataset, dataset_split, labels = spiral_pts(5, 500)
    train_set, test_set, train_labels, test_labels = prepare_data(dataset, labels)
    print('VAE')
    # - create the vae
    vae = vaes.vae(original_dim, latent_dim, r_layers, g_layers)
    vae.compile(optimizer='adam')
    # - Train
    vae.fit(train_set, epochs=30, batch_size=20) # some random numbers
    # - Generate new data points similar to the given data
    test_preds = vae.predict(test_set)[0] # predict returns the means and the (log) stds
    print('cVAE')
    # - create the cVAE
    cvae = vaes.cvae(1, 2, 2, [4, 5, 4], [5, 6, 5], [6, 7, 6])
    cvae.compile(optimizer='adam')
    # - train
    cvae.fit([train_labels, train_set], epochs=30, batch_size=20)
    # - predict new data points similar to the test_data
    test_preds = cvae.predict([(test_labels), test_set])[0]

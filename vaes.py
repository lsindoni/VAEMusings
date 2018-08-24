import numpy as np
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input
from blocks import *

def vae(indim, latdim, g_layers, r_layers):
    """Create a regular VAE
    indim: input dimension
    latdim: latent dimension
    g_layers: dimensions for the layers of the generative model
    r_layers:

    returns a Keras Model with few additional attributes:
    - encoder: exposes the encoder Model
    - decoder: exposes the decoder Model
    - rt: exposes the sampled encoder Model with the reparametrization trick
    - sample: method to sample from randomly generated latent features (using prior)
    """
    encoder = neural_net_gaussian(indim, latdim, r_layers)
    decoder = neural_net_gaussian(latdim, indim, g_layers)
    rt = gaussian_rep_trick(latdim)
    # - This gives latent mu, logsigma
    latmusigma = encoder(encoder.inputs)
    # - This gives samples of latent z, given the inputs, sampled with q
    bw_pass = rt(latmusigma)
    # - This gives mean and covariances of the generative models
    fw_pass = decoder(bw_pass)
    to_train = Model(encoder.inputs, [latmusigma[0],latmusigma[1], fw_pass[0], fw_pass[1]])
    # - KL
    aux = - latmusigma[1] + K.exp(2 * latmusigma[1]) - 1 + K.square(latmusigma[0])
    aux = 0.5 * aux
    aux = K.sum(aux, axis=-1)
    kl = K.mean(aux)
    # - LL
    ll = K.square(encoder.inputs - fw_pass[0])
    ll = -ll * K.exp(- 2 * fw_pass[1])
    ll = ll - fw_pass[1]
    ll = K.sum(ll, axis=-1)
    ll = K.mean(ll)
    elbo = ll - kl
    # - VAE
    vae = Model(encoder.inputs, decoder(rt(encoder(encoder.inputs))))
    vae.add_loss(-elbo)  # - Keras minimizes
    # - Expose the inner models. This will enable encoding/decoding operations
    vae.encoder = encoder
    vae.decoder = decoder
    vae.rt = rt
    # - We want to be able to sample
    def sample_model(n_pts):
    	zs = np.random.randn(n_pts, latdim)
    	return decoder.predict(zs)
    vae.sample = sample_model
    return vae


def cvae(d_x, d_y, d_z, layers1, layers2, layers3):
    """Create a Keras Model implementing the conditional VAE

    Arguments:
    d_x: dimension of the input (conditioning) data
    d_y: dimension of the output (predicted) data
    d_z: dimension of the latent variables
    layers1: layers of the conditional recognition model
    layers2: layers of the recognition network
    layers3: layers of the generative model

    Output:
    The Keras Model with the auxiliary attributes:
    - qz_xy: the recognition network
    - py_z: the generative model
    - pz_x: the conditional recognition model
    """
    # - p(z | x) / nn1
    layers1 = [8]
    nn1 = neural_net_gaussian(d_x, d_z, layers1)
    # - q(z | x, y) / nn2
    layers2 = [9]
    nn2 = neural_net_gaussian(d_x + d_y, d_z, layers2)
    # - p(y | z) / nn3
    layers3 = [10]
    nn3 = neural_net_gaussian(d_z, d_y, layers3)
    # - variational_distro
    q = stack(combined_inputs(d_x, d_y), nn2)
    # - decoder
    py_z = nn3
    # - conditional encoder
    pz_x = nn1
    # # Kullback-Leibler between gaussian models outer layers
    def kl(g_layer1, g_layer2):
        (mu1, ls1), (mu2, ls2) = g_layer1, g_layer2
        var1 = K.exp(2 * ls1)
        var2 = K.exp(2 * ls2)
        aux = 0.5 * (K.square(mu1 - mu2) + var1 - var2)
        aux = aux * K.exp(-2 * ls2)
        aux = aux + ls2 - ls1
        aux = K.sum(aux, axis=-1)
        return K.mean(aux)

    # # Log-likelihood for layers
    def ll(input_x_layer, decoder_layer):
        ll = K.square(input_x_layer - decoder_layer[0])
        ll = - ll * K.exp(-2 * decoder_layer[1])
        ll = K.sum(ll, axis=-1)
        ll = K.mean(ll)
        return ll

    inputs_x = Input(shape=(d_x,))
    inputs_y = Input(shape=(d_y,))
    _variational_distro = q([inputs_x, inputs_y])
    _cond_enc = pz_x(inputs_x)
    _grt = gaussian_rep_trick(d_z)(_variational_distro)
    _forward_pass = py_z(_grt)
    _kl = kl(_variational_distro, _cond_enc)
    _ll = ll(inputs_y, _forward_pass)

    cvae = Model([inputs_x, inputs_y], [_forward_pass[0], _forward_pass[1], _cond_enc[0], _cond_enc[1]])
    cvae.add_loss(_kl - _ll)

    cvae.qz_xy = q
    cvae.py_z = py_z
    cvae.pz_x = pz_x
    return cvae

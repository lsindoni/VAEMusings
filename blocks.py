import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Lambda, concatenate
# -
def combined_inputs(dx, dy):
	"""concatenate"""
	inputs = [Input(shape=(dx,)), Input(shape=(dy,))]
	return Model(inputs, concatenate(inputs))

# - pick two layers and return the two concatenated and the first
def splitter(dx, dy):
	"""Pick two layers and return the two concatenated and the first"""
	inputs = [Input(shape=(dx,)), Input(shape=(dy,))]
	return Model(inputs, [concatenate(inputs), inputs[0]])

# - Stack two models on top of each other
def stack(model1, model2):
	"""Stack two models on top of each other"""
	inputs = model1.inputs
	return Model(inputs, model2(model1(inputs)))

# - reparametrization trick
def sampling(args):
	z_mean, z_log_sigma = args
	batch = K.shape(z_mean)[0]
	dim = K.int_shape(z_mean)[1]
	epsilon = K.random_normal(shape=(batch, dim))
	return z_mean + K.exp(z_log_sigma) * epsilon

def gaussian_rep_trick(latdim):
	"""Reparametrization trick for gaussian variables"""
	imu = Input(shape=(latdim,))
	isigma = Input(shape=(latdim,))
	rt = Lambda(sampling)([imu, isigma])
	return Model([imu, isigma], rt)

# - gaussian nn model
def neural_net_gaussian(idim, odim, layers):
	inputs = Input(shape=(idim,))
	x = inputs
	for l in layers:
		x = Dense(l, activation='relu')(x)
	pred_mu = Dense(odim, activation='linear')(x)
	pred_log_sigma = Dense(odim, activation='linear')(x)
	return Model(inputs, [pred_mu, pred_log_sigma])

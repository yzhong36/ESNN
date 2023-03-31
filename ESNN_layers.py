import tensorflow as tf
import numpy as np
import sys
import random
from scipy.stats import norm
import os
from sklearn.model_selection import train_test_split
import tensorflow_probability as tfp
import math


#helper functions
def normal_logprob(x, u, sigma):
    return -tf.math.log(tf.sqrt(2*math.pi)) - tf.math.log(sigma) - (x - u)**2/(2*(sigma)**2)

def normal_pdf(x, u, sigma):
    return tf.exp(-0.5*((x - u)/sigma)**2)/(tf.sqrt(2*math.pi)*sigma)

def normal_cdf(x, u, sigma):
    z = (x-u)/(sigma*tf.sqrt(2.0))
    return 0.5*(1 + tf.math.erf(z))


def getprob(w_eta, w_alpha):
    spike_and_slab = tf.nn.softmax(w_eta)
    mixture = tf.nn.softmax(w_alpha)*spike_and_slab[:,1:]
    prbs = tf.concat([spike_and_slab[:,:1], mixture], axis = 1)
    return prbs

def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = tf.random.uniform(shape,minval=0,maxval=1)
    return -tf.math.log(-tf.math.log(U + eps) + eps)

def gumbel_softmax_sample(logits, nsample, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(np.concatenate([[nsample], np.asarray(np.asarray(logits).shape)]))
    # y = logits + sample_gumbel((nsample, logits.shape[0], logits.shape[1]))
    return tf.nn.softmax( y / temperature)

def gumbel_softmax(logits, nsample, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, nsample, temperature)
    if hard:
        k = tf.shape(logits)[-1]
        #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
        y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1, keepdims = True)),y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y


def scaled_softmax(logits):
    return tf.exp(logits)/tf.reduce_sum(tf.exp(logits))


#Bayesian Neural Network Layer
class BNNGroupLayer(tf.keras.layers.Layer):
    #Initialization
    def __init__(self, input_size, output_size, n_output, cov_traits, temperature, tau, init_val):
        """
        """
        super(BNNGroupLayer, self).__init__()
        #Number of features
        self.input_size = input_size
        #Number of hidden neurons of the first layer
        self.output_size = output_size
        #
        self.n_output = n_output
        #
        self.tau = tau
        #
        self.temperature = temperature
        #Free-parameters
        #probability of being the one trial
        if len(init_val)>0:
            self.w_alpha = tf.Variable(initial_value= init_val)
        else:
            self.w_alpha = tf.Variable(tf.random.truncated_normal([input_size, self.n_output], mean=0.0, stddev=0.1, dtype=tf.dtypes.float32))
        if self.n_output > 1:
            # cov matrix across traits by default (only for test)
            #self.cov = np.repeat(cov_traits, 2*self.n_output).reshape(self.n_output, self.n_output)
            #np.fill_diagonal(self.cov, 1)
            #self.cov_decomp = np.linalg.cholesky(self.cov)
            #self.cov = tf.convert_to_tensor(self.cov, dtype=tf.dtypes.float32)
            #self.cov_decomp = tf.convert_to_tensor(self.cov_decomp, dtype=tf.dtypes.float32)

            #means of slab
            self.w_mean = tf.Variable(tf.random.truncated_normal([input_size, output_size, self.n_output], mean=0.0, stddev=0.1, dtype=tf.dtypes.float32))
            
            if cov_traits is None:
                print("sampling a cov matrix ...")
                cov_sampler = tfp.distributions.WishartTriL(df=self.n_output+5, scale_tril=tf.eye(self.n_output),
                                                            input_output_cholesky = False, allow_nan_stats = False)
                cov_traits = tf.linalg.inv(cov_sampler.sample(sample_shape = (input_size, output_size)))
                #cov_sampler_cho = cov_sampler.sample(sample_shape = (input_size, output_size))

                #self.cov = tf.Variable(tf.transpose(tf.reshape(tf.repeat(tf.repeat(cov_traits,output_size),input_size),
                #                              (cov_traits.shape[0],cov_traits.shape[1],output_size,input_size))))
                self.cov_decomp = tf.Variable(tf.transpose(tf.linalg.cholesky(cov_traits), perm=[0,1,3,2]))
                #self.cov_decomp = tf.Variable(tf.transpose(cov_sampler_cho, perm=[0,1,3,2]))

            #log sigma of slab
            else:
                print("have a pre-defined cov matrix ...")
                cov_traits = tf.convert_to_tensor(cov_traits, dtype=tf.dtypes.float32)
                self.cov = tf.transpose(tf.reshape(tf.repeat(tf.repeat(cov_traits,output_size),input_size),
                                                    (cov_traits.shape[0],cov_traits.shape[1],output_size,input_size)))
                self.cov_decomp = tf.transpose(tf.linalg.cholesky(self.cov), perm=[0,1,3,2])
                
            #self.w_rho = tf.Variable(tf.random.truncated_normal([input_size, output_size, self.n_output, self.n_output], mean=0.0, stddev=0.1, dtype=tf.dtypes.float32))

        else:
            #means of slab
            self.w_mean = tf.Variable(tf.random.truncated_normal([input_size, output_size], mean=0.0, stddev=0.1, dtype=tf.dtypes.float32))
            #log sigma of slab
            self.w_rho = tf.Variable(tf.random.truncated_normal([input_size, output_size], mean=0.0, stddev=0.1, dtype=tf.dtypes.float32))

    def sample_gamma(self, logits, nsample = 1):
        #sample from gumbel
        samples_gamma = gumbel_softmax(logits, nsample, self.temperature, True)
        return samples_gamma

    def sample_w(self, nsample = 1):
        if self.n_output > 1:
            eps = tf.random.normal((nsample, self.input_size, self.output_size, self.n_output), mean=0.0, stddev = 1.0)
            w = tf.add(self.w_mean, tf.squeeze(tf.matmul(eps[:,:,:,None,:], self.cov_decomp), axis = -2))

            """ tmp_w = np.mean(w, axis = 0)
            for i in range(self.n_output):
                print("trait",i+1,": ")
                print(tmp_w[163,0,i])
                print(self.w_mean[163,0,i])
            
            print("w_1_1 cov: ",np.cov(tf.transpose(w[:,163,0,:]))) """
            #print("w_1_2 cov: ",np.cov(tf.transpose(w[:,0,1,:])))
            #print("w_1_3 cov: ",np.cov(tf.transpose(w[:,0,2,:])))

            #all_samples_gamma = self.sample_gamma(self.w_alpha[:,0]/self.tau, nsample)
            map_gamma = lambda alpha: self.sample_gamma(alpha, nsample)
            all_samples_gamma = tf.convert_to_tensor(tf.map_fn(map_gamma, tf.transpose(self.w_alpha/self.tau)))
            all_samples_gamma = tf.transpose(all_samples_gamma, perm = [1, 2, 0])
            #print("all_samples_gamma: ",all_samples_gamma)

            mask = all_samples_gamma
            mask = tf.reshape(tf.repeat(mask, self.output_size, axis = 1), (mask.shape[0], mask.shape[1], self.output_size, mask.shape[2]))
            #mask = tf.reshape(tf.repeat(mask, self.n_output), (mask.shape[0], mask.shape[1], mask.shape[2], self.n_output))
            #print("mask: ",mask)
            #print("w: before: ",w)

            w = tf.multiply(mask, w)
            #print("w after: ",w)
            #klw = self.kl_mv_w(self.cov)
            cov = tf.matmul(tf.transpose(self.cov_decomp,perm=[0,1,3,2]), tf.transpose(self.cov_decomp,perm=[0,1,2,3]))
            #print("cov: ",cov)
            klw = self.kl_mv_w(cov)
            
            #print("w_alpha shape: ", tf.shape(self.w_alpha))
            prbs = tf.convert_to_tensor(tf.map_fn(scaled_softmax, tf.transpose(self.w_alpha/self.tau)))
            #print("prbs shape: ",tf.shape(prbs))
            kl = tf.convert_to_tensor(tf.map_fn(self.kl_gamma, prbs))
            #print("kl shape before: ", tf.shape(kl))
            kl = tf.reduce_sum(kl)
            #print("kl shape after: ", tf.shape(kl))
            
            prbs = tf.transpose(prbs)
            #print("after prbs shape: ",tf.shape(prbs))
            #print(prbs)

            #print("klw shape: ", tf.shape(klw))

            tmp = tf.reshape(tf.repeat(prbs, self.output_size, axis = 0),(prbs.shape[0],self.output_size,prbs.shape[1]))
            #print("after prbs shape: ",tf.shape(prbs))
            #print(prbs)

            tmp = tf.transpose(tmp, perm = [2,0,1])
            #print("tmp: ", tmp)
            #print("klw: ", klw)
            kl += tf.reduce_sum(klw*tmp)

        else:
            #sample from posterior
            #reparam for the slab
            #derive sigma
            w_sigma = tf.exp(self.w_rho)
            #sample standard normal noise
            eps = tf.random.normal((nsample, self.input_size, self.output_size), mean=0.0, stddev = 1.0)
            #derive w using normal reparam trick
            w = tf.add(self.w_mean, tf.multiply(w_sigma, eps))
            #mask out
            all_samples_gamma = self.sample_gamma(self.w_alpha[:,0]/self.tau, nsample)
            mask = all_samples_gamma
            mask = tf.reshape(tf.repeat(mask, self.output_size), (mask.shape[0], mask.shape[1], self.output_size))
            w = tf.multiply(mask, w)
            klw = self.kl_w(w_sigma)

            prbs = scaled_softmax(self.w_alpha[:,0]/self.tau)
            kl = tf.reduce_sum(self.kl_gamma(prbs))
            tmp = tf.reshape(tf.repeat(prbs, self.output_size), (prbs.shape[0], self.output_size))
            kl += tf.reduce_sum(klw*tmp)
        
        return  all_samples_gamma, w, kl

    def kl_mv_w(self, cov):
        return 0.5*(tf.squeeze(tf.matmul(self.w_mean[:,:,None,:],self.w_mean[:,:,:,None])) + tf.linalg.trace(cov) - self.n_output - tf.linalg.logdet(cov))

    def kl_w(self, w_sigma):
        return 0.5*(w_sigma**2 + self.w_mean**2 - 1 - tf.math.log(w_sigma**2))


    def kl_gamma(self, prbs):
        log_q_gamma = tf.math.log(prbs+1e-20)
        kl_g = prbs*(log_q_gamma - tf.math.log(1.0/self.input_size))
        return kl_g

    def call(self, sample = False, nsample = 1):
        #if training then sample
        if sample:
            samples_gamma, samples_w, kl = self.sample_w(nsample)
            return samples_gamma, samples_w, kl
        else:
            prbs = scaled_softmax(self.w_alpha[:,0])
            prbs = tf.reshape(tf.repeat(prbs, self.output_size), (prbs.shape[0], self.output_size))
            w = tf.multiply(prbs, self.w_mean)
            return w


#Bayesian Sparse Multi-Layer Perceptron
class SNN(tf.keras.Model):
    def __init__(self, model_type, reg_type, sigma, input_size, output_size, cov_traits, hidden_sizes, temperature, tau, joint, init_val):
        """
        """
        super(SNN, self).__init__()
        #model type: classification or regression
        self.model_type = model_type
        #reg type: logistic or probit
        self.reg_type = reg_type
        #sigma: noise sd
        self.sigma = sigma
        #number of features
        self.input_size = input_size
        #list store number of hidden sizes
        self.hidden_sizes = hidden_sizes
        #
        self.output_size = output_size
        #Bayesian layer
        self.bnn = BNNGroupLayer(self.input_size, self.hidden_sizes[0], output_size, cov_traits, temperature, tau, init_val)
        #
        self.joint = joint
        #bias for first hidden layer and later layers
        self.mylayers = list()
        if self.output_size > 1:
            self.mylayers.append(tf.Variable(tf.random.truncated_normal([self.output_size, self.hidden_sizes[0]], mean=0.0, stddev=0.1, dtype=tf.dtypes.float32)))
        else:
            self.mylayers.append(tf.Variable(tf.random.truncated_normal([self.hidden_sizes[0]], mean=0.0, stddev=0.1, dtype=tf.dtypes.float32)))
        for i in range(1, len(self.hidden_sizes)):
            self.mylayers.append(tf.keras.layers.Dense(hidden_sizes[i], use_bias=True, activation=None))
        if self.model_type == 'classification':
            self.dist_n = tfp.distributions.Normal(loc = 0.0, scale = 1.0)
            self.mylayers.append(tf.keras.layers.Dense(1, use_bias=False, activation=None))
        else:
            self.mylayers.append(tf.keras.layers.Dense(1, use_bias=True, activation=None))
    """
    """
    def call(self, x, y, sample = True, nsample = 1):
        y_dim = y.shape
        if len(y_dim) == 1:
            y = tf.transpose(tf.reshape(tf.repeat(y, nsample), (y.shape[0], nsample)))
            y = tf.reshape(y, (y.shape[0], y.shape[1], 1))
        else:
            y = tf.transpose(tf.reshape(tf.repeat(y, nsample, axis = 1), (-1,nsample)))
            y = tf.reshape(y, (nsample, -1, y_dim[1]))

        kl = 0
        if sample:
            samples_gamma, w, tmp = self.bnn.call(sample, nsample)
            kl += tmp
        else:
            w = self.bnn.call(sample, nsample)
        
        if self.output_size > 1:
            # x: N, J   w: NN, J, H1, D => NN, N, D, H1
            C = tf.tensordot(x, w, axes = ([1],[1]))
            C = tf.transpose(C, perm = [1,0,3,2])
        else:
            C = tf.matmul(x, w)
        x = C + self.mylayers[0]
        x = tf.nn.relu(x)
        for i in range(1, len(self.hidden_sizes)):
           x = self.mylayers[i](x)
           x = tf.nn.relu(x)
        if self.model_type == 'classification' and self.reg_type == 'probit':
            probits = self.mylayers[len(self.hidden_sizes)](x)
            eps = tf.random.normal(probits.shape, mean=0.0, stddev = 1.0)
            probits_n = probits + eps*self.sigma
            if not self.joint:
                probability = self.dist_n.cdf(probits_n)
                bce = tf.keras.losses.BinaryCrossentropy()
                nll = bce(y, probability)
                return probits, probability, nll, kl
            return probits, kl
        elif self.model_type == 'classification' and self.reg_type == 'logistic':
            logits = self.mylayers[len(self.hidden_sizes)](x)
            if not self.joint:
                bce = tf.keras.losses.BinaryCrossentropy()
                nll = bce(y, probability)
                return logits, probability, nll, kl
            return logits, kl
        else:
            pred = self.mylayers[len(self.hidden_sizes)](x)
            if self.output_size > 1:
                pred = tf.squeeze(pred)
            if not self.joint:
                # mse for likelihood
                nll = tf.reduce_mean(tf.losses.MSE(y, pred))
                return pred, nll, kl
            return pred, kl


class ESNN(tf.keras.Model):
    def __init__(self, L, model_type, reg_type, sigma, input_size, hidden_sizes, temperature, tau, init_vals):
        """
        """
        super(ESNN, self).__init__()
        self.models = list()
        self.model_type = model_type
        self.all_cs = list()
        self.L = L
        for i in range(L):
            self.models.append(SNN(model_type, reg_type, sigma, input_size, hidden_sizes, temperature, tau, True, init_vals[i]))
    def call(self, x, y, sample, nsample, l):
        pred, kl = self.models[0].call(x, y, sample, nsample)
        for i in range(1, l+1):
            ###add for cs
            subx = np.copy(x)
            if len(self.all_cs)>0:
                if len(self.all_cs) == 1:
                    toremove = np.unique(self.all_cs[0])
                else:
                    toremove = np.unique(np.concatenate(self.all_cs[:i]))
                # x[:,toremove] *= 0
                subx = np.delete(x, toremove, axis = 1)
            subx = tf.convert_to_tensor(subx, dtype = tf.float32)
            ###add for cs
            t1, t2 = self.models[i].call(subx, y, sample, nsample)
            pred += t1
            kl += t2
        if self.model_type == 'classification':
            y = tf.transpose(tf.reshape(tf.repeat(y, nsample), (y.shape[0], nsample)))
            y = tf.reshape(y, (y.shape[0], y.shape[1], 1))
            probability = tf.nn.sigmoid(pred)
            bce = tf.keras.losses.BinaryCrossentropy()
            nll = bce(y, probability)
            return pred, probability, nll, kl
        else:
            y = tf.transpose(tf.reshape(tf.repeat(y, nsample), (y.shape[0], nsample)))
            y = tf.reshape(y, (y.shape[0], y.shape[1], 1))
            nll = tf.reduce_mean(tf.losses.MSE(y, pred))
            return pred, nll, kl




def accuracy(probability, y):
    nsample = probability.shape[0]
    y = tf.transpose(tf.reshape(tf.repeat(y, nsample), (y.shape[0], nsample)))
    y = tf.reshape(y, (y.shape[0], y.shape[1], 1))
    pred = np.zeros(probability.shape)
    pred[np.where(probability>0.5)]=1
    return np.mean(pred == y)

def train_bnn(model, x, y, batch_size, learning_rate, sample, nsample, lamb, l1):
    nbatch = round(y.shape[0]/batch_size)
    for i in range(nbatch):
        temp_id = batch_size*i + np.array(range(batch_size))
        temp_x = x[np.min(temp_id):(np.max(temp_id)+1), :]
        temp_y = y[np.min(temp_id):(np.max(temp_id)+1),]
        #update fixed params
        if model.model_type == 'classification':
            with tf.GradientTape() as tape:
                logits, probability, nll, kl = model.call(temp_x, temp_y, sample, nsample)
                elbo = nll+kl*lamb
            gradients=tape.gradient(elbo, model.trainable_variables)
        else:
            with tf.GradientTape() as tape:
                pred, nll, kl = model.call(temp_x, temp_y, sample, nsample)
                elbo = nll+kl*lamb
            gradients=tape.gradient(elbo, model.trainable_variables)
        optimizer = tf.optimizers.Adam(learning_rate = learning_rate*l1)
        optimizer.apply_gradients(zip([gradients[0]], [model.trainable_variables[0]]))
        optimizer = tf.optimizers.Adam(learning_rate = learning_rate)
        optimizer.apply_gradients(zip(gradients[1:], model.trainable_variables[1:]))



def train_bnn_joint(model, x, y, batch_size, learning_rate, sample, nsample, lamb, l1, l):
    nbatch = round(y.shape[0]/batch_size)
    for i in range(nbatch):
        temp_id = batch_size*i + np.array(range(batch_size))
        temp_x = x[np.min(temp_id):(np.max(temp_id)+1), :]
        temp_y = y[np.min(temp_id):(np.max(temp_id)+1)]
        #update fixed params
        if model.model_type == 'classification':
            with tf.GradientTape() as tape:
                logits, probability, nll, kl = model.call(temp_x, temp_y, sample, nsample, l)
                elbo = nll+kl*lamb
            gradients=tape.gradient(elbo, model.models[l].trainable_variables)
        else:
            with tf.GradientTape() as tape:
                pred, nll, kl = model.call(temp_x, temp_y, sample, nsample, l)
                elbo = nll+kl*lamb
            gradients=tape.gradient(elbo, model.models[l].trainable_variables)
        optimizer = tf.optimizers.Adam(lr = learning_rate*l1)
        optimizer.apply_gradients(zip([gradients[0]], [model.models[l].trainable_variables[0]]))
        optimizer = tf.optimizers.Adam(lr = learning_rate)
        optimizer.apply_gradients(zip(gradients, model.models[l].trainable_variables))

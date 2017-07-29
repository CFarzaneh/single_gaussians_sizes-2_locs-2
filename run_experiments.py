import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython import embed
#from vae import VariationalAutoencoder
from VAE_Models.VAE import VAE as model
from VAE_Models.architectures import DNN

# Load Data
synthData = np.load("data/single_gaussians_sizes=2_locs=2.npy")
synthDataLabels = np.load("data/single_gaussians_sizes=2_locs=2_labels.npy")
n_samples, input_dim = synthData.shape
winSize = int(np.sqrt(input_dim)) # The window size

np.random.seed(92090)
tf.set_random_seed(16399)


def exp1(overwrite=False):
    """Vanilla autoencoder. Gaussian prior and posterior, Bernoulli likelihood."""
    print("\nRunning VAE with two dimensional latency space...\n")

    latency_dim = 2
    theEncoder = [200]*2 # num neurons in each layer of encoder network
    theDecoder = [200]*2 # num neurons in each layer of generator network

    training_epochs=3
    batch_size=200
    learning_rate=0.0001

    encoder = DNN(theEncoder, tf.nn.relu)
    decoder = DNN(theDecoder, tf.nn.relu)

    hyperParams = {'reconstruct_cost': 'bernoulli',
                   'learning_rate': learning_rate,
                   'optimizer': tf.train.AdamOptimizer,
                   'batch_size': batch_size
                  }

    '''
    network_architecture = \
        dict(n_hidden_recog=encoder, # num neurons in each layer of encoder network
            n_hidden_gener=decoder,  # num neurons in each layer of generator network
            n_input=input_dim, # (img shape: 28*28)
            n_z=latency_dim)  # dimensionality of latent space
    '''

    filename = 'experiments/latency_dim=%d/enc=%s_dec=%s_epochs=%d_batches=%d_opt=adam_learnRate=%s' \
                % (latency_dim, '-'.join([str(l) for l in theEncoder]),
                   '-'.join([str(l) for l in theDecoder]),
                   training_epochs, batch_size, float('%.4g' % learning_rate))
    if not os.path.exists(os.path.join(filename,'latency_space_while_training')):
        os.makedirs(os.path.join(filename,'latency_space_while_training'))
    if os.path.exists('./'+filename+'/exp.meta') and not overwrite:
        vae = model(input_dim, encoder, latency_dim, decoder, hyperParams)
        new_saver = tf.train.import_meta_graph('./'+filename+'/exp.meta')
        new_saver.restore(vae.sess, tf.train.latest_checkpoint('./'+filename+'/'))
    else:

        #vae = VariationalAutoencoder(network_architecture,
                                   # learning_rate=learning_rate,
                                   # batch_size=batch_size)

        vae = model(input_dim, encoder, latency_dim, decoder, hyperParams)

        numPlots = 0
        nx = ny = 20
        rnge = 4
        x_values = np.linspace(-rnge, rnge, nx)
        y_values = np.linspace(-rnge, rnge, ny)
        interval = 10 # save fig of latency space every 10 batches

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            avg_cost2 = 0.
            total_batch = int(n_samples / batch_size)
            # Loop over all batches
            canvas = np.empty((winSize*ny,winSize*nx))
            for i in range(total_batch):
                batch_xs = synthData[i*batch_size:(i+1)*batch_size]

                # Fit training using batch data
                #cost = vae.partial_fit()
                cost, reconstr_loss, KL_loss = vae(batch_xs)
                # Compute average loss
                avg_cost += cost / n_samples * batch_size
                avg_cost2 += cost / (batch_size*interval) * batch_size


                if i%interval == 0:
                    fig = plt.figure()
                    axes = fig.add_subplot(1,2,1)
                    axes2 = fig.add_subplot(1,2,2)
                    axes.set_aspect('equal')
                    axes.set_title("%d Iterations" % (epoch*n_samples + i*batch_size))
                    axes2.set_title("Average Cost = %f" % (avg_cost2))
                    avg_cost2 = 0.
                    axes.set_xlim((-rnge,rnge))
                    axes.set_ylim((-rnge,rnge))
                    test_xs = synthData[19000:20000]
                    test_ys = synthDataLabels[19000:20000]
                    #means, stds = vae.getLatentParams(test_xs)
                    means, stds = vae.transform(test_xs)
                    latent_xs = means + stds*np.random.normal(size=stds.shape)
                    axes.scatter(latent_xs[:,0], latent_xs[:,1], edgecolor='k', c=test_ys)

                    ws = winSize
                    for i, yi in enumerate(x_values):
                        for j, xi in enumerate(y_values):
                            z_mu = np.array([[xi, yi]]*vae.batch_size)
                            x_mean = vae.generate(z_mu)
                            canvas[(nx-i-1)*ws:(nx-i)*ws, j*ws:(j+1)*ws] = x_mean[0].reshape(ws, ws)

                    axes2.imshow(canvas, origin="upper")
                    plt.tight_layout()

                    fig.savefig(filename+'/latency_space_while_training/fig%04d.png' % numPlots)
                    numPlots += 1
                    plt.close()

            # Display logs per epoch step
            print("Epoch:", '%04d' % (epoch), "cost=", "{:.9f}".format(avg_cost))

        vae.saver.save(vae.sess, filename+'/exp.meta')


def exp2(overwrite=False):
    """Vanilla autoencoder. Gaussian prior and posterior, Bernoulli likelihood."""
    print("\nRunning VAE with one dimensional latency space...\n")

    latency_dim = 1
    theEncoder = [500]*2 # num neurons in each layer of encoder network
    theDecoder = [500]*2 # num neurons in each layer of generator network

    training_epochs=2
    batch_size=100
    learning_rate=0.0001

    encoder = DNN(theEncoder, tf.nn.relu)
    decoder = DNN(theDecoder, tf.nn.relu)

    hyperParams = {'reconstruct_cost': 'bernoulli',
                   'learning_rate': learning_rate,
                   'optimizer': tf.train.AdamOptimizer,
                   'batch_size': batch_size
                  }

    '''
    network_architecture = \
        dict(n_hidden_recog=encoder, # num neurons in each layer of encoder network
            n_hidden_gener=decoder,  # num neurons in each layer of generator network
            n_input=input_dim, # (img shape: 28*28)
            n_z=latency_dim)  # dimensionality of latent space
    '''
    
    filename = 'experiments/latency_dim=%d/enc=%s_dec=%s_epochs=%d_batches=%d_opt=adam_learnRate=%s' \
                % (latency_dim, '-'.join([str(l) for l in theEncoder]),
                   '-'.join([str(l) for l in theDecoder]),
                   training_epochs, batch_size, float('%.4g' % learning_rate))
    if not os.path.exists(os.path.join(filename,'latency_space_while_training')):
        os.makedirs(os.path.join(filename,'latency_space_while_training'))
    if os.path.exists('./'+filename+'/exp.meta') and not overwrite:
        #vae = VariationalAutoencoder(network_architecture)
        vae = model(input_dim, encoder, latency_dim, decoder, hyperParams)
        new_saver = tf.train.import_meta_graph('./'+filename+'/exp.meta')
        new_saver.restore(vae.sess, tf.train.latest_checkpoint('./'+filename+'/'))
    else:
        '''
        vae = VariationalAutoencoder(network_architecture,
                                    learning_rate=learning_rate,
                                    batch_size=batch_size)
        '''
        vae = model(input_dim, encoder, latency_dim, decoder, hyperParams)
        numPlots = 0
        nx = 20
        x_values = np.linspace(-8, 8, nx)
        interval = 10 # save fig of latency space every 10 batches

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            avg_cost2 = 0.
            total_batch = int(n_samples / batch_size)
            # Loop over all batches
            canvas = np.empty((winSize,winSize*nx))
            for i in range(total_batch):
                batch_xs = synthData[i*batch_size:(i+1)*batch_size]

                # Fit training using batch data
                #cost = vae.partial_fit(batch_xs, batch_xs)
                cost, reconstr_loss, KL_loss = vae(batch_xs)
                # Compute average loss
                avg_cost += cost / n_samples * batch_size
                avg_cost2 += cost / (batch_size*interval) * batch_size


                if i%interval == 0:
                    fig = plt.figure()
                    axes = fig.add_subplot(2,1,1)
                    axes2 = fig.add_subplot(2,1,2)
                    #axes.set_aspect('equal')
                    axes.set_title("%d Iterations" % (epoch*n_samples + i*batch_size))
                    axes2.set_title("Average Cost = %f" % (avg_cost2))
                    avg_cost2 = 0.
                    axes.set_xlim((-8,8))
                    #axes.set_ylim((-8,8))
                    test_xs = synthData[10000:20000]
                    test_ys = synthDataLabels[10000:20000]
                    #mean, std = vae.getLatentParams(test_xs)
                    mean, std = vae.transform(test_xs)
                    latent_xs = mean + std*np.random.normal(size=std.shape)
                    bins = np.linspace(-8,8,100)
                    x1 = latent_xs[test_ys==0]
                    #embed()
                    axes.hist(x1, bins, alpha=0.5, label='right-small')
                    x2 = latent_xs[test_ys==1]
                    axes.hist(x2, bins, alpha=0.5, label='right-large')
                    x3 = latent_xs[test_ys==2]
                    axes.hist(x3, bins, alpha=0.5, label='left-small')
                    x4 = latent_xs[test_ys==3]
                    axes.hist(x4, bins, alpha=0.5, label='left-large')
                    axes.legend(loc='upper right')

                    ws = winSize
                    for i, xi in enumerate(x_values):
                        z_mu = np.array([[xi]]*vae.batch_size)
                        x_mean = vae.generate(z_mu)
                        canvas[:, (nx-i-1)*ws:(nx-i)*ws] = x_mean[0].reshape(ws, ws)

                    axes2.imshow(canvas, origin="upper")
                    plt.tight_layout()

                    fig.savefig(filename+'/latency_space_while_training/fig%04d.png' % numPlots)
                    numPlots += 1
                    plt.close()

            # Display logs per epoch step
            print("Epoch:", '%04d' % (epoch), "cost=", "{:.9f}".format(avg_cost))

        vae.saver.save(vae.sess, filename+'/exp.meta')


if __name__ == "__main__":

    exp2(True)

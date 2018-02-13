%matplotlib inline
import tensorflow as tf
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')

# input to the discriminator and generator
# In GAN's the generater produces the fake image while we provide the real image. The role of discriminator is to
# identify the image as real or fake.In order to fool the discriminator, the generator makes better images to fool it
# In this way the generator learns to produce images that are similar to the real world.
def model_inputs(real_dim, z_dim):
    inputs_real = tf.placeholder(tf.float32, (None, real_dim), name = 'inputs_real')                # input to the discriminator
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name = 'inputs_z')                   # input to the generator

    return inputs_real, inputs_z

# input z is a random white noise vector that we feed into the generator, which learns to convert this in image
# leaky relu are similar to normal relu but have small fraction for negative numbers
# tanh has outputs from negative one to one, which means we have to rescale out images
def generator(z, out_dim, n_units=128, reuse=False,  alpha=0.01):
    # we want to make the variables in our generator and discriminator differently, we can do this with name scope also.
    # but sometimes we have to create the network over again so we can reuse the variable names from previous creation
    # if we don't reuse the variables we'll create totally new variables, we use same variables for fake and real images
    with tf.variable_scope('generator', reuse=reuse): # finish this
        # Hidden layer
        h1 = tf.layers.dense(z, n_units, activation = None)
        # Leaky ReLU
        h1 = tf.maximum(alpha*h1, h1)

        # Logits and tanh output on the generator
        logits = tf.layers.dense(h1, out_dim, activation = None)
        out = tf.tanh(logits)

        return out

def discriminator(x, n_units=128, reuse=False, alpha=0.01):

    with tf.variable_scope('discriminator', reuse=reuse): # finish this
        # Hidden layer
        h1 =tf.layers.dense(x, n_units, activation = None)
        # Leaky ReLU
        h1 = tf.maximum(alpha*h1, h1)

        logits = tf.layers.dense(h1, 1, activation = None)
        out = tf.sigmoid(logits)

        return out, logits

# HYPERPARAMETERS
# when we have a non-linear hidden layer in the network then it can be a universal function approximator
# hence we are using hidden layers with non linear activation i.e., leaky relu's

#Size of input image to discriminator
input_size = 784 # 28x28 MNIST images flattened
# Size of latent vector to generator
z_size = 100
# Sizes of hidden layers in generator and discriminator
g_hidden_size = 128
d_hidden_size = 128
# Leak factor for leaky ReLU
alpha = 0.01
# Label smoothing
smooth = 0.1

# BUILDING THE NETWORK
tf.reset_default_graph()
# Create our input placeholders
input_real, input_z = model_inputs(input_size, z_size)
# input_real is real input, input_z is random noise, input_size is 784, z_size is 100

# Generator network here
g_model = generator(input_z, input_size, n_units = g_hidden_size, alpha = alpha)
# g_model is the generator output

# Disriminator network here
d_model_real, d_logits_real = discriminator(input_real, n_units=d_hidden_size, alpha=alpha)
d_model_fake, d_logits_fake = discriminator(g_model, reuse=True, n_units=d_hidden_size, alpha=alpha)


# DISCRIMINATOR AND GENERATOR LOSSES
# we train the generator and discriminator networks at the same time

# for discriminator the loss is sum of losses for the real and fake images
# the labels are reduced a bit from 1.0 to 0.9, also known as label smoothing
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_logits_real) * (1 - smooth)))

d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_logits_real)))

d_loss = d_loss_real + d_loss_fake

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_logits_fake)))

# we want to update our generator and discriminator variables seperately

# OPTIMIZERS
learning_rate = 0.002

# we use the variable_scope to name all of our generator variables so we can use 'generator' name in seperate list
# Get the trainable_variables, split into G and D parts
t_vars = tf.trainable_variables()
g_vars = [variable for variable in t_vars if variable.name.startswith('generator')]
d_vars = [variable for variable in t_vars if variable.name.startswith('discriminator')]

d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list = d_vars)
g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list = g_vars)

batch_size = 100
epochs = 100
samples = []
losses = []
saver = tf.train.Saver(var_list = g_vars)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for ii in range(mnist.train.num_examples//batch_size):
            batch = mnist.train.next_batch(batch_size)
            
            # Get images, reshape and rescale to pass to D
            batch_images = batch[0].reshape((batch_size, 784))
            batch_images = batch_images*2 - 1
            
            # Sample random noise for G
            batch_z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            
            # Run optimizers
            _ = sess.run(d_train_opt, feed_dict={input_real: batch_images, input_z: batch_z})
            _ = sess.run(g_train_opt, feed_dict={input_z: batch_z})
        
        # At the end of each epoch, get the losses and print them out
        train_loss_d = sess.run(d_loss, {input_z: batch_z, input_real: batch_images})
        train_loss_g = g_loss.eval({input_z: batch_z})
            
        print("Epoch {}/{}...".format(e+1, epochs),
              "Discriminator Loss: {:.4f}...".format(train_loss_d),
              "Generator Loss: {:.4f}".format(train_loss_g))    
        # Save losses to view after training
        losses.append((train_loss_d, train_loss_g))
        
        # Sample from generator as we're training for viewing afterwards
        sample_z = np.random.uniform(-1, 1, size=(16, z_size))
        gen_samples = sess.run(
                       generator(input_z, input_size, n_units=g_hidden_size, reuse=True, alpha=alpha),
                       feed_dict={input_z: sample_z})
        samples.append(gen_samples)
        saver.save(sess, './checkpoints/generator.ckpt')

# Save training generator samples
with open('train_samples.pkl', 'wb') as f:
    pkl.dump(samples, f)

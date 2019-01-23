
"""
Simple GANs with MNIST_data
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

#import easy MNIST data
mnist=input_data.read_data_sets("MNIST_data")
tf.reset_default_graph()

#Generator fun
def Generator(z,reuse=None):
    with tf.variable_scope('gen',reuse=reuse):
        hidden_1=tf.layers.dense(inputs=z,units=128,activation=tf.nn.leaky_relu)
        hidden_2=tf.layers.dense(inputs=hidden_1,units=128,activation=tf.nn.leaky_relu)
        output=tf.layers.dense(inputs=hidden_2,units=784,activation=tf.nn.tanh)
        return output

#Discriminator fun
def Discriminator(X,reuse=None):
    with tf.variable_scope('dis',reuse=reuse):
        hidden_1=tf.layers.dense(inputs=X,units=128,activation=tf.nn.leaky_relu)
        hidden_2=tf.layers.dense(inputs=hidden_1,units=128,activation=tf.nn.leaky_relu)
        logits=tf.layers.dense(hidden_2,units=1)
        output=tf.sigmoid(logits)
        return output,logits

#create placeholder for x and z
real_images=tf.placeholder(tf.float32,shape=[None,784])
z=tf.placeholder(tf.float32,shape=[None,100])

G=Generator(z)
D_output_real,D_logits_real=Discriminator(real_images)
D_output_fake,D_logits_fake=Discriminator(G,reuse=True)

#loss fun
def loss_func(logits_in,labels_in):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_in,labels=labels_in))

D_real_loss=loss_func(D_logits_real,tf.ones_like(D_logits_real)*0.9) #Smoothing for generalization
D_fake_loss=loss_func(D_logits_fake,tf.zeros_like(D_logits_real))
D_loss=D_real_loss+D_fake_loss

G_loss= loss_func(D_logits_fake,tf.ones_like(D_logits_fake))

learning_rate=0.001

tvars=tf.trainable_variables()  #returns all variables
d_vars=[var for var in tvars if 'dis' in var.name]
g_vars=[var for var in tvars if 'gen' in var.name]

#two optimizer for D and G
D_trainer=tf.train.AdamOptimizer(learning_rate=0.001).minimize(D_loss,var_list=d_vars)
G_trainer=tf.train.AdamOptimizer(learning_rate=0.001).minimize(G_loss,var_list=g_vars)

#please only try to train with GPUs
batch_size=180
epochs=1800
init=tf.global_variables_initializer()
samples=[] #generator samples

#start tf sesion to exucate operations
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        num_batches=mnist.train.num_examples//batch_size
        for i in range(num_batches):
            batch=mnist.train.next_batch(batch_size)
            batch_images=batch[0].reshape((batch_size,784))
            batch_images=batch_images*2-1
            batch_z=np.random.uniform(-1,1,size=(batch_size,100))
            _=sess.run(D_trainer,feed_dict={real_images:batch_images,z:batch_z})
            _=sess.run(G_trainer,feed_dict={z:batch_z})

        print("on epoch{}".format(epoch))

        sample_z=np.random.uniform(-1,1,size=(1,100))
        gen_sample=sess.run(Generator(z,reuse=True),feed_dict={z:sample_z})

        samples.append(gen_sample)

#print samples if you want
plt.imshow(samples[0].reshape(28,28))

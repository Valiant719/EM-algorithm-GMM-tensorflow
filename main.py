import matplotlib
matplotlib.use("Pdf")
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
import tensorflow as tf
from model import EM_GMM


flags = tf.app.flags
flags.DEFINE_float("learning_rate", 0.1, "Learning rate of adam optimizer [0.1]")
flags.DEFINE_float("decay_rate", 0.96, "Decay rate of learning rate [0.96]")
flags.DEFINE_integer("max_epochs", 100, "Maximum of Epochs [100]")
flags.DEFINE_integer("num_clusters", 3, "The dimension of latent variable [3]")
flags.DEFINE_boolean("use_GD", False, "False for EM-algorithm, True for gradient descent [False]")
FLAGS = flags.FLAGS

def generate_data(num_clusters, num_steps):
    vector_values = []
    for i in np.random.choice(num_clusters, num_steps):
      if i == 0:
        vector_values.append([np.random.normal(1, 0.7),
                              np.random.normal(1, 0.7)])
      elif i == 1:
        vector_values.append([np.random.normal(4, 0.5),
                              np.random.normal(8, 0.6)])
      else:
        vector_values.append([np.random.normal(8, 0.7),
                             np.random.normal(4, 0.8)])
    return vector_values

def main(_):
    """generate data"""
    x_data = generate_data(FLAGS.num_clusters,100)

    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

        EM = EM_GMM(sess,x_data, num_clusters=FLAGS.num_clusters,
                   learning_rate=FLAGS.learning_rate,decay_rate=FLAGS.decay_rate,
                   max_epochs=FLAGS.max_epochs,use_GD=FLAGS.use_GD)
        EM.train()

if __name__ == '__main__':
    tf.app.run()

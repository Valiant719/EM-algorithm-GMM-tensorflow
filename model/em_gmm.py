import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# to see in 2-dimension
input_dim = 2

def gaussian_2d(x, y, x0, y0, xsig, ysig):
        return 1/(2*np.pi*xsig*ysig) * np.exp(-0.5*(((x-x0) / xsig)**2 + ((y-y0) / ysig)**2))

class EM_GMM(object):
    def __init__(self,sess,x_data,num_clusters=3,learning_rate=0.1,decay_rate=0.96,
                max_epochs=100,use_GD=False):
        self.sess = sess
        
        self.x_data = x_data   
        self.num_clusters = num_clusters
        self.input_dim = input_dim
        
        self.use_GD = use_GD
        
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.max_epochs = max_epochs
        self.step = tf.Variable(0, trainable=False)  
        self.lr = tf.train.exponential_decay(
            learning_rate, self.step, 10000, decay_rate, staircase=True, name="lr")
        
        _ = tf.scalar_summary("learning rate", self.lr)
        
        self._attrs = ["use_GD","num_clusters","learning_rate", "decay_rate", "max_epochs"]
       
        self.build_model()
      
    def build_model(self):
        self.x = tf.placeholder(tf.float32, [None,self.input_dim], name="input")
        
        self.mu = tf.get_variable('mu',[self.num_clusters,self.input_dim],initializer=tf.random_normal_initializer(mean=np.mean(self.x_data)))
        self.cov = tf.get_variable('cov',[self.num_clusters,self.input_dim,self.input_dim],initializer=tf.constant_initializer(np.tile(np.identity(self.input_dim),[self.num_clusters,1,1])))
        self.prior = tf.get_variable('prior',[self.num_clusters],initializer=tf.constant_initializer(1/float(self.num_clusters)))
        
        p = []
        for k in xrange(self.num_clusters):
            dist = tf.contrib.distributions.MultivariateNormal(mu=tf.gather(self.mu,k), sigma=tf.gather(self.cov,k))
            p1_k = dist.pdf(self.x)
            p2_k = tf.gather(self.prior,k) * p1_k
            p.append(p2_k)
        sum_p = sum(p)
        self.NLL = -tf.reduce_sum(tf.log(sum_p))

        _ = tf.scalar_summary("negative log likelihood", self.NLL)

        self.optim =  tf.train.AdamOptimizer(learning_rate=self.lr) \
                 .minimize(self.NLL, global_step=self.step,var_list=[self.mu,self.cov,self.prior])

        for k in xrange(self.num_clusters):
            _ = tf.histogram_summary("mu_"+str(k), tf.gather(self.mu,k))
    
    def E_step(self):
        """ E-step 
        Calculate responsibility.
        Variable:
            resp: list of p(z_k|x), [p(z_0|x),p(z_1|x), ... , p(z_num_clusters)]
            p : list of p(z_k)p(x|z_k)
            p1_k : p(x|z_k) : Multivariate normal distribution
            p2_k : p(z_k)p(x|z_k)
        """

        self.resp = []
        p = []
        for k in xrange(self.num_clusters):
            dist = tf.contrib.distributions.MultivariateNormal(mu=tf.gather(self.mu,k), sigma=tf.gather(self.cov,k))
            p1_k = dist.pdf(self.x)
            p2_k = tf.gather(self.prior,k) * p1_k
            p.append(p2_k)
        sum_p = sum(p)  
        self.resp = [i/sum_p for i in p]
       
    def M_step(self):

        """ M-step
        Variable:
            N_k: (sigma over all samples) p(z_k|x)
        """    
        new_mu = []
        new_cov = []
        new_prior = []
        for k in xrange(self.num_clusters):
            N_k = tf.reduce_sum(self.resp[k])

            new_mu_k = tf.squeeze(tf.matmul(tf.expand_dims(self.resp[k],0),self.x)) / N_k
            new_mu.append(new_mu_k)

            new_cov_k = tf.batch_matmul(tf.expand_dims(self.x-tf.gather(self.mu,k),-1),tf.expand_dims(self.x-tf.gather(self.mu,k),1))
            new_cov_k = tf.matmul(tf.expand_dims(self.resp[k],0),tf.reshape(new_cov_k,[-1,2*self.input_dim]))
            new_cov_k = tf.reshape(new_cov_k,[self.input_dim,self.input_dim]) / N_k
            new_cov.append(new_cov_k)

            new_prior_k = N_k/tf.cast(tf.shape(self.x)[0],tf.float32)
            new_prior.append(new_prior_k)
        new_mu = tf.pack(new_mu)
        new_cov = tf.pack(new_cov)
        new_prior = tf.pack(new_prior)
    
        #self.mu = new_mu
        #self.cov = new_cov
        #self.prior = new_prior
        self.sess.run([self.mu.assign(new_mu),self.cov.assign(new_cov),
                       self.prior.assign(new_prior)],feed_dict={self.x:self.x_data})

        
    def train(self):
        merged_sum = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter("./logs/%s" % self.get_model_dir(), self.sess.graph)
        
        tf.initialize_all_variables().run()
        
        start_time = time.time()
        start_iter = self.step.eval()
        
        feed_dict={self.x:self.x_data}
        
        for step in xrange(start_iter, start_iter + self.max_epochs):
            writer.add_summary(self.sess.run(merged_sum,feed_dict), step)
            
            if step % 10 == 0:
                self.plotGMM()
                if self.use_GD:
                    self.sess.run(self.optim,feed_dict)
                    self.sess.run(self.prior.assign(tf.squeeze(tf.nn.softmax(tf.expand_dims(self.prior,0)))))
                else:
                    self.E_step()
                    self.M_step()
                NLL = self.sess.run(self.NLL,feed_dict)
                print("Step: [%4d/%4d] time: %4.4f, loss: %.8f" \
                    % (step, 100, time.time() - start_time, NLL))

    def plotGMM(self):
        delta = 0.025
        x = np.arange(-2, 12, delta)
        y = np.arange(-2, 12, delta)
        X, Y = np.meshgrid(x, y)
 
        for i in xrange(self.num_clusters):
            mu,cov = self.sess.run([self.mu,self.cov],feed_dict={self.x:self.x_data})
            Z1 = gaussian_2d(X, Y, mu[i][0], mu[i][1], cov[i][0][0], cov[i][1][1])
            plt.contour(X, Y, Z1, linewidths=0.5)
        
        plt.plot([v[0] for v in self.x_data],[v[1] for v in self.x_data],'.', markersize=3)
        for i in xrange(self.num_clusters):
            mu = self.sess.run(self.mu,feed_dict={self.x:self.x_data})
            plt.plot(mu[i][0],mu[i][1],'+', markersize=13, mew=3)
        plt.show()
        
    def get_model_dir(self):
        model_dir = ''
        for attr in self._attrs:
            if hasattr(self, attr):
                model_dir += "/%s=%s" % (attr, getattr(self, attr))
        return model_dir
    



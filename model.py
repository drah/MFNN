# -*- coding: utf-8 -*-
from numpy.random import randint
from numpy import eye
import tensorflow as tf

class tf_base:
    def __init__(self):
        self._sess = None

    def __del__(self):
        if self._sess != None:
            self._sess.close() 

    def openSession(self):
        if self._sess == None:
            self._sess = tf.Session()

    def closeSession(self):
        if self._sess != None:
            self._sess.close()
        self._sess = None

    def saveCkpt(self, path, global_step):
        self._saver.save(self._sess, path, global_step)

    def loadCkpt(self, path):
        self._saver.restore(self._sess, path)

    def setOptimizer(self, alg, learning_rate):
        if alg == 'GradientDescent':
            opt = tf.train.GradientDescentOptimizer(learning_rate)
        elif alg == 'AdadeltaOptimizer':
            opt = tf.train.AdadeltaOptimizer(learning_rate)
        elif alg == 'AdagradOptimizer':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif alg == 'MomentumOptimizer':
            opt = tf.train.MomentumOptimizer(learning_rate)
        elif alg == 'AdamOptimizer':
            opt = tf.train.AdamOptimizer(learning_rate)
        elif alg == 'FtrlOptimizer':
            opt = tf.train.FtrlOptimizer(learning_rate)
        elif alg == 'RMSPropOptimizer':
            opt = tf.train.RMSPropOptimizer(learning_rate)
        else:
            print("We do not have optimizer %s ..." % alg)
            return
        self._train_step = opt.minimize(self._loss)

    def initVar(self):
        uninitialized_vars = []
        for var in tf.all_variables():
            try:
                self._sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninitialized_vars.append(var)
        init_new_vars_op = tf.initialize_variables(uninitialized_vars)
        self._sess.run(init_new_vars_op)
        #self._sess.run(tf.initialize_all_variables())

class WMF(tf_base):
    def __init__(self):
        self._sess = None

    def __del__(self):
        if self._sess != None:
            self._sess.close()

    def tune_hyper(self, p_tr, c_tr, l, p, c, m, max_iter, batch_size=50):
        self.I = eye(p.shape[0])
        iter_loss = {}
        vali_err = {}
        for i in range(max_iter):
            if i % 100 == 0:
                iter_loss[i] = self._print_loss(p, c, l, i, batch_size)
                vali_err[i] = self._print_vali_err(p, c, m, i)
            self._set_feed(p_tr, c_tr, l, batch_size)
            self._sess.run(self._train_step, feed_dict=self._feed)
        iter_loss[i+1] = self._print_loss(p, c, l, i, batch_size)
        vali_err[i+1] = self._print_vali_err(p, c, m, i)
        return iter_loss, vali_err

    def fit(self, p, c, l, iteration=1000, batch_size=50, verbose=False):
        self.I = eye(p.shape[0])
        for i in range(iteration):
            if verbose and (i % 100 == 0):
                self._print_loss(p, c, l, i, batch_size)
            self._set_feed(p, c, l, batch_size)
            self._sess.run(self._train_step, feed_dict=self._feed)
        self._print_loss(p, c, l, iteration, batch_size)

    def _set_feed(self, p, c, l, batch_size): 
        sampled_index = randint(0, p.shape[0], batch_size)
        self._feed = {self._index: self.I[sampled_index, :], \
                      self._p: p[sampled_index, :], \
                      self._c: c[sampled_index, :], \
                      self._lambda: l}

    def _print_loss(self, p, c, l, it, batch_size):
        result = 0.0
        for i in range(p.shape[0] // batch_size):
            head = i * batch_size
            tail = (i+1) * batch_size
            result += self._sess.run(self._weighted_loss, \
                feed_dict={self._index: self.I[head:tail, :], \
                           self._p: p[head:tail, :], \
                           self._c: c[head:tail, :], \
                           self._lambda: l})
        print("iteration %d weighted loss: %f" % (it, result))
        return result

    def _print_vali_err(self, p, c, m, it):
        vali_err = self._sess.run(self._vali_err, \
            feed_dict={self._p: p, self._c: c, self._mask: m})
        print("iteration %d validation error: %f" %(it, vali_err))
        return vali_err

    def predict(self):
        return self._sess.run([self._A, self._B])

    def drawGraph(self, n_row, n_latent, n_col):
        with tf.name_scope('matDecomp'):
            self._p = tf.placeholder(tf.float32, shape=[None, n_col])
            self._c = tf.placeholder(tf.float32, shape=[None, n_col])
            self._lambda = tf.placeholder(tf.float32)
            self._index = tf.placeholder(tf.float32, shape=[None, n_row])
            self._A = tf.Variable(tf.truncated_normal([n_row, n_latent]))
            self._B = tf.Variable(tf.truncated_normal([n_latent, n_col]))
            self._h = tf.matmul(tf.matmul(self._index, self._A), self._B) 
            
            weighted_loss = tf.reduce_mean(tf.mul(self._c, tf.squared_difference(self._p, self._h)))
            self._weighted_loss = weighted_loss
            l2_A = tf.reduce_sum(tf.square(self._A))
            l2_B = tf.reduce_sum(tf.square(self._B))
            n_w = tf.constant(n_row * n_latent + n_latent * n_col, tf.float32)
            l2 = tf.truediv(tf.add(l2_A, l2_B), n_w)
            reg_term = tf.mul(self._lambda, l2)
            self._loss = tf.add(weighted_loss, reg_term)
            
            self._mask = tf.placeholder(tf.float32, shape=[n_row, n_col])
            one = tf.constant(1, tf.float32)
            pred = tf.cast(tf.greater_equal(tf.matmul(self._A, self._B), one), tf.float32)
            cor = tf.mul(tf.cast(tf.equal(pred, self._p), tf.float32), self._c)
            self._vali_err = tf.reduce_sum(tf.mul(cor, self._mask))

            self._saver = tf.train.Saver([v for v in tf.all_variables() if v.name.find('matDecomp') != -1])
            tf.scalar_summary('training_weighted_loss_l2', self._loss)
            tf.scalar_summary('validation_weighted_loss', self._weighted_loss)
            merged = tf.merge_all_summaries()
 

class convNet(tf_base):
    def __init__(self):
        self._sess = None

    def __def__(self):
        if self._sess != None:
            self._sess.close()

    def fit(self, x, y, iteration=1000, batch_size=32, keep_prob=0.5, verbose=False):
        # could add summary_writer
        for i in range(iteration):
            if verbose and (i % 100 == 0):
                self._print_loss(i, x, y, batch_size)
            self._set_feed(x, y, batch_size, keep_prob)
            self._sess.run(self._train_step, feed_dict = self._feed)
        self._print_loss(iteration, x, y, batch_size)

    def _set_feed(self, x, y, batch_size, keep_prob):
        sampled_index = randint(0,x.shape[0],batch_size)
        self._feed = {self._x: x[sampled_index,:], self._y: y[sampled_index,:], self._keep_prob: keep_prob}

    def _print_loss(self, it, x, y, batch_size):
        result = 0
        for i in range(x.shape[0]//batch_size):
            head = i * batch_size
            tail = (i+1) * batch_size
            result += self._sess.run(self._loss, 
                feed_dict={self._x: x[head:tail,:], self._y: y[head:tail,:], self._keep_prob:1.0})
        result /= x.shape[0]
        print("iteration %d average loss: %f" %(it, result))

    def predict(self, x, batch_size=50):
        pred = []
        for i in range(x.shape[0]//batch_size):
            head = i * batch_size
            tail = (i+1) * batch_size
            p = self._sess.run(self._h, feed_dict={self._x: x[head:tail, :], self._keep_prob:1.0})
            pred.append(p)
        return pred

    def drawGraph(self, x_shape, y_shape):
        ''' for x_cat_10_day'''
        with tf.name_scope('convNet'):
            self._x = tf.placeholder(tf.float32, shape=[None, x_shape[1], x_shape[2], x_shape[3]])
            self._y = tf.placeholder(tf.float32, shape=[None, y_shape])
            nA1 = 16
            with tf.name_scope('convA1'):
                w = self._weight_var([1, 116, x_shape[3], nA1])
                b = self._bias_var([nA1])
                A1a = tf.nn.relu(tf.nn.conv2d(self._x, w, \
                    strides=[1, 1, 116, 1], padding='VALID') + b)
            nA2 = 8
            with tf.name_scope('convA2'):
                w = self._weight_var([3, 13, nA1, nA2])
                b = self._bias_var([nA2])
                A2a = tf.nn.relu(tf.nn.conv2d(A1a, w, \
                    strides=[1, 1, 1, 1], padding='SAME') + b)
            nF1 = 128
            self._keep_prob = tf.placeholder(tf.float32)
            with tf.name_scope('fullyNet_drop_1'):
                flat = tf.reshape(A2a, [-1,3*13*nA2])
                w = self._weight_var([3*13*nA2, nF1])
                b = self._bias_var([nF1])
                F1a = tf.nn.relu(tf.matmul(flat, w) + b)
                F1a_drop = tf.nn.dropout(F1a, self._keep_prob)
            nF2 = 64
            with tf.name_scope('fullNet_drop_2'):
                w = self._weight_var([nF1, nF2])
                b = self._bias_var([nF2])
                F2a = tf.nn.relu(tf.matmul(F1a_drop, w) + b)
                F2a_drop = tf.nn.dropout(F2a, self._keep_prob)
            nF3 = y_shape
            with tf.name_scope('fullNet_3'):
                w = self._weight_var([nF2, nF3])
                b = self._bias_var([nF3])
                F3a = tf.nn.relu(tf.matmul(F2a_drop, w) + b)
            self._h = F3a
            self._loss = tf.reduce_mean((self._y-self._h)*(self._y-self._h))
            self._saver = tf.train.Saver([v for v in tf.all_variables() if v.name.find('convNet') != -1])

    def _weight_var(self, shape):
        init = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(init)

    def _bias_var(self, shape):
        init = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(init)
        
class convNet_WMF(tf_base):
    def __init__(self):
        self._sess = None

    def __def__(self):
        if self._sess != None:
            self._sess.close()

    def loadCkpt(self, cnn_path, wmf_path):
        self.wmf._saver.restore(self._sess, wmf_path)
        self.cnn._saver.restore(self._sess, cnn_path)

    def fine_tune(self, x, p, c, iteration, batch_size=32, keep_prob=0.5, verbose=False):
        for i in range(iteration):
            if verbose and (i % 100 == 0):
                self._print_loss(i, x, p, c, batch_size)
            self._set_feed(x, p, c, batch_size, keep_prob)
            self._sess.run(self._train_step, feed_dict=self._feed)
        self._print_loss(iteration, x, p, c, batch_size)

    def predict(self, x):
        pred = []
        for i in range(x.shape[0]//batch_size):
            head = i * batch_size
            tail = (i+1) * batch_size
            p = self._sess.run(self._h, feed_dict={
                self.cnn._x: x[head:tail, :], \
                self.cnn._keep_prob:1.0, \
                self.wmf._p: p[head:tail, :], \
                self.wmf._c: c[head:tail, :]
                })
            pred.append(p)
        return pred

    def _set_feed(self, x, p, c, batch_size, keep_prob):
        sampled_index = randint(0, x.shape[0], batch_size)
        self._feed = {self.cnn._x: x[sampled_index, :], \
                      self.cnn._keep_prob: keep_prob, \
                      self.wmf._p: p[sampled_index, :], \
                      self.wmf._c: c[sampled_index, :]}

    def _print_loss(self, it, x, p, c, batch_size):
        result = 0.0
        for i in range(x.shape[0] // batch_size):
            head = i * batch_size
            tail = (i+1) * batch_size
            result += self._sess.run(self._loss, feed_dict={ \
                self.cnn._x: x[head:tail, :], \
                self.cnn._keep_prob: 1.0, \
                self.wmf._p: p[head:tail, :], \
                self.wmf._c: c[head:tail, :]})
        print("iteration %d loss: %f" % (it, result))

    def drawGraph(self, x_shape, n_row, n_latent, n_col):
        self.cnn = convNet()
        self.wmf = WMF()
        self.cnn.drawGraph(x_shape, n_latent)
        self.wmf.drawGraph(n_row, n_latent, n_col)
        self._h = tf.matmul(self.cnn._h, self.wmf._B)
        self._loss = tf.reduce_mean(tf.mul(self.wmf._c, tf.squared_difference(self.wmf._p, self._h)))
        self._saver = tf.train.Saver()
        


# -*- coding: utf-8 -*-
import model
import numpy as np

# train pc_mat -> wmf -> A, B mat
# train x -> cnn -> A mat
# fine-tune x -> cnn -> A mat -> B mat -> y

# validation rate?
#   try 50% 70% 90%
# how to validate wmf ? 
#   pick out some of the elements in the matrix, use them as the validation error
#   then pick the optimal hyperparameters, use the whole data to train wmf.
# 
def wmf_exp():
    p = np.load('p_mat.npy')
    c = np.load('c_mat.npy')
    wmf = model.WMF()
    n_lat = 16
    wmf.drawGraph(p.shape[0], n_lat, p.shape[1])
    wmf.setOptimizer('AdagradOptimizer', 0.001)
    wmf.openSession()
    wmf.initVar()
    """
    # 50% train, 100% valid
    ind_vali = np.random.rand(p.shape[0], p.shape[1]) >= 0.5
    ind_tr = np.negative(ind_vali)
    p_tr = p * ind_tr
    c_tr = c * ind_tr
    """
    lamb = 0.001
    gstep = 1000
    for i in range(gstep):
        print("gstep %d: " % i, end="")
        wmf.fit(p, c, lamb)
        wmf.saveCkpt('wmf',i)
    wmf.saveCkpt('wmf',i+1)
    """
    iter_loss, vali_err = \
        wmf.tune_hyper(p_tr, c_tr, lamb, p, c, ind_vali, max_iter)
    min_vali_err = min(vali_err.values())
    for k, v in vali_err.items():
        if v == min_vali_err:
            print("iteration %d with vali_err %f" % (k, v))
            break
    """
    
def cnn_exp():
    x6 = np.load('x6_cat.npy') #train
    #x7 = np.load('x7_cat.npy') #validation
    #x8 = np.load('x8_cat.npy') #test  
    A = np.load('A-wmf1000.npy')
    
    cnn = model.convNet()
    cnn.drawGraph(x6.shape, A.shape[1])
    cnn.setOptimizer('AdagradOptimizer', 0.001)
    cnn.openSession()
    cnn.initVar()
    
    gstep = 15
    for i in range(gstep):
        print("gstep %d: " % i, end="")
        cnn.fit(x6, A)
        cnn.saveCkpt('cnn', i)
    cnn.saveCkpt('cnn', i+1)

def cnn_wmf_exp():
    x6 = np.load('x6_cat.npy')
    p = np.load('p_mat.npy')
    c = np.load('c_mat.npy')
    #x7 = np.load('x7_cat.npy') #validation
    #x8 = np.load('x8_cat.npy') #test
    cnn_wmf = model.convNet_WMF()
    cnn_wmf.drawGraph(x6.shape, 45000, 16, 13)
    cnn_wmf.setOptimizer('AdagradOptimizer', 0.0001)
    cnn_wmf.openSession()
    cnn_wmf.loadCkpt('cnn-15', 'wmf-1000')
    cnn_wmf.initVar()

    gstep = 10
    for i in range(gstep):
        print("gstep %d: " % i, end="")
        cnn_wmf.fine_tune(x6, p, c, 1000)
        cnn_wmf.saveCkpt('cnn_wmf', i)
    cnn_wmf.saveCkpt('cnn_wmf', i+1)

def main():
    #wmf_exp()
    #cnn_exp()
    cnn_wmf_exp()

if __name__ == '__main__':
    main()

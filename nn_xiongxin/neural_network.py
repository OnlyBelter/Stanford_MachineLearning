# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 10:47:51 2017

@author: xin
"""
# Neural Network
import numpy as np
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt


HIDDEN_LAYER_SIZE = 25
INPUT_LAYER = 400  # input feature
NUM_LABELS = 10  # output class number
mat = scipy.io.loadmat('ex4data1.mat')
X = mat.get('X')  # all samples, 5000*400
y = mat.get('y')  # all y, 5000*1

def random_show_one_pic(X):
    (row_n, col_n) = X.shape
    pic_index = np.random.randint(0, row_n)
#    pic_index = np.random.randint(0, 100)
    pic_vector = X[pic_index]
    pic = pic_vector.reshape(20, 20)
    plt.imshow(pic, cmap=plt.cm.gray)
    plt.show()

random_show_one_pic(X)
def rand_initialize_weights(L_in, L_out):
    """
    Randomly initialize the weights of a layer with L_in 
    incoming connections and L_out outgoing connections;
    
    Note that W should be set to a matrix of size(L_out, 1 + L_in) as
    the first column of W handles the "bias" terms
    """
    W = np.zeros((L_out, L_in + 1))
    epsilon_init = 0.12
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;
    return W

def sigmoid(z):
    return 1/(1+np.power(np.e, -z))
    
def sigmoid_gradient(z):
    g = np.zeros(z.shape);
    g = sigmoid(z)*(1-sigmoid(z))
    return g
    
def convert_y(y):
    # eg: convert 3 to its logical array [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    row_n, col_n = y.shape
    y_values = np.arange(10)
    y[y==10] = 0  # convert 10 to 0
    y_new = (y_values == y).astype(int)
    return y_new
    
def calculate_h(theta1, theta2, X):
    m,n = X.shape  # m is sample number, n is feature number
    a_1 = np.vstack((np.ones((1, m)), X.T))
    z_2 = np.dot(theta1, a_1)
    a_2 = np.vstack((np.ones((1, m)), sigmoid(z_2)))
    z_3 = np.dot(theta2, a_2)
    a_3 = sigmoid(z_3)
    h_3 = a_3.T
    return h_3  # m*10, probability
    
def nn_cost_function(theta1, theta2, input_layer_size,
                     hidden_layer_size, num_labels,
                     X, y, lamdba=0):
    m = X.shape[0]
    J = 0
    theta1_grad = np.zeros(theta1.shape)
    theta2_grad = np.zeros(theta2.shape)
    h_total = calculate_h(theta1, theta2, X)
    
    y_mat = convert_y(y) #  now, y is a logical matrix, m·10
#    theta1_no_t0 = theta1[:, 1:]
    theta2_no_t0 = theta2[:, 1:]
    J = (1/m)*np.sum(-y_mat*np.log(h_total) - (1-y_mat)*(np.log(1-h_total)))  
    # 计算所有参数的偏导数（梯度）
    D_1 = np.zeros(theta1.shape)  # Δ_1
    D_2 = np.zeros(theta2.shape)  # Δ_2
    for t in range(m):
        a_1 = np.hstack((np.array([[1]]),X[t:t+1,:])).T # 401*1
        # a_1 = np.concatenate((np.array([1]), X[t:t+1,:]), axis=0) 
        z_2 = np.dot(theta1, a_1) # 25*1
        a_2 = np.vstack((np.array([[1]]), sigmoid(z_2))) #  26*1
        z_3 = np.dot(theta2, a_2) # 10*1
        a_3 = sigmoid(z_3)
        h = a_3 # 列向量, 10·1
        delta_3 = h - y_mat[t:t+1,:].T  # δ_3, 10*1
        delta_2 = np.dot(theta2_no_t0.T, delta_3) * sigmoid_gradient(z_2) # 25*1
        D_2 = D_2 + np.dot(delta_3, a_2.T) # 10*26
        D_1 = D_1 + np.dot(delta_2, a_1.T) # 25*401
    theta1_grad = (1/m)*D_1  # 第一层参数的偏导数，没有加正则项
    theta2_grad = (1/m)*D_2
    return {'theta1_grad': theta1_grad, 
            'theta2_grad': theta2_grad, 
            'J': J, 'h': h_total}
    
def calculate_accuracy(y, h):
    y_est_max = np.amax(h, axis=1)
    h_mat = (h == np.matrix(y_est_max).T).astype(int)
    h_final = np.where(h_mat==1)[1]
    accu = np.sum((y_tr == np.matrix(h_final).T).astype(int))/len(h_final)
    return accu

# theta1 is 25*401, theta2 is 10*26
theta1 = rand_initialize_weights(INPUT_LAYER, HIDDEN_LAYER_SIZE)
theta2 = rand_initialize_weights(HIDDEN_LAYER_SIZE, NUM_LABELS)


X_tr = X[:4000] # training set
X_te = X[4000:] # test set
y_tr = y[:4000] # training set
y_te = y[4000:] # test set
h = calculate_h(theta1, theta2, X_tr)  # m*10, probability

iter_times = 1000
alpha = 0.3
result = []
for i in range(iter_times):
    cost_fun_result = nn_cost_function(theta1=theta1, theta2=theta2, 
                                       input_layer_size=INPUT_LAYER,
                                       hidden_layer_size=HIDDEN_LAYER_SIZE, 
                                       num_labels=NUM_LABELS,
                                       X=X_tr, y=y_tr)
    theta1_g = cost_fun_result.get('theta1_grad')
    theta2_g = cost_fun_result.get('theta2_grad')
    J = cost_fun_result.get('J')
    h_current = cost_fun_result.get('h')
    accuracy = calculate_accuracy(y_tr, h_current)
    theta1 -= alpha * theta1_g
    theta2 -= alpha * theta2_g
    result.append((i, J, accuracy))
    print(i, J, accuracy)

#J_list = []
#for j in result:
#    J_list.append(j[1])
#plt.plot(J_list)
#theta1_unroll = theta1.reshape(1, 25*401)
#theta2_unroll = theta2.reshape(1, 10*26)
#theta1_unroll_str = '\t'.join(str(_) for _ in theta1_unroll.tolist()[0])
#theta2_unroll_str = '\t'.join(str(_) for _ in theta2_unroll.tolist()[0])
#
#with open('theta.txt', 'a') as o_handle:
#    o_handle.write('theta1_unroll_str\t' + theta1_unroll_str+ '\n')
#    o_handle.write('theta2_unroll_str\t' + theta2_unroll_str+ '\n')


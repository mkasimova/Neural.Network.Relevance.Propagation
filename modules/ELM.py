import logging
import sys
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import numpy as np

logger = logging.getLogger("elm")

def pseudo_inverse(x):
    # see eq 3.17 in bishop
    try:
        inv= np.linalg.inv(np.matmul(x.T,x))
        return np.matmul(inv, x.T)
    except np.linalg.linalg.LinAlgError as ex:
        #logger.exception(ex)
        logger.debug("Singular matrix")
        #Moore Penrose inverse rule, see paper on ELM
        inv = np.linalg.inv(np.matmul(x, x.T))
        return np.matmul(x.T, inv)

def random_matrix(L,n):
    return np.random.rand(L,n)

def g_ELM(x, func_name):
    if func_name == "soft_relu": #good
        return np.log(1+np.exp(x))
    elif func_name == "hard_relu": #good if you use regularization
        x[x< 0] = 0
        return x
    elif func_name == "arctan": #also good
        return np.arctan(x)
    elif func_name == "identity":
        return x
    else:
        raise Exception("No such activation function {}".format(func_name))

class MultiLayerELMClassifier(object):
    def __init__(self, nlayers, L, activation_func):
        self.nlayers = nlayers
        self.L = L
        self.activation_func = activation_func
        self.weights = []
        self.biases = []
        self.weights_E = []
        
    def train(self, x, t):
        XE = x.T
        (N, n) = x.shape
        T = t.T
        H, H_prev = None, None
        W, b, WE = None, None, None
        for l_idx in range(self.nlayers):
           
            if l_idx == 0:
                #print"x, XE, n", x.shape, XE.shape, n)
                #print"t, T", t.shape, T.shape)
                #step 1
                WE = random_matrix(self.L, n)
                
                #print"layer 1", self.W1.shape, self.b1.shape, self.W1E.shape)
                #step 2
                H = g_ELM(np.matmul(WE,XE), self.activation_func).T
                #print"H, inv", H.shape, pseudo_inverse(H).shape)
                #step 3
                self.beta = np.matmul(pseudo_inverse(H), t)
                #print"beta", beta.shape)
            else:
                #step 4
                #print("H_prev", H_prev.shape)
                H = np.matmul(t, pseudo_inverse(self.beta))
                #print("H", H.shape)
                #step 5
                #append one column to include the bias at this layer
                dummyvec = np.zeros((N,1)) + 1 
                #print("dummy", dummyvec.shape)
                HE = np.append(dummyvec, H_prev, axis=1) 
                #print("HE", HE.shape)
                WE = np.matmul(pseudo_inverse(HE),inv_g_ELM(H, self.activation_func))
                #print("WE", WE.shape)
                b = WE[:,0]
                self.W2 = WE[:,1:]
                #print"layer 2", self.W2.shape, self.b2.shape)
                
                #step 6
                H = g_ELM(np.matmul(HE, WE), self.activation_func)
                #print"H2", H2.shape)
                
                #step 7
                self.beta = np.matmul(pseudo_inverse(H), t)
                #print"beta_new", self.beta_new.shape)
                
            W = WE[1:, :]
            b = WE[0, :]
            self.weights.append(W)
            self.weights_E.append(WE)
            b = np.expand_dims(b, axis=0)
            self.biases.append(b)
            H_prev = H
            
    def predict(self,x):
        #print"\nprediction")
        res = None
        for l_idx, W in enumerate(self.weights):
            if l_idx == 0:
                #due to the data format of X we've baked in the biases in to the first weight vector
                H = g_ELM(np.matmul(self.weights_E[l_idx],x.T), func_name=self.activation_func).T
                res = H
            else:
                res = np.matmul(res, W)
                res += self.biases[l_idx]
                res = g_ELM(res, func_name=self.activation_func)
        res = np.matmul(res, self.beta)
        return res

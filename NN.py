import numpy as np
import json



# CLASS TO SERIALIZE NUMPY ARRAYS TO SAVE MODEL TO JSON
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
    
def save_keras(nn_model):
    k = {}
    for i in range(len(nn_model.layers)):
        a = nn_model.layers[i].get_weights()
        s = list(a)

        k[str(i+1)] = {}
        k[str(i+1)]["weights"] = s[0]
        k[str(i+1)]["bias"] =s[1]
        k[str(i+1)]["activation"] = str(nn_model.layers[i].activation).split()[1]

    with open("model.json", "w", encoding="utf-8") as json_f:
        json.dump(k, json_f, cls=NumpyEncoder, ensure_ascii=False, indent=4)
    
    
class Network(object):

    
    # initialize network
    def __init__(self):
        self.architecture = {}
        self.activation_funcs = {
            "relu" :    [self.relu, self.relu_prime],
            "softmax" : [self.softmax, self.softmax_prime],
            "sigmoid" : [self.sigmoid, self.sigmoid_prime],
            "identity": [self.identity, self.identity]
        }

            
    # INITIALIZE, LOAD AND SAVE MODEL--------
            
    def initialize_parameters(self, sizes, activations):
        np.random.seed(99)
        if len(sizes) -1 != len(activations):
            print("number of activations must be equal to number of layers - 1")
        else:
            for i in range(len(sizes) -1):
                self.architecture[str(i+1)] = {
                    "weights" : np.random.rand(sizes[i], sizes[i+1])* 0.1,
                    "bias" : np.random.rand(sizes[i+1])* 0.1,
                    "activation" : activations[i]
                }
            
            
    def load_model(self):
        with open("model.json", "r") as dict_file:
            model = json.load(dict_file) 
        for layer in model:
            for key in model[layer]:
                if isinstance(model[layer][key], list):
                    model[layer][key] = np.array(model[layer][key])
        self.architecture = model.copy()
           
    
    def save_model(self):
        with open("model.json", "w", encoding="utf-8") as json_f:
            json.dump(self.architecture, json_f, cls=NumpyEncoder, ensure_ascii=False, indent=4)
        
     
    # ACTIVATION FUNCTIONS AND DERIVATIVES--
    
    def sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))
    
    def sigmoid_prime(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
    ###
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_prime(self, z):
        z[z > 0] = 1
        z[z <= 0] = 0
        return z

    ###
    
    def softmax(self, z):
        m = np.max(z, axis=1, keepdims=True)
        e = np.exp(z - m)
        return e / np.sum(e, axis=1, keepdims=True)
    
    def softmax_prime(self, z):
        return softmax(z)  * (1 - softmax(z))
    
    def identity(self, x):
        return x
    
    # ERROR-----------------------------------
    
    def cost(self, y_hat, y):
        y_hat = y_hat.reshape(y.shape)
        cost = np.sum((y_hat - y)**2) / 2.0
        return cost

    def cost_prime(self, y_hat, y):
        y_hat = y_hat.reshape(y.shape)
        return y_hat - y

    
    # METHODS---------------------------------
    
    def feedforward(self, a):
        for i in range(len(self.architecture)):
            activation = self.activation_funcs[self.architecture[str(i+1)]["activation"]][0]
            a = activation(np.dot(a, self.architecture[str(i+1)]["weights"]) + self.architecture[str(i+1)]["bias"])
        return a
    
    #def backprop(self, a):

import numpy as np
import json



# Class to serialize numpy arrays to save model to json
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    

# Function to save a keras model to Json so that it can be imported    
def save_keras(nn_model):
    k = {}
    for i in range(len(nn_model.layers)):
        a = nn_model.layers[i].get_weights()
        s = list(a)

        k[str(i+1)] = {}
        k[str(i+1)]["weight"] = s[0]
        k[str(i+1)]["bias"] =s[1]
        k[str(i+1)]["activation"] = str(nn_model.layers[i].activation).split()[1]

    with open("model.json", "w", encoding="utf-8") as json_f:
        json.dump(k, json_f, cls=NumpyEncoder, ensure_ascii=False, indent=4)
    
    
class Network():

    
    # initialize network
    def __init__(self):
        self.architecture = {}
        self.activation_funcs = {
            "relu" :    self.relu,
            "softmax" : self.softmax,
            "sigmoid" : self.sigmoid,
            "identity": self.identity        
            }

            
    # INITIALIZE, LOAD AND SAVE MODEL--------
            
    def init(self, sizes, activations):
        np.random.seed(99)
        if len(sizes) -1 != len(activations):
            print("number of activations must be equal to number of layers - 1")
        else:
            for i in range(len(sizes) -1):
                self.architecture[str(i+1)] = {
                    "weight" : np.random.rand(sizes[i], sizes[i+1])* 0.1,
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
    
    def sigmoid(self, z, deriv = False):
        if deriv == True:
            return z * (1 - z)
        else:
            return 1.0/(1.0 + np.exp(-z))
    
        
    def relu(self, z, deriv = False):
        if deriv == True:
            z[z > 0] = 1
            z[z <= 0] = 0
            return z
        else:
            return np.maximum(0, z)

        
    def softmax(self, z, deriv = False):
        if deriv == True:
            return self.softmax(z)  * (1 - self.softmax(z))
        else:
            m = np.max(z, axis=1, keepdims=True)
            e = np.exp(z - m)
            return e / np.sum(e, axis=1, keepdims=True)
    

    def identity(self, x):
        return x
    
    # ERROR-----------------------------------
    
    def cost(self, y_hat, y, deriv = False):
        if deriv == False:
            y_hat = y_hat.reshape(y.shape)
            cost = np.sum((y_hat - y)**2) / 2.0
            return cost
        else:
            y_hat = y_hat.reshape(y.shape)
            return y_hat - y

    
    # METHODS---------------------------------
    
    def predict(self, x, mem=False):
        
        memory = {}

        for i in range(len(self.architecture)):

            memory[str(i+1)] = {"z" : None,
                                "activation" : None}


            activation = self.activation_funcs[self.architecture[str(i+1)]["activation"]]
            z = np.dot(x, self.architecture[str(i+1)]["weight"]) + self.architecture[str(i+1)]["bias"]
            x = activation(z)
            
            if mem == True:
                memory[str(i+1)]["z"] = z
                memory[str(i+1)]["activation"] = x


        if mem == True:
            return x, memory
        else:
            return x

    
    
    def backprop(self, x, y):
        nablas = {}
        for i in range(len(self.architecture)):
            nablas[str(i +1)] = {"weight": np.zeros((self.architecture[str(i+1)]["weight"]).shape) ,
                            "bias" : np.zeros((self.architecture[str(i+1)]["bias"]).shape)}
        
        y_hat, memory = self.predict(x, mem=True)
        
        # Output error (delta)
        f_acti = self.activation_funcs[self.architecture[str(len(self.architecture))]["activation"]]
        Eo = (memory[str(len(self.architecture))]["activation"] - y.reshape(y_hat.shape)) * f_acti(memory[str(len(self.architecture))]["z"], deriv=True)
        
        nablas[str(len(self.architecture))]["bias"] = np.sum(Eo, axis=0, keepdims=True)
        nablas[str(len(self.architecture))]["weight"] = np.dot(Eo, memory[str(len(self.architecture))]["activation"].T)
        
        for i in nablas:
            print(i)
            print(nablas[i])

        for i in range(len(self.architecture), 1, -1):
            pass
            # acti_f = self.activation_funcs[self.architecture[str(i-1)]["activation"]]
            # Eh = np.dot(Eo, activations[-i].T) * acti_f(Zs[i-2], deriv=True)
            # nablas[str(i-1)]["weight"] = np.dot(Eh, x).T
            # nablas[str(i-1)]["bias"] = np.sum(Eh, axis=0, keepdims=True)

        return nablas
            
        
        

if __name__ == "__main__":
    net = Network()
    net.init([2, 3, 4, 1],["sigmoid","sigmoid", "sigmoid"])
    x = np.array([[2, 5], [8, 4], [4, 3]])
    y = np.array([1, 0, 1])
    net.backprop(x, y)
    
   
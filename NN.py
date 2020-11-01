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
        if len(sizes) -1 != len(activations):
            print("number of activations must be equal to number of layers - 1")
        else:
            for i in range(len(sizes) -1):
                self.architecture[str(i+1)] = {
                    "weight" : np.random.uniform(-1, 1, (sizes[i], sizes[i+1])),
                    "bias" : np.random.uniform(-1, 1, (1, sizes[i+1])),
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
            return self.sigmoid(z) * (1 - self.sigmoid(z))
        else:
            return 1.0/(1.0 + np.exp(-z))
    
        
    def relu(self, z, deriv = False):
        if deriv == True:
            return np.where(z > 0, 1, 0)
        else:
            return np.maximum(0, z)

        
    def softmax(self, z, deriv = False):
        if deriv == True:
            return self.softmax(z) * (1 - self.softmax(z))
        else:
            m = np.max(z, axis=1, keepdims=True)
            e = np.exp(z - m)
            return e / np.sum(e, axis=1, keepdims=True)
    

    def identity(self, x, deriv = False):
        return x
    

    # ERROR-----------------------------------
    
    def cost(self, y_pred, y_true, deriv = False):
        if deriv == False:
            n = y_pred.shape[1]
            cost = (1./(2*n)) * np.sum((y_true - y_pred) ** 2)
            return cost
        else:
            #cost_prime = y_pred - y_true.reshape(y_pred.shape)
            cost_prime = y_pred - y_true
            return cost_prime

    
    # METHODS---------------------------------

    def shuffle(self, a, b):
        np.random.seed()
        rnd_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rnd_state)
        np.random.shuffle(b)
        return a, b


    def dataset(self, x, y, test_size=0.8 ):
        np.random.seed()
        rnd_state = np.random.get_state()
        np.random.shuffle(x)
        np.random.set_state(rnd_state)
        np.random.shuffle(y)

        limit = int(len(x)*test_size)

        x_train = x[:limit]
        y_train = y[:limit]
        x_test = x[limit:]
        y_test = y[limit:]

        return x_train, x_test, y_train, y_test

    
    def predict(self, x, mem=False, backprop=False):
        memory = {}
        memory["0"] = {"activation" : x}
        for i in range(len(self.architecture)):
            activation = self.activation_funcs[self.architecture[str(i+1)]["activation"]]
            z = np.dot(x, self.architecture[str(i+1)]["weight"]) + self.architecture[str(i+1)]["bias"]
            x = activation(z)
            
            if mem == True:
                memory[str(i+1)] = {"z" : None,
                                    "activation" : None}
                memory[str(i+1)]["z"] = z
                memory[str(i+1)]["activation"] = x

        if backprop == True:
            if mem == True:
                return x.T, memory
            else:
                return x.T
        else:
            return x.T[0]
    
    
    def backpropagation(self, x, y):
        deltas = {}
        nablas = {}
        for i in range(len(self.architecture)):
            nablas[str(i +1)] = {"weight": np.zeros((self.architecture[str(i+1)]["weight"]).shape) ,
                            "bias" : np.zeros((self.architecture[str(i+1)]["bias"]).shape)}
            deltas[str(i+1)] = None
        y_hat, memory = self.predict(x, mem=True, backprop=True)

        f_acti = self.activation_funcs[self.architecture[str(len(self.architecture))]["activation"]]
        Eo = self.cost(y_hat, y, deriv=True).T * f_acti(memory[str(len(self.architecture))]["z"], deriv=True)
        
        deltas[str(len(self.architecture))] = Eo 
        
        for l in range(len(self.architecture), 1, -1):
            acti_f = self.activation_funcs[self.architecture[str(l-1)]["activation"]]
            Eh = np.dot(deltas[str(l)], self.architecture[str(l)]["weight"].T) * acti_f(memory[str(l-1)]["z"], deriv=True)
            deltas[str(l-1)] = Eh

        for l in range(len(self.architecture)):
            nablas[str(l+1)]["weight"] = np.dot(deltas[str(l+1)].T, memory[str(l)]["activation"]).T
            nablas[str(l+1)]["bias"] = np.expand_dims(deltas[str(l+1)].mean(axis=0), 0)
            
        return nablas
            

    def update_weights(self, nablas, learning_rate):
        for i in range(len(self.architecture)):           
            self.architecture[str(i+1)]["weight"] -= learning_rate * nablas[str(i+1)]["weight"]
            self.architecture[str(i+1)]["bias"] -= learning_rate * nablas[str(i+1)]["bias"]


    def fit(self, x, y, l_rate=0.01, epochs=100, batch_size=0.3):
        for epoch in range(1, epochs+1):
            X, Y = self.shuffle(x, y)
            
            batches = len(x) // int(len(x)*batch_size) 
            
            batches_x = np.array_split(X, batches)
            batches_y = np.array_split(Y, batches)
            
            epoch_nablas = {}

            for i in range(len(self.architecture)):
                epoch_nablas[str(i+1)] = {"weight" : np.zeros(self.architecture[str(i+1)]["weight"].shape),
                                          "bias"   : np.zeros(self.architecture[str(i+1)]["bias"].shape) }
            
            for batch_x, batch_y in zip(batches_x, batches_y):
                x_trainb, x_testb, y_trainb, y_testb = self.dataset(batch_x, batch_y)
                batch_nablas = self.backpropagation(x_trainb, y_trainb)
                
                for i in batch_nablas:
                    epoch_nablas[i]["weight"] += batch_nablas[i]["weight"] / len(batch_x)
                    epoch_nablas[i]["bias"] += batch_nablas[i]["bias"] / len(batch_x)

                y_pred = self.predict(x_testb, backprop=True)
                loss = self.cost(y_pred, y_testb)

            self.update_weights(epoch_nablas, l_rate)
           
            if epoch%50 == 0:
                print(f"epoch {epoch}/{epochs}\t| loss: {loss:10.6f}") 
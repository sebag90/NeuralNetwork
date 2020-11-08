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


def print_progress_bar(iteration, total, prefix = "", suffix = "", decimals = 1, length = 100, fill = "#", printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '.' * (length - filledLength)
    print(f"\r{prefix} [{bar}] {percent}% {suffix}", end = printEnd)
    if iteration == total:
        print()
    
    
class Network():
    
    
    def __init__(self):
        self.architecture = {}
        self.history = { "loss": []}
        self.activation_funcs = {
            "relu" :    self.relu,
            "softmax" : self.softmax,
            "sigmoid" : self.sigmoid,
            "linear": self.linear,
            "tanh" : self.tanh        
            }

            
    # INITIALIZE, PRINT LOAD AND SAVE MODEL--------
            
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

    
    def summary(self):
        if len(self.architecture) == 0:
            print("The network was not initialized yet")
        else:
            total_parameters = 0
            print(f"Input Layer:\tNeurons:{self.architecture['1']['weight'].shape[0]}")
            for i in self.architecture:
                neurons = self.architecture[i]['weight'].shape[1]
                activation = self.architecture[i]['activation'].capitalize()
                parameters = self.architecture[i]["weight"].shape[0]*self.architecture[i]["weight"].shape[1]
                parameters += self.architecture[i]["bias"].shape[0]*self.architecture[i]["bias"].shape[1]
                total_parameters += parameters
                print(f"Layer {i}:\tNeurons:{neurons}, Activation: {activation}, Parameters: {parameters}")
            print(f"Total parameters: {total_parameters}")
        
     
    # ACTIVATION FUNCTIONS AND DERIVATIVES--
    
    def sigmoid(self, z, deriv=False):
        if deriv == True:
            return self.sigmoid(z) * (1 - self.sigmoid(z))
        else:
            return 1.0/(1.0 + np.exp(-z))
    
        
    def relu(self, z, deriv=False):
        if deriv == True:
            return (z > 0).astype(z.dtype)
        else:
            return np.maximum(0, z)

        
    def softmax(self, z, deriv=False):
        if deriv == True:
            return self.softmax(z) * (1 - self.softmax(z))
        else:
            m = np.max(z, axis=1, keepdims=True)
            e = np.exp(z - m)
            return e / np.sum(e, axis=1, keepdims=True)
    

    def linear(self, z, deriv=False):
        if deriv == False:
            return z
        else:
            return np.ones(z.shape)
    

    def tanh(self, z, deriv=False):
        if deriv == True:
            return 1 - np.power(self.tanh(z),2)
        else:
            return np.tanh(z)


    # ERROR-----------------------------------
    
    def cost(self, y_pred, y_true, deriv = False):
        if deriv == False:
            n = y_pred.shape[1]
            cost = 1/(n**2) * np.sum((y_true.reshape(y_pred.shape) - y_pred)**2)
            return cost
        else:
            cost_prime = y_pred - y_true.reshape(y_pred.shape)
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

    
    def predict(self, x, mem=False):
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

        if mem == True:
            return x, memory
        else:
            return x
      
    
    def backpropagation(self, x, y):
        deltas = {}
        nablas = {}
        for i in range(len(self.architecture)):
            nablas[str(i +1)] = {"weight": np.zeros((self.architecture[str(i+1)]["weight"]).shape) ,
                            "bias" : np.zeros((self.architecture[str(i+1)]["bias"]).shape)}
            deltas[str(i+1)] = None
        y_hat, memory = self.predict(x, mem=True)

        f_acti = self.activation_funcs[self.architecture[str(len(self.architecture))]["activation"]]
        Eo = self.cost(y_hat, y, deriv=True) * f_acti(memory[str(len(self.architecture))]["z"], deriv=True)
        
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


    def fit(self, x, y, l_rate=0.01, epochs=10, batch_size=32):
        for epoch in range(1, epochs+1):
            X, Y = self.shuffle(x, y)
            
            batches = len(x) // batch_size
            
            batches_x = np.array_split(X, batches)
            batches_y = np.array_split(Y, batches)

            it = 0
            print(f"Epoch {epoch}/{epochs}\t")
            
            for batch_x, batch_y in zip(batches_x, batches_y):
                batch_loss = []
                x_trainb, x_testb, y_trainb, y_testb = self.dataset(batch_x, batch_y)
                batch_nablas = self.backpropagation(x_trainb, y_trainb)
                y_pred = self.predict(x_testb)
                loss = self.cost(y_pred, y_testb)
                batch_loss.append(loss)
                self.update_weights(batch_nablas, l_rate)
                
                if it+1 == len(batches_x):
                    loss = np.mean(batch_loss)
                    self.history["loss"].append(loss)
                
                print_progress_bar(it + 1, len(batches_x), prefix=f"{it+1}/{len(batches_x)}", suffix=f"Loss: {loss:10.6f}", length=40)
                it += 1



if __name__ == "__main__":
    net = Network()
    net.init([3, 5, 7, 2], ["sigmoid", "sigmoid", "sigmoid"])
    net.summary()
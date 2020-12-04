import numpy as np 
import layers as layers
import activations as activation
import loss_functions as error
import extra_functions as extra


class Network:

    def __init__(self):
        self.layers = []
        self.history = {"loss": []}
        self.cost = None

        self.activation_funcs = {
            "relu" :    activation.relu(),
            "softmax" : activation.softmax(),
            "sigmoid" : activation.sigmoid(),
            "linear":   activation.identity(),
            "tanh" :    activation.tanh(),
            "swish" :   activation.swish(), 
            "lrelu" :   activation.lrelu()      
            }

        self.cost_funcs = {
            "squared loss" :    error.SquaredError(),
            "cross entropy" :   error.CrossEntropy()
        }

        self.layer_types = {
            "dense" : layers.Dense
        }


    def init(self, input_dimension, loss_function, layers):
        self.cost = self.cost_funcs[loss_function]
        layers = [{"units": input_dimension}] + layers
        
        if ((layers[-1]["activation"] == "softmax" and loss_function == "cross entropy") or
            (layers[-1]["activation"] != "softmax" and loss_function != "cross entropy")):

            for i in range(len(layers) - 1):
                layer_type = self.layer_types[layers[i+1]["type"]]
                activation = self.activation_funcs[layers[i+1]["activation"]]
                input_units = layers[i]["units"]
                output_units = layers[i+1]["units"]

                layer = layer_type(input_units, output_units, activation)
                self.layers.append(layer)

        else:
            print("The model could not be initialized")


    def summary(self):
        if len(self.layers) == 0:
            print("The model was not initialized")

        else:
            print(f"Input:\t\tUnits: {self.layers[0].weight.shape[0]}")
            
            for i, layer in enumerate(self.layers):
                if i+1 != len(self.layers):
                    print(f"Layer {i+1}:\t{layer}")
                else:
                    print(f"Output:\t\t{layer}")


    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x


    def backpropagation(self, x, y, learning_rate):
        prediction = self.predict(x)
        error = self.cost.prime(prediction, y)

        for i in range(len(self.layers) -1, -1 ,-1):       
            error = self.layers[i].backward(error, learning_rate)
        

    def fit(self, x, y, learning_rate=0.01, epochs=10, batch_size=32):
        x = x.astype("float32")
        y = y.astype("float32")
        for epoch in range(1, epochs + 1):
            X, Y = extra.shuffle(x, y)
            batches = len(x) // batch_size
            batches_x = np.array_split(X, batches)
            batches_y = np.array_split(Y, batches)

            it = 1
            print(f"Epoch {epoch}/{epochs}\t")
            epoch_loss = []

            for batch_x, batch_y in zip(batches_x, batches_y):
                x_trainb, x_testb, y_trainb, y_testb = extra.dataset(batch_x, batch_y)
                self.backpropagation(x_trainb, y_trainb, learning_rate)
                y_pred = self.predict(x_testb)
                batch_loss = self.cost(y_pred, y_testb)
                epoch_loss.append(batch_loss)
                loss = np.mean(epoch_loss)
                
                extra.print_progress_bar(it , len(batches_x), prefix=f"{it}/{len(batches_x)}", suffix=f"Loss: {loss:10.6f}", length=40)
                it += 1
            
            self.history["loss"].append(loss)
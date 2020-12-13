import numpy as np 
import nn.layers as layers
import nn.activations as activation
import nn.loss_functions as error
import nn.extra_functions as extra


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
        """
        function to initialize the network based on input parameters:
        
        net.init(input_dimension=784, loss_function="cross entropy", layers=[
            {"units": 128, "activation": "relu", "type":"dense"},
            {"units": 64, "activation": "sigmoid", "type":"dense"},
            {"units": 10, "activation": "softmax", "type":"dense"}
        ])

        input: 
            input_dimension: number of dimensions (columns) of the input matrix
        
        loss_function: 
            loss function for the neural network (cross entropy or squared error)
        
        layers:
            a list of dictionaries with following structure:
                {"units": 64, "activation": "sigmoid", "type":"dense"}
            
            units: number of units in the layer
            activation: activation function for the layer
            type: type of layer         
        """
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
        """
        print a summary of the Network 
        example:
            Input:      Units: 784
            Layer 1:    Units: 128,	Activation: ReLu,       Type: Dense
            Layer 2:    Units: 64,	Activation: ReLu,	Type: Dense
            Output:     Units: 10,	Activation: Softmax,	Type: Dense
        """
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
        """
        forward step over the entire network (layer for layer).
        input: input matrix
        output: predictions
        """
        for layer in self.layers:
            x = layer.forward(x)

        return x


    def backpropagation(self, x, y, learning_rate):
        """
        backpropagation (single learning step) layer for layer
        through the entire network.     

        input: input matrix x, target labels y and learning rate
        """
        prediction = self.predict(x)
        error = self.cost.prime(prediction, y)

        for i in range(len(self.layers) -1, -1 ,-1):       
            error = self.layers[i].backward(error, learning_rate)
        

    def fit(self, x, y, learning_rate=1e-3, epochs=10, batch_size=32):
        """
        function to train the neural network using mini batches.

        input: 
            x: matrix x of the train set 
            y: target labels for the input x
            learning_rate: learning_rate of the network
            epochs: number of epochs (how many times the network looks over the entire
                    training data set)
            batch_size: number of instances in every minibatch

        At the end of the training the network is ready to make predictions.
        The loss over every epoch is saved in the attribute self.history["loss"] (dictionary)
        """
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
                x_trainb, x_testb, y_trainb, y_testb = extra.split_dataset(batch_x, batch_y)
                self.backpropagation(x_trainb, y_trainb, learning_rate)
                y_pred = self.predict(x_testb)
                batch_loss = self.cost(y_pred, y_testb)
                epoch_loss.append(batch_loss)
                loss = np.mean(epoch_loss)
                
                extra.print_progress_bar(it , len(batches_x), prefix=f"{it}/{len(batches_x)}", suffix=f"Loss: {loss:10.6f}", length=40)
                it += 1
            
            self.history["loss"].append(loss)
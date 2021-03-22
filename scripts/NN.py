import numpy as np
import random
import sklearn.metrics as metrics

class NeuralNetwork():
    
    """
    Generic class for a neural network with an arbitrary number of layers and specified activation function
        
    Attributes:
        self.activation (function) : specified activation function to use
        self.dx_activation (function) : derivative of specified activation function
        self.losses (list<float>) : list of average loss at each epoch
        self.z_values (list<np.array>) : list to store pre-activation values at each layer to make backpropagation calculations easier
        self.bias (list<float>) : list of biases for each layer of the network
        self.weights (list<np.ndarray>) : list of weights between each layer of the network
        self.layers (list<np.array>) : list of vectors containing input/activation values at each layer
        self.alpha (float or int) : learning rate of the network 
        self.epsilon (float) : random float to prevent divide by 0 errors
        self.epochs (int) : number of times to run dataset through model
        self.threshold (float) : threshold for stopping when the loss is below this value

    Parameters:
        sizes (list<int>) : stored as self.layers
        activation (str) : "sigmoid" or "ReLU" to specify which activation function should be used, default="sigmoid"
        seed (int) : seed to set for numpy.random, use None for random seed, default=0
        alpha (float) : stored as self.alpha, default=0.1
        epsilon (float) : stored as self.epsilon, default=0.001
        epochs (int) : stored as self.epochs, default=250
    
    """
    
    def __init__(self, sizes, activation="sigmoid", seed=0, alpha=0.1, epsilon=0.001, epochs=250, threshold=0.01):
        """
        Constructor method for NeuralNetwork class
        """
        # set seed
        np.random.seed(seed)
        
        # initialize hyperparameters
        self.epsilon=epsilon
        self.alpha=alpha
        self.epochs=epochs
        self.threshold=threshold
        
        # intialize the activation function and its corresponding derivative
        if activation =="ReLU":
        	self.activation=NeuralNetwork._ReLU
	        self.dx_activation=NeuralNetwork._dxReLU
        else:
        	self.activation=NeuralNetwork._sigmoid
	        self.dx_activation=NeuralNetwork._dxSigmoid

        
        # initialize an empty list to store the average loss per epoch
        self.losses = []
        
        # initialize an empty list to store the activations
        self.z_values = list(range(len(sizes)))
        
        # initalize empty arrays to hold the values of the different layers
        # which will be added below as part of the feed-forward calculations
        self.layers = list(range(len(sizes)))
        
        # initalize an empty list to store the biases
        self.bias = np.random.rand(len(sizes))
        
        # Make weights for each layer as an array of random
        # small numbers. The dimensions of each weight is equal to
        # the (size of current layer) x (size of next layer)
        self.weights = []
        
        for ind in range(len(sizes) - 1):
            curr_weights = np.random.randn(sizes[ind+1], sizes[ind]) * np.sqrt(2.0/8)
            self.weights.append(curr_weights)
        
    @staticmethod
    def _sigmoid(x):
        """
        Sigmoid activation function
        """
        return 1.0 / (1.0 + np.exp(-x))
    
    @staticmethod
    def _dxSigmoid(x):
        """
        Derivative of the sigmoid activation function
        """
        return NeuralNetwork._sigmoid(x) * (1 - NeuralNetwork._sigmoid(x))
    
    @staticmethod
    def _ReLU(x):
        """
        ReLU activation function
        """
        return np.where(x<=0, 0, x)
    
    @staticmethod
    def _dxReLU(x):
        """
        Derivative of the ReLU function
        """
        return np.where(x<=0, 0, 1)
    
    def forward(self, values):
        """
        Propagates input values through the network, and sets the values of each layer to the activation of the previous layer 
        
        Parameters:
            values (np.ndarray) : array containing values in sample x feature format 
        
        """
        
        # store the transpose of the input layer as the first layer
        # also set the input layer as the first "z" value
        curr_activation = values.transpose()
        self.z_values[0] = curr_activation
        self.layers[0] = curr_activation
        
        # Propagate the input and weights through the network, updating the value
        # to the current layer as it is calculated. The value is also stored as
        # the values of the hidden & output layers. The output layer is the final layer
        for i in range(len(self.weights)):
        
            curr_z = np.dot(self.weights[i], curr_activation) + self.bias[i]
            curr_activation = self.activation(curr_z)
            
            self.z_values[i+1] = curr_z
            self.layers[i+1] = curr_activation
    
    def backpropagate(self, y):
        """
        Perform backpropagation to update the weights and bias
        
        Parameters:
            y (np.array) : array containing ground truth values for error calculations

        """
        
        # take the transpose of y, which are the actual values
        y = y.transpose()
        
        # get the output layer, which stores the predicted values
        # clip values to prevent log(0) errors when calculating loss
        pred = self.layers[-1]
        pred = np.where(pred==0, self.epsilon, pred)
        pred = np.where(pred==1, 1+self.epsilon, pred)
        
        # Calculate the average error for the current batch using the BCE loss function
        # When y = 0, bce_loss = -log(1-pred) and approaches 0 as pred approaches 0
        # When y = 1, bce_loss = -log(pred) and approaches 0 as pred approaches 1
        loss = (-1/y.shape[1]) * np.sum((y*np.log(pred) + (1-y)*np.log(1-pred)))
        self.losses.append(loss)

        # If the loss is less than a specified threshold, use this as a stop 
        # criterion that assumes that the model has been adequetely trained
        if loss < self.threshold:
        	return

        # Calculate the local gradient of the final layer, which is the derivative of the loss function
        # When y = 0, returns 1/(1-pred)
        # When y = 1, returns -1/pred
        dx_cost = ((-y/pred) + (1-y)/(1-pred)) / y.shape[1]
        
        # Calculate the delta change for the final layer
        prev_delta = dx_cost * self.dx_activation(self.z_values[-1]) 

        # Create a list to keep track of deltas and add the output
        # delta that was just calculated
        delta_list = list(range(len(self.weights)))
        delta_list[-1] = prev_delta

        # Calculate the gradients for each hidden layer as a local gradient * global gradient
        # The local gradient at each hidden layer is calculated using the derivative of the activation function
        # And the global gradient is recursively calculated as np.dot(W(i+1), prev_delta)
        for i in reversed(range(len(self.weights)-1)):
            prev_delta = np.dot(self.weights[i+1].transpose(), prev_delta) * self.dx_activation(self.z_values[i+1])
            delta_list[i] = prev_delta

        # Use the deltas calculated at each layer to update the weights
        for i in range(len(self.layers)-1):
            self.weights[i] -= np.dot(delta_list[i], self.layers[i].transpose()) * self.alpha
            self.bias[i] -= np.sum(delta_list[i] * self.alpha)
    
    def fit(self, test_x, test_y):
        """
        Trains the network using feedforward and backpropagation 
        
        Parameters:
            test_x (np.ndarray) : array containing values in sample x feature format 
            test_y (np.ndarray) : array of ground truth values for each sample in text_x
        """
        
        # create an empty list to store average loss values per epoch
        # in each itera
        losses = []
        
        # 
        for i in range(self.epochs):
            self.forward(test_x)
            self.backpropagate(test_y)
        
    def predict(self, test_values):
        """
        Fits a trained network to a new set of values
        
        Parameters:
            test_values (np.ndarray) : array containing values in sample x feature format 
        """
        output = test_values.transpose()
        
        for curr_weight, curr_bias in zip(self.weights, self.bias):
            output = self.activation(np.dot(curr_weight, output) + curr_bias)
        
        return output.transpose()

def encode(fasta):
    """
    One-hot encoding of nucleotide sequences 
    
    Parameters:
        fasta (str) : nucleotide sequence to be one-hot encoded
    
    Returns:
        np.array : one-hot and flattened vector encoding input fasta sequence
    
    """
    # some quality control on sequence
    fasta = fasta.upper() 

    # create empty list to contain encoding of each item in sequence
    vector = []
    
    # 1-hot encoding 
    for f in fasta:
        if f=="A": vector.append([1,0,0,0])
        elif f=="C": vector.append([0,1,0,0])
        elif f=="G": vector.append([0,0,1,0])
        elif f=="T": vector.append([0,0,0,1])
        
    return np.ndarray.flatten(np.array(vector))

# input fasta files of alignments
def read_fasta(input_file):
    """
    Parses fasta file to retrieve fasta sequence
    
    Parameter:
        input_file (str, path-like) : Path to fasta file to be read in. Fasta file can contain list of sequences with or without fasta headers

    """

    with open(input_file, 'r') as f:
        
        # read in fasta file
        file_text = f.read().strip()
        
        # if in fasta format, parse lines and return only sequences
        if ">" in file_text: 
            file_text = file_text.split(">")
            fastas = {}
            
            for fasta in file_text:
                if "\n" in fasta:
                    curr_fasta = fasta.split("\n")
                    fasta_header = curr_fasta[0]
                    curr_fasta = "".join(curr_fasta[1:])
                    
                    fastas[curr_fasta] = ">"+fasta_header #.append(curr_fasta)
                    
            return fastas
        
        # if there are no fasta headers, just return each line 
        else: 
            return [line for line in file_text.splitlines()]

#NeuralNetwork(sizes=layers, activation=activation, alpha=lr)

def kfold(k, x_val, y_val, seed=0, quiet=True, **nn_params):
    """
    Split the training set into k random folds by shuffling then splitting the dataset into k folds
    
    Parameters:
        x_val : 
        y_val : 
        epochs : default=250
        quiet : default=True

    Returns:
        float : average AUC calculated across the folds
    
    """
    
    # Set random seed to ensure random sampling is consistent
    random.seed(seed)

    # Shuffle the data
    shuffled = [(x, y) for x,y in zip(x_val, y_val)]
    random.shuffle(shuffled)

    # Generate indices that mark the end of each of the k folds
    k_folds = [0] + [int(len(shuffled)/k)*fold for fold in range(1,k+1)]

    # Create variable to store the average auc across folds 
    auc = 0

    # Perform k-fold cross-validation 
    for ind in range(k):

        # create new neural network
        nn = NeuralNetwork(**nn_params)

        # Hold out the current fold as a test set
        testing = shuffled[k_folds[ind]:k_folds[ind+1]]
        x_test_k = np.array([item[0] for item in testing])
        y_test_k = np.array([item[1] for item in testing])

        # Remove the hold out from the training set
        training = shuffled.copy()
        del training[k_folds[ind]:k_folds[ind+1]]
        x_train_k = np.array([item[0] for item in training])
        y_train_k = np.array([item[1] for item in training])


        # Train the model using subset of data
        nn.fit(x_train_k, y_train_k) 

        # Print the final average loss for the current fold
        if not quiet: print("Final average loss for fold %s = %s"%(ind+1, nn.losses[-1]))

        # Use the model to generate predictions for the hold out set
        pred = nn.predict(x_test_k)

        # Store the current auc for this fold
        auc += metrics.roc_auc_score(y_test_k.flatten(), pred.flatten())

    # Return the average AUC across folds
    return auc/k

def geneticAlgo(params, x_vals, y_vals, model=NeuralNetwork, k=7, max_iter=5):
    
    # keep track of the best model
    best_model = params[0]
    best_auc = 0
    
    # Run the genetic algorithm for a specified number of iterations
    for i in range(max_iter) :
        
        # create an empty list to store the child models
        children = []
        
        # Create as many children as parents
        while len(children) < len(params):

            # Randomly select 4 models
            p = random.choices(params, k=4)
            
            # Randomly mutate values with a mutation rate = 10%
            if random.random() < 0.1:
                p[0][2] = random.random() * random.randint(1,5) # mutate alpha
                p[3][3] = random.randint(25,150) # mutate epochs

            # Create children as a hybrid of the 4 selected parents with crossover = 50%
            child1 = {"sizes":p[0][0], "activation":p[1][1], "alpha":p[0][2], "epochs":p[1][3]}
            child2 = {"sizes":p[2][0], "activation":p[2][1], "alpha":p[3][2], "epochs":p[3][3]}

            # Hunger games by k-fold cross validation
            # May the best child win
            auc1 = kfold(k, x_vals, y_vals, **child1)
            auc2 = kfold(k, x_vals, y_vals, **child2)

            # Select the best child to add to the new population
            if auc1 > auc2:
                children.append([child1["sizes"], child1["activation"], child1["alpha"], child1["epochs"]])
            else:
                children.append([child2["sizes"], child2["activation"], child2["alpha"], child2["epochs"]])
                
            # Update the best model if necessary
            if max(auc1,auc2) > best_auc:
                best_model = children[-1]
                best_auc = max(auc1,auc2)
        
        # Replace the parents with the child population
        params = children
        
        # Add the best model to the children
        # This implementation allows the same child to be added from both 
        # the tournament selection and as the best model.
        params.append(best_model)
        print(best_model)
        
    return best_model
        
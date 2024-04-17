from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras import optimizers
import matplotlib.pyplot as pl

TANGERINE = '#fc3003'
TEAL = '#09ba97'

class MachineLearning:
    def __init__(self, features, labels, layers, batch_size, epochs, learn_rate):
        self.features = features
        self.labels = labels
        self.epochs = epochs 
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        self.layers = layers 

    # Thanks to keras.io for being a very helpful resource to make a simple ANN
    def create_model(self):
        self.model = Sequential()
        # Using ReLU as it is the most commonly used activation function
        self.model.add(Dense(int(self.layers[1]), input_dim = self.features.shape[1], activation = 'relu'))
        if len(self.layers) == 4:
            self.model.add(Dense(int(self.layers[2]), activation = 'relu'))
        elif len(self.layers) == 5:
            self.model.add(Dense(int(self.layers[2]), activation = 'relu'))
            self.model.add(Dense(int(self.layers[3]), activation = 'relu'))
        self.model.add(Dense(int(self.layers[-1]), activation = 'sigmoid')) #output

        # AdaGrad and AdaDelta turned out to be unoptimal for this case
        if self.labels.shape[1] == 2:
            self.model.compile(loss = 'binary_crossentropy', optimizer = optimizers.Adam(learning_rate = self.learn_rate), metrics = ['accuracy'])
        else: 
            self.model.compile(loss = 'categorical_crossentropy', optimizer = optimizers.Adam(learning_rate = self.learn_rate), metrics = ['accuracy'])
        
        self.accloss = self.model.fit(self.features, self.labels, epochs = self.epochs)


    def start_training(self):
        self.create_model()
        self.accuracy = self.model.evaluate(self.features, self.labels)

    def plot_graph(self):
        pl.plot(self.accloss.history['loss'], color = TANGERINE)
        pl.plot(self.accloss.history['accuracy'], color = TEAL)
        #bg = pl.axes()
        #bg.set_facecolor("#777777")
        pl.grid(linewidth = 0.2, alpha = 0.4)
        pl.legend(['Loss', 'Accuracy'])
        pl.title('Dependency of accuracy and loss on number of epochs')
        pl.xlabel('Number of Epochs')
        pl.show()

    def test_model(self, test_features):
        probabilities = self.model.predict(test_features)
        # Store categories as binary values
        self.predictions = []
        i = 0
        # One-hot encoding, represent classification as array of binary values
        for element in probabilities:
            max_prob = max(list(element))
            self.predictions.append([])
            # Shoots up to 1.0 as max value if probability is very high
            for j in range(0, len(element)):
                if prob[j] == max_prob:
                    self.predictions[i].append('1.0')
                else: 
                    self.predictions[i].append('0.0')
                i = i + 1
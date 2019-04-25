"""
Source code for the a recurrent neural network architecture with bidirectional
LSTM cells, which will act as sentiment analysis model. A bidirectional layer is one
that duplicates the original layer, but reverses the input - output pathway.
In a way, it connects the future and the past, to draw connections between close-by
sequential words. Relevant papers:
1) Bidirectional Recurrent Neural Networks Mike Schuster and Kuldip K. Paliwal, 11, Nov. 1997
"""
import tensorflow as tf

class RNN_LSTM(tf.keras.Model):
    """
    Class for the LSTM model to perform text sentiment classification. Decrease
    initial_features_size and num_bidirectional_cells variables for faster training.
    @param sequence_length: maximum threshold length of words
    @param vocabulary_size: num. unique words in mapping provided by tokenization
    @param initial_features_size: starting number of features for LSTM cells (default: 256)
    @param num_bidirectional_cells: depth of bidirectional LSTM cells (default: 4)
    @param num_classes: number of distinct labels (default: 2 -> binary classification)
    """
    def __init__(self, sequence_length, vocabulary_size,
                 initial_features_size=128, num_bidirectional_cells=2, num_classes=2):
        super(RNN_LSTM, self).__init__(name='rnn_lstm_model')
        # Get model & data specific attributes
        self.num_classes = num_classes
        self.seq_length = sequence_length
        self.vocab_size = vocabulary_size
        self.init_size = initial_features_size
        self.K = num_bidirectional_cells

        # Define the input layer and return a placeholder tensor
        # self.input_layer = tf.keras.Input(shape=(self.seq_length,))
        # Apply embedding
        self.embedded = tf.keras.layers.Embedding(self.vocab_size, self.init_size)
        # Apply recurrent bidirectional LSTM cells (only last cell one doesn't return sequences)
        get_bidirectional_cell = lambda x, step: tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(x, return_sequences=True if step!=self.K-1 else False))
        self.bidirectional_cells = [get_bidirectional_cell(
                x=int(self.init_size * (0.5 ** i)),
                step=i) for i in range(self.K)]
        # Flatten into starting number of features
        self.dense = tf.keras.layers.Dense(self.init_size, activation="relu")
        # Generate binary (sigmoid) or multi-label classification (softmax)
        # based on predicted tensor value
        self.predictions = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        """
        Function to define and apply the forward pass of the neural network,
        using layers previously defined in initialization of subclass.
        @param inputs: inputs passed to model, in the NumPy array format
        @return Y: outputs/predictions/logits that exit the model
        """
        X = self.embedded(inputs)
        for layer in self.bidirectional_cells:
            X = layer(X)
        X = self.dense(X)
        Y = self.predictions(X)
        return Y

    def compute_output_shape(self, input_shape):
        """
        You need to override this function if you want to use the subclassed model
        as part of a functional-style model. Otherwise, this method is optional.
        """
        # shape = tf.TensorShape(input_shape).as_list()
        # shape[-1] = self.num_classes
        # return tf.TensorShape(shape)
        pass

#### FUTURE WORK ####
# TO-DO: Will try to write vanilla TensorFlow version in the future
# TO-DO: Deploy the model for prediction
# Find layers & utilities to use from here
# print(dir(tf.layers))
# print(dir(tf.nn))
# print(dir(tf.nn.rnn_cell))

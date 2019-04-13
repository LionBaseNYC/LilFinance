import tensorflow as tf

class RNN_LSTM(object):
    """
    Class for LSTM model to be used with text classification.
    @arg v_size: Number of unique words in the vocabulary to integer
         mapping provided by tokenization
    """
    def __init__(self, v_size):
        self.vocab_size = v_size
        # Create model class for Keras
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, 64),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        # Compile model for training
        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
    def get_model(self):
        return self.model

    # TO-DO: Will try to write vanilla TensorFlow version in the future
    # TO-DO: Deploy the model for prediction
    # Find layers & utilities to use from here
    # print(dir(tf.layers))
    # print(dir(tf.nn))
    # print(dir(tf.nn.rnn_cell))

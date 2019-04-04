import tensorflow as tf

from tensorflow.keras.layers import Dense

print(tf.__version__)

class Autoencoder(tf.keras.Model):
    """"""
    def __init__(self):
        super(Autoencoder, self).__init__()


        self.n_hidden = 500
        self.n_latent = 10

        self.enc_1 = Dense(self.n_hidden, activation='relu')
        self.enc_2 = Dense(self.n_hidden, activation='relu')
        self.latent = Dense(self.n_latent, activation='linear')
        self.dec_1 = Dense(self.n_hidden, activation='relu')
        self.dec_2 = Dense(self.n_hidden, activation='relu')
        self.reconstruction = Dense(784, activation='linear')

    def call(self, inputs):
        x = self.enc_1(inputs)
        x = self.enc_2(x)
        x = self.latent(x)
        x = self.dec_1(x)
        x = self.dec_2(x)

        return self.reconstruction(x)


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# flatten the dataset
x_train = x_train.reshape((-1, 28 * 28))
x_test = x_test.reshape((-1, 28 * 28))

Ae = Autoencoder()
Ae.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
Ae.fit(x=x_train, y=x_train, batch_size=30,epochs=10)

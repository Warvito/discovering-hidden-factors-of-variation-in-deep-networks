"""Discovering hidden factors of variation in deep networks

Based on:
https://github.com/Lasagne/Lasagne/blob/highway_example/examples/Hidden%20factors.ipynb
"""
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

print(tf.__version__)

# Set random seed
tf.random.set_seed(1)

# Loading data
print("Loading data...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# flatten the dataset
x_train = x_train.reshape((-1, 28 * 28))
x_test = x_test.reshape((-1, 28 * 28))

batch_size = 256
# create the database iterator
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024)
train_dataset = train_dataset.batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(batch_size)

# Building model
n_class = 10
input_dim = 784

# Encoder
inputs = tf.keras.Input(shape=(input_dim,))
x = tf.keras.layers.Dense(500, activation='relu')(inputs)
x = tf.keras.layers.Dense(500, activation='relu')(x)
observed = tf.keras.layers.Dense(n_class, activation='softmax')(x)
latent = tf.keras.layers.Dense(2, activation='linear')(x)

encoder = tf.keras.Model(inputs=inputs, outputs=[latent, observed])

# Decoder
inputted_latent = tf.keras.Input(shape=(2,))
inputted_observed = tf.keras.Input(shape=(n_class,))

x = tf.keras.layers.concatenate([inputted_latent, inputted_observed], axis=-1)
x = tf.keras.layers.Dense(500, activation='relu')(x)
x = tf.keras.layers.Dense(500, activation='relu')(x)
reconstruction = tf.keras.layers.Dense(input_dim, activation='linear')(x)
decoder = tf.keras.Model(inputs=[inputted_latent, inputted_observed], outputs=reconstruction)

# Losses
mse_loss_fn = tf.keras.losses.MeanSquaredError()


def xcov_loss_fn(latent, observed, batch_size):
    latent_centered = latent - tf.reduce_mean(latent, axis=0, keepdims=True)
    observed_centered = observed - tf.reduce_mean(observed, axis=0, keepdims=True)
    xcov_loss = 0.5 * tf.reduce_sum(
        tf.square(tf.matmul(latent_centered, observed_centered, transpose_a=True) / batch_size))

    return xcov_loss


cat_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

alpha = 1.0
beta = 10.0
gamma = 10.0

# Optimizer
optimizer = tf.keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)


# optimizer = tf.keras.optimizers.Adam()

# Training
@tf.function  # Make it fast.
def train_on_batch(batch_x, batch_y):
    # Open a GradientTape.
    with tf.GradientTape() as tape:
        batch_latent, batch_observed = encoder(batch_x)
        batch_reconstruction = decoder([batch_latent, batch_observed])

        recon_loss = alpha * mse_loss_fn(batch_x, batch_reconstruction)
        cat_loss = beta * cat_loss_fn(tf.one_hot(batch_y, n_class), batch_observed)
        xcov_loss = gamma * xcov_loss_fn(batch_latent, batch_observed, tf.cast(tf.shape(batch_x)[0], tf.float32))

        ae_loss = recon_loss + cat_loss + xcov_loss

    gradients = tape.gradient(ae_loss, encoder.trainable_variables + decoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_variables + decoder.trainable_variables))
    return recon_loss, cat_loss, xcov_loss


n_epochs = 200
for epoch in range(n_epochs):
    start = time.time()

    epoch_recon_loss_avg = tf.metrics.Mean()
    epoch_cat_loss_avg = tf.metrics.Mean()
    epoch_xcov_loss_avg = tf.metrics.Mean()

    for batch, (batch_x, batch_y) in enumerate(train_dataset):
        (r_loss, c_loss, x_loss) = train_on_batch(batch_x, batch_y)
        epoch_recon_loss_avg(r_loss)
        epoch_cat_loss_avg(c_loss)
        epoch_xcov_loss_avg(x_loss)

    epoch_time = time.time() - start
    print('EPOCH: {}, TIME: {}, ETA: {},  R_LOSS: {},  C_LOSS: {},  X_LOSS: {}'.format(epoch + 1, epoch_time,
                                                                                       epoch_time * (n_epochs - epoch),
                                                                                       epoch_recon_loss_avg.result(),
                                                                                       epoch_cat_loss_avg.result(),
                                                                                       epoch_xcov_loss_avg.result()))

z_test_list = []
for batch, (batch_x, batch_y) in enumerate(test_dataset):
    z_test_list.extend(encoder(batch_x)[0].numpy())
z_test = np.asarray(z_test_list)

plt.figure()
plt.scatter(z_test[:, 0], z_test[:, 1], alpha=0.1)
plt.show()

ys = np.repeat(np.arange(10), 9).astype('int32')
zs = np.tile(np.linspace(-0.5, 0.5, 9), 10).astype('float32')
z1s = np.vstack([zs, np.zeros_like(zs)]).T
z2s = np.vstack([np.zeros_like(zs), zs]).T

reconstructions_z1 = decoder([z1s, tf.one_hot(ys, n_class)]).numpy()
reconstructions_z2 = decoder([z2s, tf.one_hot(ys, n_class)]).numpy()

im1 = reconstructions_z1.reshape(10, 9, 28, 28).transpose(1, 2, 0, 3).reshape(9 * 28, 10 * 28)
plt.imshow(im1, cmap=plt.cm.gray)
plt.show()

im2 = reconstructions_z2.reshape(10, 9, 28, 28).transpose(1, 2, 0, 3).reshape(9 * 28, 10 * 28)
plt.imshow(im2, cmap=plt.cm.gray)
plt.show()

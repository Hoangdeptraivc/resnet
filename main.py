
import tensorflow as tf
import numpy as np

class Residual(tf.keras.layers.Layer):
    def __init__(self, num_channels, use_1x1conv=False, stride=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(num_channels, kernel_size=3, strides=stride, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(num_channels, kernel_size=3, strides=1, padding='same')
        if use_1x1conv:
            self.conv3 = tf.keras.layers.Conv2D(num_channels, kernel_size=1, strides=stride)
        else:
            self.conv3 = None
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, X):
        Y = self.bn1(self.conv1(X))
        Y = tf.nn.relu(Y)
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return tf.nn.relu(Y + X)
net = tf.keras.Sequential()
net.add(tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same', input_shape=(28, 28, 3)))
net.add(tf.keras.layers.BatchNormalization())
net.add(tf.keras.layers.ReLU())
net.add(tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same'))

def resnet_block(num_channels, num_residuals, first_block=False):
    blk = tf.keras.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(Residual(num_channels, use_1x1conv=True, stride=2))
        else:
            blk.add(Residual(num_channels))
    return blk

net.add(resnet_block(64, 2, first_block=True))
net.add(resnet_block(128, 2))
net.add(resnet_block(256, 2))
net.add(resnet_block(512, 2))
net.add(tf.keras.layers.GlobalAveragePooling2D())
net.add(tf.keras.layers.Dense(10))
lr, num_epochs, batch_size = 1.0, 10, 256
net.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)
x_train = np.repeat(x_train, 3, axis=-1)
x_test = np.repeat(x_test, 3, axis=-1)
net.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(x_test, y_test))
test_loss, test_acc = net.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')

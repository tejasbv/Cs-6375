import tensorflow as tf
import numpy as np

# set the random seeds to make sure your results are reproducible
from numpy.random import seed
seed(1)
tf.random.set_seed(1)

# specify path to training data and testing data

train_x_location = "x_train_10.csv"
train_y_location = "y_train_10.csv"
test_x_location = "x_test.csv"
test_y_location = "y_test.csv"

print("Reading training data")
x_train = np.loadtxt(train_x_location, dtype="float",
                     delimiter=",").astype(np.float64)
y_train = np.loadtxt(train_y_location, dtype="uint8", delimiter=",")

m, n = x_train.shape  # m training examples, each with n features
m_labels,  = y_train.shape  # m2 examples, each with k labels
l_min = y_train.min()

assert m_labels == m, "x_train and y_train should have same length."
assert l_min == 0, "each label should be in the range 0 - k-1."
k = y_train.max()+1

print(m, "examples,", n, "features,", k, "categiries.")


# print("Pre processing x of training data")
# x_train = x_train / 1.0

# define the training model
model = tf.keras.models.Sequential([

    tf.keras.layers.Dense(1024, activation=tf.nn.leaky_relu,
                          input_shape=(n,)),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation=tf.nn.leaky_relu),
    # batch normalization after the LSTM layer
    tf.keras.layers.BatchNormalization(),


    # another layer with l2 regularization'''
    tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu,
                          kernel_regularizer=tf.keras.regularizers.l2(0.01)),

    # dropouts layer.
    # 0.2 is "rate", the probability that a weight is dropped.
    # dropped is implemented by setting weight to 0 during training
    # during testing the weights are multiplied by 1-0.2=0.8
    tf.keras.layers.Dropout(0.2),
    # last layer is softmax
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])


# loss='categorical_entropy' expects input to be one-hot encoded
# loss='sparse_categorical_entropy' expects input to be the category as a number
# options for optimizer: 'sgd' and 'adam'. sgd is stochastic gradient descent
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("train")
model.fit(x_train, y_train, epochs=2000, batch_size=32)
# default batch size is 32


print("Reading testing data")
x_test = np.loadtxt(test_x_location, dtype="float",
                    delimiter=",").astype(np.float64)
y_test = np.loadtxt(test_y_location, dtype="uint8", delimiter=",")

m_test, n_test = x_test.shape
m_test_labels,  = y_test.shape
l_min = y_train.min()

assert m_test_labels == m_test, "x_test and y_test should have same length."
assert n_test == n, "train and x_test should have same number of features."

print(m_test, "test examples.")

# print("Pre processing testing data")
# x_test = x_test / 1.0


print("evaluate")
model.evaluate(x_test, y_test)

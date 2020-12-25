#%%
# Warning
import warnings
# Ignore warnings
warnings.filterwarnings('ignore')

#%%
# Matplotlib
import matplotlib.pyplot as plt

# Set matplotlib sizes
plt.rc('font', size=20)
plt.rc('axes', titlesize=20)
plt.rc('axes', labelsize=20)
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
plt.rc('legend', fontsize=20)
plt.rc('figure', titlesize=20)

#%%p
# TensorFlow
# The magic below allows us to use tensorflow version 2.x
import tensorflow as tf
from tensorflow import keras

#%%
# Data Preprocessing
# Create data directory
import os

abspath = "/home/ubuntu/DogBreed"

# Make directory
directory = os.path.dirname(abspath + 'data/')
if not os.path.exists(directory):
    os.makedirs(directory)

#%%
import tensorflow_datasets as tfds

# Get the name of the data
data_name = 'stanford_dogs'

# Load data
data, info = tfds.load(name=data_name, data_dir=abspath + 'data/', as_supervised=True, with_info=True)

# Get the name of the target
target = 'label'

#%%
# Get the classes
classes = info.features['label'].names

# Print the classes
print(classes)

# Get the number of classes
n_classes = info.features['label'].num_classes

# Print the number of classes
print(info.features['label'].num_classes)

# Get the training, validation and testing data

#%%
# Set the training, validation and testing split
split_train, split_valid, split_test = 'train[:70%]', 'train[70%:]', 'test'

# Get the training data
data_train = tfds.load(name=data_name, split=split_train, data_dir=abspath + 'data/', as_supervised=True)

# Get the validation data
data_valid = tfds.load(name=data_name, split=split_valid, data_dir=abspath + 'data/', as_supervised=True)

# Get the testing data
data_test = tfds.load(name=data_name, split=split_test, data_dir=abspath + 'data/', as_supervised=True)

#%%
# Resize the data for the pretrained model
# Set the default input size for the pretrained model
input_size = [299, 299]

#%%
def resize(data, label):
    """
    Resize the data into the default input size for the pretrained model
    Parameters
    ----------
    data: the data
    label: the label
    
    Returns
    ----------
    The resized data
    """

    # Resize the data into the default input size for the pretrained model
    data_resized = tf.image.resize(data, input_size)

    return data_resized, label

#%%
# Resize the training data
data_train = data_train.map(resize)

# Resize the validation data
data_valid = data_valid.map(resize)

# Resize the testing data
data_test = data_test.map(resize)

#%%
# Preprocess the data using the pretrained model

# Set the preprocess_input of the pretrained model
preprocess_input = keras.applications.xception.preprocess_input

#%%
def preprocess(data, label):
    """
    Preprocess the data using the pretrained model
    Parameters
    ----------
    data: the data
    label: the label
    
    Returns
    ----------
    The preprocessed data
    """

    # Preprocess the data using the pretrained model
    data_preprocessed = preprocess_input(data)

    return data_preprocessed, label

#%%
# Preprocess the training data
data_train = data_train.map(preprocess)

# Preprocess the validation data
data_valid = data_valid.map(preprocess)

# Preprocess the testing data
data_test = data_test.map(preprocess)

#%%
# Shuffle, batch, and prefetch

# Shuffling the training data
data_train = data_train.shuffle(buffer_size=1000, seed=42)

# Set the batch size
batch_size = 16

# Batch and prefetch the training data
data_train = data_train.batch(batch_size).prefetch(1)

# Batch and prefetch the validation data
data_valid = data_valid.batch(batch_size).prefetch(1)

# Batch and prefetch the testing data
data_test = data_test.batch(batch_size).prefetch(1)

#%%
# Training
# Create the directory for the model

# Make directory
directory = os.path.dirname(abspath + 'model/')
if not os.path.exists(directory):
    os.makedirs(directory)

#%%
# Build the architecture of the model

# Add the pretrained layers
pretrained_model = keras.applications.xception.Xception(include_top=False, weights='imagenet')

# Add GlobalAveragePooling2D layer
average_pooling = keras.layers.GlobalAveragePooling2D()(pretrained_model.output)

# Add the output layer
output = keras.layers.Dense(n_classes, activation='softmax')(average_pooling)

# Get the model
model = keras.Model(inputs=pretrained_model.input, outputs=output)

#%%
# Freeze each of the pretrained layers

for layer in pretrained_model.layers:
    # Freeze the layer
    layer.trainable = False

#%%
# Set callbacks

model.save("model.h5")


# Checkpoint callback
checkpoint_cb = keras.callbacks.ModelCheckpoint(abspath + '/model/model.h5', save_best_only=True)

# Early stopping callback
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)

#%%
# Compile the model
# Use the default learning rate of Adam optimizer

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#%%
# Train the model

history = model.fit(data_train, epochs=5, validation_data=data_valid, callbacks=[checkpoint_cb, early_stopping_cb])

#%%
# Create the directory for the plot

# Make directory
directory = os.path.dirname(abspath + 'figure/')
if not os.path.exists(directory):
    os.makedirs(directory)

#%%
import pandas as pd

# Create a figure
pd.DataFrame(history.history).plot(figsize=(8, 5))

# Set grid
plt.grid(True)

# Save and show the figure
plt.tight_layout()
plt.savefig(abspath + 'figure/learning_curve_before_unfreezing.pdf')
plt.show()

#%%
# Unfreeze the pretrained layers

# For each layer in the pretrained model
for layer in pretrained_model.layers:
    # Unfreeze the layer
    layer.trainable = True

# %%
# Compile the model
# Use a lower learning rate (factor of 10) of Adam optimizer so that it's less likely to compromise the pretrained weights

from tensorflow.keras.models import load_model
model = load_model('model.h5')

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#%%
# Train the model
history = model.fit(data_train, epochs=5, validation_data=data_valid, callbacks=[checkpoint_cb, early_stopping_cb])

#%%
# Plot the learning curve
# Create a figure
pd.DataFrame(history.history).plot(figsize=(8, 5))

# Set grid
plt.grid(True)

# Save and show the figure
plt.tight_layout()
plt.savefig(abspath + 'figure/learning_curve_after_unfreezing.pdf')
plt.show()

#
# # save the model to disk
# model.save("model.h5")

#%%
# Testing
# Loading the saved model
model = keras.models.load_model(abspath + '/model.h5')

#%%
# Evaluating the model
loss, accuracy = model.evaluate(data_test)
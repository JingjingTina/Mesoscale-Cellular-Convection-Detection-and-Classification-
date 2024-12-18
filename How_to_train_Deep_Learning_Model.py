
# code name: Train_UNet_NoGamma_GridSearch.py
# by Jingjing Tian
# Train at cumulus with GPU

#SBATCH --partition=gpu               # Use gpu
#SBATCH --cpus-per-task=16            # Request 16 CPU cores
#SBATCH --mem=512G                    # Request 512 GB of memory


import warnings
warnings.simplefilter('ignore')

import numpy as np  
import xarray as xr 
import pandas as pd  # type: ignore
import pickle


import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"  


import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, concatenate, LeakyReLU, Normalization
from keras.optimizers import Adam

import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.layers import  BatchNormalization, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Conv2DTranspose, concatenate, Dropout, Add


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle


# Set logging level to reduce output clutter
tf.get_logger().setLevel('ERROR')

# Define the file path to the .npz file and Load the data from the .npz file
input_file_path = './training3_and_validation_data.npz' 
data = np.load(input_file_path)

# Access the arrays stored in the .npz file
X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
X_test = data['X_test']
y_test = data['y_test']

num_classes = 4  # Number of classes


def preprocess_image(image):
    image = tf.expand_dims(image, -1)  # Adds a third dimension for channels
    # Resize image
    image = tf.image.resize(image, [128, 128], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image


with tf.device("CPU"):
    
    X_train_F = tf.convert_to_tensor(X_train)
    y_train_F = tf.convert_to_tensor(y_train)
    
    print(X_train_F.shape,y_train_F.shape)
    
    X_train_FF=preprocess_image(X_train_F)
    print(X_train_FF.shape)

    y_train_FF=preprocess_image(y_train_F)
    print(y_train_FF.shape)

    y_train_one_hot_F = tf.keras.utils.to_categorical(y_train_FF, num_classes=4)  # Convert to one-hot encoding
    print(y_train_one_hot_F.shape)



with tf.device("CPU"):
    X_val_F = tf.convert_to_tensor(X_val)
    y_val_F = tf.convert_to_tensor(y_val)

    X_val_FF=preprocess_image(X_val_F)
    y_val_FF=preprocess_image(y_val_F)
    print(y_val_FF.shape)

    y_val_one_hot_F = tf.keras.utils.to_categorical(y_val_FF, num_classes=4)  # Convert to one-hot encoding
    print(X_val_FF.shape,y_val_one_hot_F.shape)


####
with tf.device("CPU"):
    X_test_F = tf.convert_to_tensor(X_test)
    y_test_F = tf.convert_to_tensor(y_test)

    X_test_FF=preprocess_image(X_test_F)
    y_test_FF=preprocess_image(y_test_F)
    print(y_test_FF.shape)

    y_test_one_hot_F = tf.keras.utils.to_categorical(y_test_FF, num_classes=4)  # Convert to one-hot encoding
    print(X_test_FF.shape,y_test_one_hot_F.shape)

#####
def conv_block(input_tensor, num_filters, use_residual=False):
    """Construct a convolutional block with optional residual connection."""
    x = Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    if use_residual:
        # Apply a residual connection using an identity mapping
        identity = Conv2D(num_filters, (1, 1), padding='same')(input_tensor)
        x = Add()([x, identity])

    return x

def unet_model(input_size=(128, 128, 1), num_classes=4):
    inputs = Input(input_size)

    # Encoder pathway
    c1 = conv_block(inputs, 64, use_residual=False)
    p1 = MaxPooling2D((2, 2))(c1)
    #p1 = Dropout(0.1)(p1)  # Dropout after pooling layer
    c2 = conv_block(p1, 128, use_residual=False)
    p2 = MaxPooling2D((2, 2))(c2)
    #p2 = Dropout(0.1)(p2)  # Dropout after pooling layer
    c3 = conv_block(p2, 256, use_residual=True)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(0.2)(p3)  # Dropout with a higher rate for deeper layers
    c4 = conv_block(p3, 512, use_residual=True)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(0.2)(p4)  # Dropout after pooling layer


    # Bottleneck
    bn = conv_block(p4, 1024, use_residual=True)

    # Decoder pathway
    u1 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bn)
    u1 = concatenate([u1, c4])
    c5 = conv_block(u1, 512, use_residual=True)
    
    u2 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
    u2 = concatenate([u2, c3])
    c6 = conv_block(u2, 256, use_residual=True)
    
    u3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
    u3 = concatenate([u3, c2])
    c7 = conv_block(u3, 128, use_residual=False)
    
    u4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
    u4 = concatenate([u4, c1])
    c8 = conv_block(u4, 64, use_residual=False)

    # Output Layer
    output = Conv2D(num_classes, (1, 1), activation='softmax')(c8)

    model = Model(inputs=[inputs], outputs=[output])
    return model


# Train the model
keras.backend.clear_session()  


def weighted_categorical_crossentropy(weights):
    """A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    """
    weights = tf.constant(weights, dtype=tf.float32)
    
    def loss(y_true, y_pred):
        # scale predictions so that the class probabilities of each pixel sum to 1
        y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        # calculate the weighted loss
        loss = y_true * tf.math.log(y_pred) * weights
        loss = -tf.reduce_sum(loss, -1)
        return loss
    
    return loss


class_weights = [3.42532998, 0.32800352 ,5.42562766 ,2.10526705]  # pre-calculated


# Lists for different batch sizes and learning rates to test
batch_sizes = [8,16,32]  # [16,32,64]
learning_rates = [0.001]  # [0.0001,0.001, 0.01]

# Directory to save history files
if not os.path.exists('history_files'):
    os.makedirs('history_files')

# Convert training and validation data to tf.data.Dataset
def create_dataset(X, y, batch_size, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

# Loop through batch sizes and learning rates
for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        print(f"Testing batch_size: {batch_size}, learning_rate: {learning_rate}")

        # Create a new instance of the model for each run
        model = unet_model()

        # Compile the model with the current learning rate
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer,
                      loss=weighted_categorical_crossentropy(class_weights),
                      metrics=[
                          keras.metrics.MeanIoU(num_classes=4),
                          keras.metrics.CategoricalAccuracy(),
                      ])

        # Create tf.data.Dataset objects for training and validation
        train_dataset = create_dataset(X_train_FF, y_train_one_hot_F, batch_size)
        val_dataset = create_dataset(X_val_FF, y_val_one_hot_F, batch_size, shuffle=False)

        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=7,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                filepath=f'v1_best_model_bs{batch_size}_dropout_GridSearch.keras',
                monitor='val_loss',
                save_best_only=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',  # Monitor validation loss
                factor=0.5,  # Reduce learning rate by half if no improvement
                patience=2,  # Wait for 2 epochs with no improvement before reducing learning rate
                min_lr=1e-6  # Minimum learning rate
            )
        ]

        # Train the model with the current batch size and learning rate
        history = model.fit(
            train_dataset,  # Use the dataset pipeline
            epochs=100,
            validation_data=val_dataset,  # Validation dataset
            callbacks=callbacks,
            verbose=1
        )

        # Save the history to a .pkl file
        history_filename = f'v1_history_bs{batch_size}_dropout_GridSearch.pkl'
        with open(os.path.join('history_files', history_filename), 'wb') as f:
            pickle.dump(history.history, f)

        print(f"History saved to {history_filename}")

        # Clear the session after each run to free up GPU memory
        tf.keras.backend.clear_session()
















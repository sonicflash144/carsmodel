import tensorflow as tf
import pandas as pd
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
import os
import platform
from subprocess import check_output
from PIL import Image           # The Python Image Library (PIL)
import collections
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers, losses
from tensorflow.keras.optimizers import RMSprop

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
classes = ('airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
numClasses = len(classes)

img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols, 3)
def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':   
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        print(len(X[0]))
        Y = datadict['labels']
        X = X.reshape(10000,32, 32, 3)
        #X = np.array(X)
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):    
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte
#def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
def get_CIFAR10_data(num_training=10000, num_validation=100, num_test=200):
    # Load the raw CIFAR-10 data
    cifar10_dir = './input/cifar-10-batches-py/'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]   
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    x_train = X_train.astype('float32')
    x_test = X_test.astype('float32')

    x_train /= 255
    x_test /= 255

    return x_train, y_train, X_val, y_val, x_test, y_test

# Invoke the above function to get our data.
x_train, y_train, x_val, y_val, x_test, y_test = get_CIFAR10_data()

print('Train data shape: ', x_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', x_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', x_test.shape)
print('Test labels shape: ', y_test.shape)

# The images are index 0 of the dictionary
# They are stored as a 3072 element vector so we need to reshape this into a tensor.
# The first dimension is the red/green/blue channel, the second is the pixel row, the third is the pixel column
im = x_train[0].reshape(3,32,32)


# PIL and matplotlib want the red/green/blue channels last in the matrix. So we just need to rearrange 
# the tensor to put that dimension last.
im = np.transpose(im, axes=[1, 2, 0])  # Put the 0-th dimension at the end

# Image are supposed to be unsigned 8-bit integers. If we keep the raw images, then
# this line is not needed. However, if we normalize or whiten the image, then the values become
# floats. So we need to convert them back to uint8s.
im = (im * 255).astype(np.uint8)
im = np.uint8(im)  

im=Image.fromarray(im)

plt.imshow(im);
plt.title("test");
plt.axis('off');


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') is not None and logs.get('accuracy') > 0.99:
            print("\nReached 99.9% accuracy so cancelling training!")
            self.model.stop_training = True
            


# GRADED FUNCTION: train_happy_sad_model
def train_cars_model():
    
    train_datagen = ImageDataGenerator(rescale=1/255)
    
    # Instantiate the callback
    callbacks = myCallback()

    ### START CODE HERE

    # Define the model
    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 150x150 with 3 bytes color
        # This is the first convolution
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # The third convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
   
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class and 1 for the other
        tf.keras.layers.Dense(numClasses, activation='softmax')
    ])

    # Compile the model
    # Select a loss function compatible with the last layer of your network
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(learning_rate=0.001),
                  metrics=['accuracy']) 
    
    #print(model)


    # Specify the method to load images from a directory and pass in the appropriate arguments:
    # - directory: should be a relative path to the directory containing the data
    # - targe_size: set this equal to the resolution of each image (excluding the color dimension)
    # - batch_size: number of images the generator yields when asked for a next batch. Set this to 10.
    # - class_mode: How the labels are represented. Should be one of "binary", "categorical" or "sparse".
    #               Pick the one that better suits here given that the labels are going to be 1D binary labels.
    #x_train1=x_train.reshape(-1,32,32,3)
    #x_test1=x_test.reshape(-1,32,32,3)
    y_train2 = keras.utils.to_categorical(y_train,numClasses)
    y_test2 = keras.utils.to_categorical(y_test,numClasses)
    
    #print(x_train.shape)
    #print(y_train2.shape)
      
    #print(x_test.shape)
    #print(y_test2.shape)
    
    train_datagen.fit(x_train)
    train_datagen.fit(x_test)
    # fits the model on batches with real-time data augmentation:
    history= model.fit(
                x_train, y_train2,  # prepared data
                batch_size=6,
                epochs=15,
                validation_data=(x_test, y_test2),
                shuffle=True,
                verbose=5,
                callbacks=[callbacks],
                initial_epoch= 0
            )
    
    #model.fit(train_datagen.flow(x_train, y_train2, batch_size=5),
         #validation_data=train_datagen.flow(x_test, y_test2,
         #batch_size=5, subset='validation'),
         #steps_per_epoch=len(x_train) / 32, epochs=15, callbacks=[callbacks])
    

    #plt.plot(history.history['accuracy'], label='accuracy')
    #plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    #plt.xlabel('Epoch')
    #plt.ylabel('Accuracy')
    #plt.ylim([0.5, 1])
    #plt.legend(loc='lower right')
    #test_loss, test_acc = model.evaluate((x_test, y_test2),  classes, verbose=2)

    return history


hist = train_cars_model()

print(f"Your model reached the desired accuracy after {len(hist.epoch)} epochs")

## save model
hist.model.save("saved_model.h5");

'''
## load tensorflow model
new_model = tf.keras.models.load_model("saved_model.h5")
print (new_model.history['accuracy'])
'''
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
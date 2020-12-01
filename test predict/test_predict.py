from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
import tensorflow as tf

try:
    print("I am preparing loading test npy file...")
    x = np.load("test_x.npy")
    y = np.load("test_y.npy")
    print("I am loading npy file...")
except:
    print("I am making test npy file...")
    import test_dataset as ds
    x,y = ds.load_data()

x=x.reshape(-1,128,128,3).astype('float32')
x_normalize=x/255
y_one_hot = to_categorical(y,3)
    
def build_model():
    model = Sequential()
    model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(128, 128, 3), 
                 activation='relu', padding='same'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), 
                 activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), 
                 activation='relu', padding='same'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), 
                 activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), 
                 activation='relu', padding='same'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), 
                 activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten()) 

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    
    return model

model = build_model()    
try:
    model.load_weights("model_aug.h5")
    print("success")
except:
    print("error")

print()
scores=model.evaluate(x_normalize,y_one_hot,verbose=0)
print("test scores:",scores[1])
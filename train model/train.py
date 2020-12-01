# -*- coding: utf-8 -*-
import numpy as np
import os
import tensorflow as tf

try:
    print("I am preparing loading npy file...")
    x = np.load("train_x_aug.npy")
    y = np.load("train_y_aug.npy")
    print("I am loading npy file...")
except:
    print("I am making npy file...")
    import dataset as ds
    x,y = ds.load_data_augmentation()
    
x /= 255
#y=y.reshape(-1,1)
print(x.shape,y.shape)
print(y)
# =============================================================================
# from sklearn.utils import class_weight
# class_weights = class_weight.compute_class_weight('balanced',np.unique(y),y)
# print(class_weights)
# =============================================================================

from tensorflow.keras.utils import to_categorical
# y=np.array(y)
y_one_hot = to_categorical(y,3)

from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D

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
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

#history = model.fit(x, y, class_weight=class_weights, shuffle=True, batch_size=64, epochs=30, validation_split=0.2, verbose=1)
history = model.fit(x, y_one_hot, shuffle=True, batch_size=128, epochs=30, validation_split=0.2, verbose=1)

# =============================================================================
# logdir = os.path.join("logs")
# if not os.path.isdir(logdir):
#     os.mkdir(logdir)
# tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
# 
# history = model.fit(x, y, class_weight=class_weights, shuffle=True, batch_size=64, epochs=200, validation_split=0.2, verbose=1)
# #,callbacks=[tensorboard_callback]
# =============================================================================
import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation,image_name):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.xticks([row for row in range(0, len(train_history.history[train]))])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.savefig(image_name)
    plt.show()
show_train_history(history,'acc','val_acc','accuracy.png')
show_train_history(history,'loss','val_loss','loss.png')
# Save model
try:
    model.save_weights("model_aug.h5")
    print("success")
except:
    print("error")


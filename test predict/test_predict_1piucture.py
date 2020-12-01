# coding=utf-8

# 載入函示庫
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D

num_classes = 3

# 模型架構
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

# 載入模型
try:
    model.load_weights("model_aug.h5")
    print("success")
except:
    print("error")
	
def plot_image(image):
    fig=plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image,cmap='binary')
    plt.show()

# 讀取樣本
img=np.array(Image.open('101.jpg'))
plot_image(img)

# 樣本前處理
#x_Test = img.reshape(1,784).astype('float32')
x_Test=cv2.resize(img,(128,128))
x_Test = x_Test.reshape(-1,128,128,3).astype('float32')
x_Test_normalize = x_Test.astype('float32') / 255.0
print(x_Test_normalize)
print(x_Test_normalize.shape)
# 樣本預測
prediction=model.predict(x_Test_normalize)
print(prediction[0])

prediction=model.predict_classes(x_Test_normalize)
print(prediction[0])
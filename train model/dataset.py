import cv2
import numpy as np
def load_data_augmentation():
    with open('./train.csv', 'r') as file:
        rows=file.readlines()
    
    x = []
    y = []
    #label = 0
    for row in rows[1:]:
        row = row.replace('\n','')

        filename = row.split(",")[0]
        label = row.split(",")[1]
        
        
        img = cv2.imread("./train_dataset/" + filename , cv2.IMREAD_COLOR)
        img = cv2.resize(img,(128,128))
        #print(img)
        x.append(img)               
        y.append(label)
        
        from numpy import expand_dims
        from tensorflow.keras.preprocessing.image import img_to_array
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        # brightness augmentation
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(brightness_range=[0.2,1.0])
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(3):
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            x.append(image)
            y.append(label)
        
        # horizontal shift augmentation
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(width_shift_range=[-200,200])
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(3):
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            x.append(image)
            y.append(label)        
        
        # vertical shift augmentation
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(height_shift_range=[-200,200])
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(3):
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            x.append(image)
            y.append(label)   
        
        # rotation augmentation
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(rotation_range=80)
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(3):
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            x.append(image)
            y.append(label)  

        # zoom augmentation
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(zoom_range=[0.5,1.0])
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(3):
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            x.append(image)
            y.append(label) 

        # extra augmentation
        if label == "3":
            img_hr = cv2.flip(img,1,dst=None) #水平鏡像
            x.append(img_hr)
            y.append(label)
        
            img_vr = cv2.flip(img,0,dst=None) #垂直鏡像
            x.append(img_vr)
            y.append(label)
            
            img_sr = cv2.flip(img,-1,dst=None) #對角鏡像
            x.append(img_sr)
            y.append(label)
            
            # rotation augmentation
            data = img_to_array(img)
            # expand dimension to one sample
            samples = expand_dims(data, 0)
            # create image data augmentation generator
            datagen = ImageDataGenerator(rotation_range=150)
            # prepare iterator
            it = datagen.flow(samples, batch_size=1)
            # generate samples and plot
            for i in range(5):
                # generate batch of images
                batch = it.next()
                # convert to unsigned integers for viewing
                image = batch[0].astype('uint8')
                x.append(image)
                y.append(label)  

    x = np.array(x, dtype=np.float32)
    y = np.array(y)
    
    np.save("train_x_aug.npy",x)
    np.save("train_y_aug.npy",y)
    
    return x,y
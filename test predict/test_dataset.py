import cv2
import numpy as np

def load_data():
    with open('./test.csv', 'r') as file:
        rows=file.readlines()
        
    x = []
    y = []
    for row in rows[1:]:
        row = row.replace('\n','')

        filename = row.split(",")[0]
        label = row.split(",")[1]
        
        img = cv2.imread("./test_image/" + filename , cv2.IMREAD_COLOR)
        img = cv2.resize(img,(128,128))
        #print(img)
        x.append(img)               
        y.append(label)
    x = np.array(x, dtype=np.float32)
    y = np.array(y)
    
    np.save("test_x.npy",x)
    np.save("test_y.npy",y)
    
    return x,y    
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

dir = 'C:\\Users\\MingSoo\\Pictures\\kfood\\AISystem'

categories = ['chicken', 'gimbab', 'kimchi', 'mandu', 'ramen']

data = []

for category in categories:
    path = os.path.join(dir, category)
    label = categories.index(category)
    
    for img in os.listdir(path):
        imgpath = os.path.join(path, img)
        food_img = cv2.imread(imgpath,0)
        try:
            food_img = cv2.resize(food_img,(50,50))
            image = np.array(food_img).flatten()
            
            data.append([image,label])
        except Exception as e:
            pass
        
        
print(len(data))

pick_in = open('data1.pickle','wb')
pickle.dump(data, pick_in)
pick_in.close()
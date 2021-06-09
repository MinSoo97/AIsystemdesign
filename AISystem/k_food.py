import os
import numpy as np
import cv2

dir = 'C:\\Users\\MingSoo\\Pictures\\kfood\\AISystem'

categories = ['chicken', 'gimbab', 'kimchi', 'mandu', 'ramen']

for category in categories:
    path = os.path.join(dir, category)
    
    for img in os.listdir(path):
        imgpath = os.path.join(path, img)
        food_img = cv2.imread(imgpath,0)
        cv2.imshow('image',food_img)
        break
    break

cv2.waitKey(0)
cv2.destroyAllWindows()
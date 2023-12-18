import os
from PIL import Image
import cv2
import numpy as np
path = 'C:/Users/Dora/Desktop/out1'
all = os.walk(path)
for path, dir, filelist in all:
    for filename in filelist:
        if filename.endswith('.jpg'):
            filepath = os.path.join(path, filename)
            img = cv2.imread(filepath, 1)
            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

            calhe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
            img_yuv[:, :, 0] = calhe.apply(img_yuv[:, :, 0])
            imgLoaclEhist = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            # cv2.imshow('local hist',imgLoaclEhist)

            cv2.imwrite(filepath, imgLoaclEhist)
            cv2.waitKey(0)
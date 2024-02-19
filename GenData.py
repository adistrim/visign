import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
offset = 20
imgSize = 300
counter = 0

folder = './data/hello'
# folder = './data/yes'
# folder = './data/no'
# folder = './data/ilikeyou'
# folder = './data/thankyou'

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]
        
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
        if imgCrop.size != 0:  # Check if imgCrop is not empty
            imgCropShape = imgCrop.shape
            
            aspectRatio = h/w
            
            if aspectRatio > 1:
                k = imgSize/h
                wCal = math.ceil(w*k)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal)/2)
                imgWhite[:, wGap:wCal+wGap] = imgResize
            else:
                k = imgSize/w
                hCal = math.ceil(h*k)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal)/2)
                imgWhite[hGap:hCal+hGap, :] = imgResize
                
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)
        
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    
    if key == ord('s'):
        counter += 1
        imgName = f"{folder}/Image_{time.time()}.jpg"
        cv2.imwrite(imgName, imgWhite)
        print(f"Image_{time.time()}.jpg saved")

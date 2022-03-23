import cv2
import numpy as np

class blueDetector():
    
    def __init__(self):
        
        self.low_blue = np.array([109,187,33])
        self.high_blue = np.array([129,255,255])
    
    def detect(self,frame):
        
        frame_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(frame_hsv,self.low_blue,self.high_blue)
        
        contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        contour = sorted(contours,key = lambda x: cv2.contourArea(x),reverse=True)[0]
        
        (x,y,w,h) = cv2.boundingRect(contour)
        box = (x,y,x+w,y+h)
        return box
        
            

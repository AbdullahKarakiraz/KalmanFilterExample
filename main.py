import cv2
from BlueDetector import blueDetector
from kalmanFilter import KalmanFilter

blue_detector = blueDetector()
kf = KalmanFilter()
cap = cv2.VideoCapture(0)

while True:
    
    ret,frame = cap.read()
    
    blue_box = blue_detector.detect(frame)
    
    x1,y1,x2,y2 = blue_box
    
    pt1 = int((x1+x2) / 2)
    pt2 = int((y1+y2) / 2)
    
    predicted = kf.predict(pt1,pt2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 4)
    cv2.circle(frame,(pt1,pt2),5,[0,255,255],-1)
    cv2.circle(frame, (predicted[0], predicted[1]), 7, (0, 0, 255), 4)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
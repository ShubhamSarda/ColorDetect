import cv2
import numpy as np

lowerBound = np.array([33,80,40])
upperBound = np.array([102,255,255])

kernelOpen = np.ones((5,5))
kernelClose = np.ones((20,20))

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    check, img = cam.read()
    img = cv2.resize(img,(340,220))
	
    #Cover BGR to HSV
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
	#Create Mask
    mask = cv2.inRange(imgHSV,lowerBound,upperBound)

    #Morphology
    maskOpen = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
    maskClose = cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)
	
	#http://docs.opencv.org/2.4/doc/tutorials/imgproc/opening_closing_hats/opening_closing_hats.html
    maskFinal=maskClose
    conts, h=cv2.findContours(maskFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
	
    #cv2.findContours(image, mode, method[, contours[, hierarchy[, offset]]]) → image, contours, hierarchy
    #contours, hierarchy
    #https://stackoverflow.com/questions/28113221/findcontours-and-drawcontours-errors-in-opencv-3-beta-python
    
	cv2.drawContours(img,conts,-1,(255,0,0),3)
    for i in range(len(conts)):
        x,y,w,h=cv2.boundingRect(conts[i])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255), 2)
        cv2.putText(img, str(i+1),(x,y+h),font, 1,(0,255,255),2,cv2.LINE_AA)

    cv2.imshow("maskClose", maskClose)
    cv2.imshow("cam", img)
    if cv2.waitKey(1) == ord('q'):
        break

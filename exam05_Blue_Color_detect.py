import numpy as np
import cv2

lower_blue = np.array([90,100,100])
upper_blue = np.array([130,255,255])
lower_red = np.array([6,100,100])
upper_red = np.array([10,255,255])

cap = cv2.VideoCapture(0)
cap.set(3,480)
cap.set(4,320)

right_count = 0
left_count = 0

while True :
    ret, frame = cap.read()

    cv_image_input = frame

    hsv = cv2.cvtColor(cv_image_input,cv2.COLOR_BGR2HSV)

    #Line for dividing left and right area
    #cv2.line(cv_image_input,(90,0),(90,240),(255,0,0),3)
    #cv2.line(cv_image_input,(162,0),(162,240),(255,0,0),3)
    #cv2.line(cv_image_input,(235,0),(235,240),(255,0,0),3)

    cv2.line(cv_image_input,(160,0),(160,480),(255,0,0),3)
    #cv2.line(cv_image_input,(162,0),(162,240),(255,0,0),3)
    #cv2.line(cv_image_input,(235,0),(235,240),(255,0,0),3)


    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray,3)
    gray_blurred = cv2.GaussianBlur(median,(3,3),0)
    ret, threshold = cv2.threshold(gray_blurred,210,255,cv2.THRESH_BINARY)

    mask_blue = cv2.inRange(hsv,lower_blue,upper_blue)

    blue = cv2.bitwise_and(cv_image_input,cv_image_input,mask=mask_blue)

    blue_gray = cv2.cvtColor(blue,cv2.COLOR_BGR2GRAY)

    blue_gray_blurred = cv2.GaussianBlur(blue_gray,(5,5),0)

    ret_b, thresh_b = cv2.threshold(blue_gray_blurred,0,255,cv2.THRESH_BINARY)
    _, center_blue_contours, hierarchy = cv2.findContours(thresh_b, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    blue_x = []
    blue_y = []

    center_blue_x = []
    center_blue_y = []

    for c in center_blue_contours:
        peri = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,0.04*peri,True)
        (x,y,w,h) = cv2.boundingRect(approx)
        end=x+w

        if w>20 and w<100 and h<60 and x<250 and end<280 and len(approx)!= 3 and len(approx)!= 6 :
            center_blue_x.append(x)
            center_blue_y.append(y)
            cv2.drawContours(cv_image_input, [c], -1, (255,0,0),3)

    if not center_blue_x:
       pass

    elif center_blue_x[0] > 160 :
       print("right accracy : %d, count : %d ", center_blue_x[0], right_count+1)
       right_count +=1
       if right_count == 3:
           print("====================================")
           print("final destination : right")
           print("====================================")
           right_count = 0

    elif  center_blue_x[0] < 160 :
       print("left accuracy : %d, count : %d ", center_blue_x[0], left_count+1)
       left_count +=1

       if left_count == 3:
           print("====================================")
           print("final destination : left")
           print("====================================")
           left_count = 0

    cv2.imshow('blud_area',cv_image_input), cv2.waitKey(1) & 0xFF

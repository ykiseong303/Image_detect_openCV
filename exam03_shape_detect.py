#ure.py
#devel@devel-HP-ProBook-450-G1:~/catkin_ws/au
import numpy as np
import cv2
import math


def angle(pt1,pt2,pt0):
    dx1 = pt1[0][0] - pt0[0][0]
    dy1 = pt1[0][1] - pt0[0][1]
    dx2 = pt2[0][0] - pt0[0][0]
    dy2 = pt2[0][1] - pt0[0][1]

    return float((dx1*dx2 + dy1*dy2))/math.sqrt(float((dx1*dx1 + dy1*dy1))*(dx2*dx2 + dy2*dy2) + 1e-10)
contours = {}
approx = []
scale = 2
#pic_num = 0
cap = cv2.VideoCapture(1)
    	    #print("before_while")
pic_num = 0
#ret, frame = cap.read()
    #pic_num = 0
while True :
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(frame,300,240,3)

    canny2, contours, hierarchy = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0,len(contours)):
        approx = cv2.approxPolyDP(contours[i],cv2.arcLength(contours[i],True)*0.02,True)

        if(abs(cv2.contourArea(contours[i]))< 300 or not(cv2.isContourConvex(approx))):
            continue
        if(len(approx) == 3) :
            x,y,w,h = cv2.boundingRect(contours[i])
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
            for j in range(3):
                a = cv2.circle(frame, (approx[j][0][0], approx[j][0][1]), 2, (0,255,0), thickness=3, lineType=8, shift=0)

                if w>50 and h>50 :
                    new_img= frame[y:y+h,x:x+w]
                    print(pic_num)
                    cv2.putText(frame,'TRIANGLE',(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),2,cv2.LINE_AA)
        if(len(approx)>=4 and len(approx)<=6):
            vtc = len(approx)
            cos = []
            for j in range(2,vtc+1):
                cos.append(angle(approx[j%vtc],approx[j-2],approx[j-1]))
            cos.sort()
            mincos = cos[0]
            maxcos = cos[-1]

            x,y,w,h = cv2.boundingRect(contours[i])
            if(vtc==4):
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
                for j in range(3):
                    a = cv2.circle(frame, (approx[j][0][0], approx[j][0][1]), 2, (0,255,0), thickness=3, lineType=8, shift=0)
                if w>50 and h>50 :
                    new_img= frame[y:y+h,x:x+w]
                    print(pic_num)
                    pic_num += 1
                    cv2.putText(frame,'RECTANGLE',(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),2,cv2.LINE_AA)
        else :
            area = cv2.contourArea(contours[i])
            x,y,w,h = cv2.boundingRect(contours[i])
            radius = w/2
            if(abs(1 - (float(w)/h))<=2 and abs(1-(area/(math.pi*radius*radius)))<=0.2):
                cv2.drawContours(frame, [approx], -1, (0, 255, 0),3 )
                for j in range(3):
                    a = cv2.circle(frame, (approx[j][0][0], approx[j][0][1]), 2, (0,255,0), thickness=3, lineType=8, shift=0)

                if w>50 and h>50 :
                    new_img= frame[y:y+h,x:x+w]
                print(pic_num)
                pic_num += 1
                cv2.putText(frame,'CIRC',(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),2,cv2.LINE_AA)
    cv2.imshow('frame',frame), cv2.waitKey(1) & 0xFF

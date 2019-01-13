import tensorflow as tf
import numpy as np
import cv2
import math
import time

modelFullPath = '/home/devel/opencv_ws/TensorFlow/graph/exam02_model.pb'
labelsFullPath = '/home/devel/opencv_ws/TensorFlow/graph/exam02_labels.txt'


def create_graph():
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def angle(pt1,pt2,pt0):
    dx1 = pt1[0][0] - pt0[0][0]
    dy1 = pt1[0][1] - pt0[0][1]
    dx2 = pt2[0][0] - pt0[0][0]
    dy2 = pt2[0][1] - pt0[0][1]
    return float((dx1*dx2 + dy1*dy2))/math.sqrt(float((dx1*dx1 + dy1*dy1))*(dx2*dx2 + dy2*dy2) + 1e-10)

def run_camera(pic_num, frame) :
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(frame,80,240,3)
    kernel = np.ones((1,1), np.uint8)

    canny2, contours, hierarchy = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0,len(contours)):
        approx = cv2.approxPolyDP(contours[i],cv2.arcLength(contours[i],True)*0.02,True)

        if(abs(cv2.contourArea(contours[i]))<100 or not(cv2.isContourConvex(approx))):
            continue
        if(len(approx) == 3) :
            x,y,w,h = cv2.boundingRect(contours[i])
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
            for j in range(3):
                a = cv2.circle(frame, (approx[j][0][0], approx[j][0][1]), 2, (0, 255,0), thickness=3, lineType=8, shift=0)

            if w>50 and h>50 :
                new_img= frame[y:y+h,x:x+w]
                #cv2.imwrite('objects/'+str(pic_num)+'.jpg',new_img)
                #imagePath = 'objects/0.jpg'
                run_inference_on_image(new_img)


        elif(len(approx)>=4 and len(approx)<=6):
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
                    a = cv2.circle(frame, (approx[j][0][0], approx[j][0][1]), 2, (0, 255,0), thickness=3, lineType=8, shift=0)
                if w>50 and h>50 :
                    new_img= frame[y:y+h,x:x+w]
                    #cv2.imwrite('objects/'+str(pic_num)+'.jpg',new_img)
                    #imagePath = 'objects/0.jpg'
                    run_inference_on_image(new_img)
        else:
            area = cv2.contourArea(contours[i])
            x,y,w,h = cv2.boundingRect(contours[i])
            radius = w/2
            if(abs(1 - (float(w)/h))<=2 and abs(1-(area/(math.pi*radius*radius)))<=0.2):
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
                for j in range(3):
                    a = cv2.circle(frame, (approx[j][0][0], approx[j][0][1]), 2, (0, 255,0), thickness=3, lineType=8, shift=0)
                if w>50 and h>50 :
                    new_img= frame[y:y+h,x:x+w]
                    #cv2.imwrite('objects/'+str(pic_num)+'.jpg',new_img)
                    #imagePath = 'objects/0.jpg'
                    run_inference_on_image(new_img)
    cv2.imshow('frame',frame), cv2.waitKey(1) & 0xFF
	    #cv2.imshow('Canny',canny)

def run_inference_on_image(frame):
    print("Start tensorflow")
    answer = None

    image_data = np.array(frame)[:,:,0:3]

    create_graph()

    with tf.Session() as sess:

        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg:0': image_data})
        predictions = np.squeeze(predictions)

        top_k = predictions.argsort()[-2:][::-1]
        f = open(labelsFullPath, 'rb')
        lines = f.readlines()
        labels = [str(w).replace("\n", "") for w in lines]
        for node_id in top_k:
            human_string = labels[node_id]
            score = predictions[node_id]
            print('%s (score = %.5f)' % (human_string, score))
        print('---------------------------')
        answer = labels[top_k[0]]
        #return answer



contours = {}
approx = []
scale = 2
cap = cv2.VideoCapture(1)
if __name__ == '__main__':
       #cap = cv2.VideoCapture(1)
       #ret, frame = cap.read()
       pic_num = 0
       while True :
          ret, frame = cap.read()
          run_camera(pic_num,frame)

import numpy as np
import cv2



def Calculate_MSE (arr1, arr2) :
    squared_diff = (arr1-arr2) **2
    sum = np.sum(squared_diff)
    num_all = arr1.shape[0] * arr1.shape[1]

    err = sum / num_all
    return err
cap = cv2.VideoCapture(0)
sift = cv2.xfeatures2d.SIFT_create()
image_path = 'exam_image/exam01.jpg'

img_origin_1 = cv2.imread(image_path,cv2.IMREAD_COLOR)
origin_kp1, origin_des1 = sift.detectAndCompute(img_origin_1,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params,search_params)

while True :
    ret, frame = cap.read()
    cv_image_input = frame

    MIN_MATCH_COUNT = 9
    MIN_MSE_DECISION = 80000

    recog_kp1, recog_des1 = sift.detectAndCompute(cv_image_input, None)

    recog_match1 = flann.knnMatch(recog_des1,origin_des1,k=2)

    image_result = 1

    recog_value = []
    for m,n in recog_match1 :
        if m.distance < 0.7*n.distance :
            recog_value.append(m)
    print("Image_recognition Value : %d",len(recog_value))

    if len(recog_value) > MIN_MATCH_COUNT :
        src_pts = np.float32([ recog_kp1[m.queryIdx].pt for m in recog_value ]).reshape(-1,1,2)
        dst_pts = np.float32([ origin_kp1[m.trainIdx].pt for m in recog_value ]).reshape(-1,1,2)

        M,mask = cv2.findHomography(src_pts,dst_pts, cv2.RANSAC,5.0)

        matchesMask = mask.ravel().tolist()

        mse = Calculate_MSE(src_pts,dst_pts)
        print("Image_recognition MSE :%d",mse)
        if mse > MIN_MSE_DECISION :
            print("RECOGNITION COMPLETED %d %d",len(recog_value),mse)
            image_result = 2
    else :
        matchesMask = None

    if image_result == 1:
        pass
    elif image_result == 2:
        draw_params2 = dict(matchColor = (0,0,255), # draw matches in green color
                        singlePointColor = None,
                        matchesMask = matchesMask, # draw only inliers
                        flags = 2)
        final_image = cv2.drawMatches(cv_image_input, recog_kp1, img_origin_1,origin_kp1,recog_value,None,**draw_params2)
        cv2.imshow('final_image', final_image), cv2.waitKey(1) & 0xFF
    #cv2.imshow('frame',cv_image_input), cv2.waitKey(1) & 0xFF

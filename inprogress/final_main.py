import cv2
import dlib
import numpy as np
import threading
from person_and_phone import load_darknet_weights, draw_outputs
from eye_tracker import *
from dlib_helper import (shape_to_np, 
                          eye_on_mask,
                          contouring,
                          process_thresh, 
                          print_eye_pos,
                          nothing)
from define_mouth_distances import return_distances

load_darknet_weights(yolo, 'yolov3.weights') 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_68.dat')

d_outer, d_inner = return_distances(detector, predictor)
cap = cv2.VideoCapture(0)
_, frame_size = cap.read()

def eyes_mouth():
    
    face_model = get_face_detector()
landmark_model = get_landmark_model()
left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

cap = cv2.VideoCapture(0)
ret, img = cap.read()
thresh = img.copy()

cv2.namedWindow('image')
kernel = np.ones((9, 9), np.uint8)

def nothing(x):
    pass
    cv2.createTrackbar('threshold', 'image', 75, 255, nothing)

    while(True):
        ret, img = cap.read()
        rects = find_faces(img, face_model)
        
        for rect in rects:
            shape = detect_marks(img, landmark_model, rect)
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            mask, end_points_left = eye_on_mask(mask, left, shape)
            mask, end_points_right = eye_on_mask(mask, right, shape)
            mask = cv2.dilate(mask, kernel, 5)
            
            eyes = cv2.bitwise_and(img, img, mask=mask)
            mask = (eyes == [0, 0, 0]).all(axis=2)
            eyes[mask] = [255, 255, 255]
            mid = int((shape[42][0] + shape[39][0]) // 2)
            eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
            threshold = cv2.getTrackbarPos('threshold', 'image')
            _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
            thresh = process_thresh(thresh)
            
            eyeball_pos_left = contouring(thresh[:, 0:mid], mid, img, end_points_left)
            eyeball_pos_right = contouring(thresh[:, mid:], mid, img, end_points_right, True)
            print_eye_pos(img, eyeball_pos_left, eyeball_pos_right)
            # for (x, y) in shape[36:48]:
            #     cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
            
        cv2.imshow('eyes', img)
        cv2.imshow("image", thresh)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
def count_people_and_phones():
    while(True):
        ret, image = cap.read()
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (320, 320))
        frame = frame.astype(np.float32)
        frame = np.expand_dims(frame, 0)
        frame = frame / 255
        class_names = [c.strip() for c in open("classes.txt").readlines()]
        boxes, scores, classes, nums = yolo(frame)
        count=0
        for i in range(nums[0]):
            if int(classes[0][i] == 0):
                count +=1
            if int(classes[0][i] == 67):
                print("Mobile Phone Detected")
        if count == 0:
            print('No person detected')
        elif count > 1: 
            print('More than one person detected')
        image = draw_outputs(image, (boxes, scores, classes, nums), class_names)
        cv2.imshow('Prediction', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

         
t1 = threading.Thread(target=eyes_mouth) 
# t2 = threading.Thread(target=count_people_and_phones) 
t1.start() 
# t2.start() 
# t1.join() 
# t2.join() 
cap.release()
cv2.destroyAllWindows()
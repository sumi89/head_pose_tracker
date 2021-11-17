import cv2
import numpy as np
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
img = cv2.imread('test3.jpg')

h, w = img.shape[:2] # h=431, w=756
#ROI
roi_h_l = int(h*0.2)
roi_h_u = h - int(h*0.2)
roi_w_l = int(w*0.2)
roi_w_u = w - int(w*0.2)

invalid_h = int((roi_h_u-roi_h_l)*0.1)
invalid_w = int((roi_w_u-roi_w_l)*0.1)

center_h = (roi_h_u+roi_h_l)//2
center_w = (roi_w_u+roi_w_l)//2


blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
(300, 300), (104.0, 117.0, 123.0))
net.setInput(blob)
faces = net.forward()
#to draw faces on image
for i in range(faces.shape[2]):
    confidence = faces[0, 0, i, 2]
    if confidence > 0.5:
        box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x, y, x1, y1) = box.astype("int")

        # choose the faces those are inside the ROI 
        # discard the faces those are less than 10% of the ROI
        if x>=roi_w_l and x1<=roi_w_u and y>=roi_h_l and y1<=roi_h_u and x1-x>=invalid_w and y1-y>=invalid_h:
            print('x=', x, 'y=', y, 'x1=', x1, 'y1=', y1, 'x1-x:', x1-x, 'y1-y:', y1-y)
            # mark the face in the center with green color
            if x<center_w<x1 and y<center_h<y1:
                req_x, req_y, req_x1, req_y1 = x, y, x1, y1
                cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 2)
            else:
                cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)

# to write the resulting image           
#cv2.imwrite('face_result_3_roi_2.png', img)

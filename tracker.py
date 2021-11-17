import cv2
import numpy as np

cap = cv2.VideoCapture('videos/foreign_object_and_occlusion.mp4')
font = cv2.FONT_HERSHEY_SIMPLEX 

while(True):
    ret, img = cap.read()
    if ret == True:
        img = cv2.resize(img, None, fx=0.5, fy=0.5)
        height, width = img.shape[:2]
        img2 = img.copy()

        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)),
                                        1.0, (300, 300), (104.0, 117.0, 123.0))
        net.setInput(blob)
        faces3 = net.forward()

        for i in range(faces3.shape[2]):
            confidence = faces3[0, 0, i, 2]
            if confidence > 0.5:
                box = faces3[0, 0, i, 3:7] * np.array([width, height, width, height])
                (x, y, x1, y1) = box.astype("int")
                cv2.rectangle(img2, (x, y), (x1, y1), (0, 0, 255), 2)
        cv2.putText(img2, 'dnn', (30, 30), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
                

        cv2.imshow("dnn", img2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

    
cap.release()
cv2.destroyAllWindows()


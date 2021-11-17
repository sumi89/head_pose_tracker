# Head Pose Tracker
The goal of this project is to create a head pose tracker followed by detecting face in an image and identifying facial landmarks in the detected face. OpenCV's Face Detection Neural Network is used in this project.

# Setup
1) Download and install Anaconda (https://www.anaconda.com/products/individual)
2) Install all dependencies using pip install -r requirements.txt

# Usage
1) "test.jpg" is the sample image 
2) 'face_detection.py' detects the faces inside the ROI, puts green bounding box around the face that is in the center of the image and red bounding box for the other images
3) 'tracker.py' tracks the position of head movement. There are some '.mp4' files to show that the tracking algorithm is able to track the head pose with a phone. This still works even if the head is occluded with the phone. This algorithm stops tracking when the head is out of focus, but starts tracking again when the head comes back. 






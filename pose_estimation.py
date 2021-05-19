import cv2
import numpy as np
import os
import mediapipe as mp


image_path = './images/'

# Read images with OpenCV.
print(os.listdir(image_path))
images = {name: cv2.imread(image_path + name) for name in os.listdir(image_path)}

# for name, image in images.items():
#     print(name)
#     cv2.imshow(name, image)

# key = cv2.waitKey(0)
# cv2.destroyAllWindows()



mp_pose = mp.solutions.pose
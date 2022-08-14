import cv2
import matplotlib.pyplot as plt
import numpy as np
from deepface import DeepFace


img = cv2.imread('sad_woman.jpg')

plt.imshow(img)
print(plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)))

predictions = DeepFace.analyze(img)
print(predictions['dominant_emotion'])


cv2.waitKey(0)














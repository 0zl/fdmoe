import urllib.request
import numpy as np
import cv2

from anime_face_detector import create_detector

test_image = 'https://thumbs.dreamstime.com/z/anime-girl-cat-ears-cigarette-hand-anime-girl-cat-ears-cigarette-hand-vector-227737852.jpg'

req_image = urllib.request.urlopen(test_image)
img_array = np.array(bytearray(req_image.read()), dtype=np.uint8)

img = cv2.imdecode(img_array, -1)

detector = create_detector('yolov3', device='cpu')
preds = detector(img)

print(preds)

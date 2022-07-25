from typing import Union
from fastapi import FastAPI
from anime_face_detector import create_detector
from starlette.responses import StreamingResponse

import io
import cv2
import urllib.request
import numpy as np

waifu_detector = create_detector('yolov3', device='cpu')

def get_image(url):
	req = urllib.request.urlopen(url)
	arr = np.array(bytearray(req.read()), dtype=np.uint8)
	return cv2.imdecode(arr, -1)

def detect_waifu(raw, simple=True):
	preds = waifu_detector(raw)
	result = []

	for pr in preds:
		box = pr['bbox']
		box, score = box[:4], box[4]

		if simple:
			box = box.tolist()
			score = score.astype(float)

		result.extend([ [box, score] ])

	return result

def bounding_box(img, preds, threshold=0.5):
	for pr in preds:
		box, score = pr[0], pr[1]

		if np.float32(score).astype(float) < threshold:
			continue

		box = np.round(box).astype(int)
		linew = max(2, int(3 * (box[2:] - box[:2]).max() / 256))

		cv2.rectangle(img, tuple(box[:2]), tuple(box[2:]), (255, 0, 0), linew)

	return img

app = FastAPI()

@app.get("/")
def index():
	return 'fdmoe'

@app.get("/data")
def get_data(url: str):
	raw = get_image(url)
	data = detect_waifu(raw)
	return data

@app.get("/draw")
def draw_box(url: str):
	raw = get_image(url)
	data = detect_waifu(raw, False)
	img = bounding_box(raw, data)

	res, im_png = cv2.imencode('.png', img)
	return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type='image/png')

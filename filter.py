# import necessary packages 
import face_recognition
import numpy as np
import cv2

mask = cv2.imread("santa.png", -1) # Reads the filter image
cap = cv2.VideoCapture(0) 

class Mask:

	# this method will overlay the filter to the video
	def overlay(self, frame, mask, pos=(0,0), scale = 1): 

		h, w, _ = mask.shape
		rows, cols, _ = frame.shape
		y, x = pos[0], pos[1]

		for i in range(h):
			for j in range(w):
				if x + i >= rows or y + j >= cols:
					continue

				alpha = float(mask[i][j][3] / 255.0)
				frame[x + i][y + j] = alpha * mask[i][j][:3] + (1 - alpha) * frame[x + i][y + j]

		return frame

while True:

	santa = Mask()

	ret, image = cap.read()

	img_frame = image[:, :, ::-1]

	# this variable will detect the face locations in the video
	ext = face_recognition.face_locations(img_frame)
	faces = [(0,0,0,0)] # this will initialize the coordinates of the face

	# if there is a face detected in the stream, this part will be executed
	if ext != []:

		faces = [[ext[0][3], ext[0][0], abs(ext[0][3] - ext[0][1]) + 150, abs(ext[0][0] - ext[0][2])]]
		
		for (x, y, w, h) in faces:

			# Manually adjust the location of image filter
			x -= 60
			w -= 30
			y -= 35
			h -= 10			

			mask_min = int(y - 3 * h / 5)
			mask_max = int(y + 8 * h / 5)

			face_mask = mask_max - mask_min

			face_frame = image[mask_min:mask_max, x:x+w]
			mask_resized= cv2.resize(mask, (w, face_mask), interpolation=cv2.INTER_CUBIC) # resize the image filter

			santa.overlay(face_frame, mask_resized) # calls the overlay method from the Mask class

	# show video stream
	cv2.imshow("Face Filter", image)

	key = cv2.waitKey(1) & 0xff	

	# press q to quit
	if key == ord("q"):
		break

cap.release()
cv2.destroyAllWindows()
	

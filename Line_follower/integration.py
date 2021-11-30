#! /usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import roslib
import sys
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

sess1 = tf.Session()    
graph1 = tf.get_default_graph()
set_session(sess1)


class image_converter:

	def __init__(self):
		self.image_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.callback)
		self.plate_NN = models.load_model('/home/fizzer/Downloads/model.h5')


	def callback(self, data):
		try:
			img = self.bridge.imgmsg_to_cv2(data, "bgr8")

		except CvBridgeError as e:
			print(e)

		cv2.imshow("Camera feed", img)
		cv2.waitKey(3)




		pre_mask_img = cv2.medianBlur(img,5)
		pre_mask_img = cv2.cvtColor(pre_mask_img, cv2.COLOR_BGR2HSV)


		lower_hsv_car = np.array([120,135,100])
		upper_hsv_car = np.array([120,255,105])

		masked_img = cv2.inRange(pre_mask_img, lower_hsv_car, upper_hsv_car)

		cv2.imshow("Masked feed", masked_img)
		cv2.waitKey(3)

		contours, hierarchy = cv2.findContours(image=masked_img, mode=cv2.RETR_EXTERNAL, 
			method=cv2.CHAIN_APPROX_SIMPLE)[-2:]

		# filter out all contours except the 2 with the largest area
		contours = sorted(contours, key = cv2.contourArea, reverse = True)[:2] 

		# print(len(contours))

		if len(contours) >= 2:
			smaller_contour_area = cv2.contourArea(contours[-1])
			print(smaller_contour_area)
			if smaller_contour_area > 1000:
				# contour detection

				print("Entering contour detection")

				# convert image to grayscale
				# img_gray = cv2.cvtColor(mask_car, cv2.COLOR_BGR2GRAY)
				# h, s, v = cv2.split(masked_img)
				# img_gray = v
				# img_gray = masked_img[:,:,2]

				# # apply binary thresholding
				ret, thresh = cv2.threshold(masked_img, 150, 255, cv2.THRESH_BINARY)

				# detect contours on the binary image using cv2.CHAIN_APPROX_SIMPLE
				contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, 
					method=cv2.CHAIN_APPROX_SIMPLE)[-2:]

				# filter out all contours except the 2 with the largest area
				contours = sorted(contours, key = cv2.contourArea, reverse = True)[:2] 

				contour0 = cv2.boundingRect(contours[0])
				contour1 = cv2.boundingRect(contours[1])

				# determine which contour is on the left 
				if contour0[0] < contour1[0]:
					left = contour0
					right = contour1
				else:
					left = contour1
					right = contour0

				# append contour coords to array
				arr = []

				xL,yL,wL,hL = left
				arr.append((xL+(wL-1),yL))
				arr.append((xL+wL,yL+hL))

				xR,yR,wR,hR = right
				arr.append((xR-1,yR))
				arr.append((xR,yR+hR))

				box = cv2.minAreaRect(np.asarray(arr))
				box = cv2.boxPoints(box) # (BL, TL, TR, BR)

				# calculate height to width ratio to find desired height of output image
				width, height = np.amax(box, axis=0) - np.amin(box, axis=0) # max[width, height] - min[width, height]
				hw_ratio = height/width

				dst_width = 600 # license plates are 600 pixels wide
				height_offset = 350 # to improve height to width ratio of dst image
				dst_height = int(600*hw_ratio) + height_offset

				# locate points for perspective transform in source and destination images
				# (TL, TR, BL, BR)
				src_pts = np.float32([box[1], box[2], box[0], box[3]])
				dst_pts = np.float32([[0, 0], [dst_width, 0], [0, dst_height], [dst_width, dst_height]])

				img_copy = img.copy()

				# draw contours on original image 
				box = np.int0(box)
				cv2.drawContours(img,[box],0,(0,0,255),2)

				matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
				result = cv2.warpPerspective(img_copy, matrix, (dst_width, dst_height))

				# convert image from BGR to RGB
				result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

				# crop image to only license plate
				# convert image to grayscale
				im_gray = cv2.cvtColor(np.float32(result), cv2.COLOR_RGB2GRAY)

				# crop to only license plate by summing columns and finding colour changes
				im_arr = np.array(im_gray)

				# print(im_arr.shape)
				sum = np.sum(im_arr[:,:50], axis=1)

				colour_threshold = 50

				col_idx = [] 

				for i in range(1, len(sum)):
					current_row = sum[i]
					prev_row = sum[i-1]

					if np.abs(current_row - prev_row) > colour_threshold:
						col_idx.append(i)

				# filter out values within 100 pixels of the top and bottom of the image
				filter_threshold = 100
				col_idx = np.array(col_idx)
				col_idx = col_idx[np.where(col_idx > filter_threshold)]
				col_idx = col_idx[np.where(col_idx < (len(sum)-filter_threshold))]

				top = col_idx[0] 
				bottom = col_idx[-1] 

				crop_im = result[top:bottom, 0:600, :]

				# resize image to 600 x 298
				resize_img = cv2.resize(crop_im, dsize=(600,298), interpolation=cv2.INTER_CUBIC)
				
				pos = [(48, 75, 153, 240), 
					   (153, 75, 258, 240),
					   (345, 75, 450, 240),
					   (450, 75, 555, 240)]

				encoding = {}

				alphanum = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
				for i in range(36):
					encoding[alphanum[i]] = i

				# creating a new mapping with the keys and values reversed from the original mapping
				# https://stackoverflow.com/questions/483666/reverse-invert-a-dictionary-mapping 
				unencoding = dict(zip(encoding.values(), encoding.keys()))

				def preprocess_input(test_image):
					predicted_license_plate = ""

					im = test_image

					for i in range(4):
						crop_im = im[pos[i][1]:pos[i][3], pos[i][0]:pos[i][2], :]
						normalized_im = np.array([crop_im/255.])

						global sess1
						global graph1
						with graph1.as_default():
							set_session(sess1)
							prediction = self.plate_NN.predict(normalized_im)


						# prediction = loaded_model.predict(normalized_im) # array of probabilities

						# array of the locations of probabilities sorted from highest to lowest
						# note: the location is the predicted label
						# https://stackoverflow.com/questions/27473601/next-argmax-values-in-python/27473888
						ordered_prediction = np.argsort(-prediction, axis=1) 
						if i < 2:
							for index in ordered_prediction[0]:
								if index <= 25:
									predicted_label = index
									break
						else:
							for index in ordered_prediction[0]:
								if index > 25:
									predicted_label = index
									break

						unencoded_label = unencoding[predicted_label]
						predicted_license_plate += unencoded_label

					return predicted_license_plate

				print(preprocess_input(resize_img))







		# width = cv_image.shape[1]
		# camera_center = width // 2

		# bottom = cv_image[-100,:,0]

		# array = []
		# # threshold = int((np.amax(bottom) + np.amin(bottom)) / 2)
		# threshold = 75

		# for x in bottom:
		# 	if x > threshold:
		# 		array.append(1)
		# 	else:
		# 		array.append(0)

		# arr = np.array(array)

		# line = np.where(arr == 0)
		# line = np.array(line)
		# line = line[0,:]

		# rate = rospy.Rate(2)
		# move = Twist()
		# move.linear.x = 0.5
		# # move.angular.z = 0.2
		# move.angular.z = 0.2

		# margin = 20

		# if len(line) != 0:
		#     x = (line[0] + line[-1]) // 2

		#     if x > (camera_center + margin):
		#     	# move.angular.z = -1  # positive is left
		#     	pass
		#     elif x < (camera_center + margin):
		#     	move.angular.z = 3
		#     else: 
		#     	move.angular.z = 0

		# print(move.angular.z, move.linear.x)
		# self.image_pub.publish(move)


def main(args):	
	ic = image_converter()
	rospy.init_node('image_converter', anonymous=True)
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")
	cv2.destroyAllWindows()

if __name__ ==  '__main__':
	main(sys.argv)
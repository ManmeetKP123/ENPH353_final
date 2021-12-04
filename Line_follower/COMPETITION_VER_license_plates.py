#! /usr/bin/env python

import rospy
import time
import roslib
import sys
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from geometry_msgs.msg import Twist

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

sess1 = tf.Session()    
graph1 = tf.get_default_graph()
set_session(sess1)


class image_converter:

	def __init__(self):
		self.bridge = CvBridge()
		self.nav_pub = rospy.Publisher("/R1/cmd_vel",Twist, queue_size=1)
		self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.callback)
		self.plate_pub = rospy.Publisher("/license_plate", String, queue_size=1)
		self.plate_NN = models.load_model('/home/fizzer/ros_ws/src/controller_pkg/Line_follower/hopesandprayers.h5')

		self.team_id = "Vroom"
		self.team_pswd = "vroom88"

		self.predict = None
		self.list_of_plates = []
		self.accumulative_counter = 0
		self.got_3 = False
		self.got_5 = False
		self.got_1 = False
		self.got_2 = False

	def callback(self, data):
		try:
			img = self.bridge.imgmsg_to_cv2(data, "bgr8")

		except CvBridgeError as e:
			print(e)

		pre_mask_img = cv2.medianBlur(img,5)
		pre_mask_img = cv2.cvtColor(pre_mask_img, cv2.COLOR_BGR2HSV)

		lower_hsv1 = np.array([120,135,100])
		upper_hsv1 = np.array([120,255,105])

		lower_hsv2 = np.array([120,200,120])
		upper_hsv2 = np.array([120,255,130])

		lower_hsv3 = np.array([120,120,130])
		upper_hsv3 = np.array([120,255,255])

		hsv_ranges = [[lower_hsv1, upper_hsv1], 
					  [lower_hsv2, upper_hsv2],
					  [lower_hsv3, upper_hsv3]]

		masked_img = 0
		for lower, upper in hsv_ranges:
			pre_mask_img_copy = pre_mask_img.copy()
			masked_img += cv2.inRange(pre_mask_img_copy, lower, upper)

		contours, hierarchy = cv2.findContours(image=masked_img, mode=cv2.RETR_EXTERNAL, 
			method=cv2.CHAIN_APPROX_SIMPLE)[-2:]

		# filter out all contours except the 2 with the largest area
		contours = sorted(contours, key = cv2.contourArea, reverse = True)[:2] 

		self.predict = None
		if len(contours) >= 2:
			smaller_contour_area = cv2.contourArea(contours[-1])
			if smaller_contour_area > 2800:
				print("Entering contour detection")

				# apply binary thresholding
				ret, thresh = cv2.threshold(masked_img, 150, 255, cv2.THRESH_BINARY)

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
				matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
				result = cv2.warpPerspective(img_copy, matrix, (dst_width, dst_height))

				# convert image to grayscale
				im_gray = cv2.cvtColor(np.float32(result), cv2.COLOR_BGR2GRAY)

				# crop to only license plate by summing columns and finding colour changes
				im_arr = np.array(im_gray)
				sum = np.sum(im_arr[:,:50], axis=1)

				colour_threshold = 50
				col_idx = [] 

				for i in range(1, len(sum)):
					current_row = sum[i]
					prev_row = sum[i-1]

					if np.abs(current_row - prev_row) > colour_threshold:
						col_idx.append(i)

				# filter out values within 800 pixels of the top and 100 pixels of the bottom of the image
				filter_threshold = 100
				col_idx = np.array(col_idx)
				col_idx = col_idx[np.where(col_idx > 800)]
				col_idx = col_idx[np.where(col_idx < (len(sum)-filter_threshold))]

				try:
					top = col_idx[0] 
					bottom = col_idx[-1] 

					crop_im = result[top:bottom, 0:600, :]

					# resize image to 600 x 298
					resize_img = cv2.resize(crop_im, dsize=(600,298), interpolation=cv2.INTER_CUBIC)
					cv2.imshow("Cropped", resize_img)
					cv2.waitKey(1)

					pos = [(42, 75, 147, 240), 
							(148, 75, 253, 240),
							(345, 75, 450, 240),
							(445, 75, 550, 240)]

					# create mapping for reversing one-hot encoding
					unencoding = {}
					alphanum = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
					for i in range(36):
						unencoding[i] = alphanum[i]

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
								prediction = self.plate_NN.predict(normalized_im) # array of probabilities

							# array of the locations of probabilities sorted from highest to lowest
							# note: the location is the predicted label
							# https://stackoverflow.com/questions/27473601/next-argmax-values-in-python/27473888
							ordered_prediction = np.argsort(-prediction, axis=1) 
							
							# limiting the first 2 characters to letters and the last 2 to numbers
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
					
					self.predict = preprocess_input(resize_img)
					if (self.predict != None):
						self.list_of_plates.append(self.predict)
						print("Prediction:", self.predict)

				except:
					pass
				
		if (self.predict == None):
			self.accumulative_counter +=1
			print("Accumulative counter:" + str(self.accumulative_counter))
	
			if self.accumulative_counter > 300 and self.got_3 == False:
				self.list_of_plates = self.list_of_plates[-10:]
				most_frequent = max(self.list_of_plates, key = self.list_of_plates.count)
				self.list_of_plates = []
				self.sendPlates(most_frequent, 3)
				self.got_3 = True

			elif self.accumulative_counter > 900 and self.got_5 == False:
				self.list_of_plates = self.list_of_plates[-10:]
				most_frequent = max(self.list_of_plates, key = self.list_of_plates.count)
				self.list_of_plates = []
				self.sendPlates(most_frequent, 5)
				self.got_5 = True

			elif self.accumulative_counter > 1450 and self.got_1 == False:
				self.list_of_plates = self.list_of_plates[-10:]
				most_frequent = max(self.list_of_plates, key = self.list_of_plates.count)
				self.list_of_plates = []
				self.sendPlates(most_frequent, 1)
				self.got_1 = True
			
			elif self.accumulative_counter > 1700 and self.got_2 == False:
				self.list_of_plates = self.list_of_plates[-10:]
				most_frequent = max(self.list_of_plates, key = self.list_of_plates.count)
				self.list_of_plates = []
				self.sendPlates(most_frequent, 2)
				self.got_2 = True

	def sendPlates(self, licencePlate, parkingID):
		self.plate_pub.publish(str(self.team_id + "," + self.team_pswd + "," + str(parkingID) + "," + 
                                  str(licencePlate)))


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
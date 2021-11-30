#! /usr/bin/env python
# from _typeshed import Self 
from logging import FATAL, currentframe, fatal
import sys
import rospy
import cv2 
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from skimage import measure
from std_msgs.msg import String
from rosgraph_msgs.msg import Clock
import math
import time
import serial


class robot_navigation:

    def __init__(self):
        self.nav_pub = rospy.Publisher("/R1/cmd_vel",Twist, queue_size=1)
        self.bridge = CvBridge()
        self.nav_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback, queue_size = 1)
        # self.red_line_cam = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback)
        self.plate_pub = rospy.Publisher("/license_plate", String, queue_size=1)

        self.P = 9.25 #previously was 7
        self.linear_speed = 0.1

        self.team_id = "Vroom"
        self.team_pswd = "vroom24"
        self.number_turns = 0
        self.prevError = 0
        self.startTime = 0
        self.count_ped = 0
        self.no_ped = 0
        self.stop_counter = 0
        self.timer_counter = 0
        self.motion_boolean = False
        self.number_red_lines = 0
        self.angular_increase = 0
        self.sequence_ran = False
      
        self.crosswalk = 0


        self.last_frame = np.zeros((20, 20, 3), np.uint8) #just a placeholder to initiliaze this variable
        self.diff_threshold = cv2.cvtColor(np.zeros((359, 400, 3), np.uint8), cv2.COLOR_BGR2GRAY)
        self.max_threshold = cv2.cvtColor(np.zeros((359, 400, 3), np.uint8), cv2.COLOR_BGR2GRAY)
        time.sleep(1) 

    def callback(self,data):
        try:
          img = self.bridge.imgmsg_to_cv2(data, "bgr8")
          cv2.imshow("image feed", img)
          cv2.waitKey(2)
          oneThird = img[360:-1, 445:835] #for all the functions just pass in this

          blurred = cv2.GaussianBlur(oneThird, (7, 7), 0)
          lower_hsv = np.array([0,0,80])
          upper_hsv = np.array([255,255,88])
          mask = cv2.inRange(blurred, lower_hsv, upper_hsv)
          # cv2.imshow("gaussian blur", blurred)
          # cv2.waitKey(1)
          move = Twist()

          if (self.motion_boolean == False):
            self.startTime = rospy.get_time()
            print("time elapsed " + str(rospy.get_time() - self.startTime))
            print("this do be the starting time " + str(self.startTime))
            self.motion_boolean = True
          else:
            # print("time elapsed " + str(rospy.get_time() - self.startTime))
            if ((rospy.get_time() - self.startTime) < 1.8):
              print("moving straight")
              move.linear.x = 0.2
              move.angular.z = 0
            elif (((rospy.get_time() - self.startTime) >= 1.8) and ((rospy.get_time() - self.startTime) < 3.2)):
              print("turning")
              move.linear.x = 0
              move.angular.z = 0.6
            else:
              binary = cv2.bitwise_not(mask) # OR
              binary = cv2.bitwise_not(binary) 
              (rows, cols) = binary.shape
              centreRobot = cols/2 #need to keep this line here
              
              M = cv2.moments(binary)
              # calculate x,y coordinate of center
              cX = int(M["m10"] / M["m00"])
            
              centreIndex = cX
              centreCoords = (int(centreIndex), 200)
              color = [0, 0, 0]
              frame_ball= cv2.circle(binary, centreCoords, 20, color, -1)
              
              cv2.imshow("circle", frame_ball)
              cv2.waitKey(1)

              if((self.find_red_line(img) and self.number_red_lines == 0)):
                move.linear.x = 0
                move.angular.z = 0
                self.nav_pub.publish(move)

                rospy.sleep(0.7)
                # if (self.angular_increase < 2):
                #   move.angular.z = 0.4
                #   self.angular_increase += 1
                # else:
                #   move.angular.z = 0
                # self.nav_pub.publish(move)
                # rospy.sleep(0.5)

                if (self.detect_pedestrain(oneThird)):
                  self.count_ped += 1
                  self.no_ped = 0
                  if (self.count_ped > 2):
                    print("TIME TOOOOO MOVEEEEEE")
                    self.crosswalk = 1 
                    self.number_red_lines = 1
                  print("PEDESTRIAN")
                else:
                  self.no_ped += 1
                  print("no ped count " + str(self.no_ped)) 
                # self.detect_pedestrain(oneThird)
                if (self.no_ped > 1):
                  print("TIME TOOOOO MOVEEEEEE")
                  self.crosswalk = 1 
                  self.number_red_lines = 1
              else:
                print("inside PID")
                self.count_ped = 0
                self.no_ped = 0
                self.number_red_lines = 0
                self.angular_increase = 0

                move.linear.x = self.linear_speed
                currError = centreRobot - centreIndex
                P = (float) (currError * (self.P )) / cols
                
                move.angular.z = P
                if (move.angular.z > 0.5):
                  move.angular.z = 0.5
                elif (move.angular.z < -0.5):
                  move.angular.z = -0.5
                self.prevError = currError 


            if (self.crosswalk == 1):
              move.linear.x = 0.6
              move.angular.z = 0.1
              self.nav_pub.publish(move)
              self.number_red_lines = 1
              rospy.sleep(0.55)
              move.linear.x = 0.4
              move.angular.z = 0.1
              self.nav_pub.publish(move)
              rospy.sleep(0.2)
              move.linear.x = 0.2
              move.angular.z = 0.3
              self.nav_pub.publish(move)
              self.crosswalk = 0
              rospy.sleep(0.1)
              # move.linear.x = 0.3
              # move.angular.z = 0.7
              # self.nav_pub.publish(move)
              self.crosswalk = 0
              self.number_red_lines = 1
              img = self.bridge.imgmsg_to_cv2(data, "bgr8")
              rospy.sleep(1.2)

            else:
              self.nav_pub.publish(move) 
              self.last_frame = oneThird            
          
        except CvBridgeError as e:
            print(e)  
    


    def find_red_line(self, img):
      
      hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
      oneThird = hsv[360:-1, 400:880]

      blurred = cv2.GaussianBlur(oneThird, (7, 7), 0)
      lower_hsv = np.array([0,41,0])
      upper_hsv = np.array([13,255,255])
      mask = cv2.inRange(blurred, lower_hsv, upper_hsv)
      binary = cv2.bitwise_not(mask) # OR
      binary = cv2.bitwise_not(mask)

      size = int(binary.shape[0] / 1.2)
      croppedBin = binary[size: -1, :]
      cv2.imshow("binary cropped for red line ", croppedBin)
      cv2.waitKey(1)
      b = np.nonzero(croppedBin == 0)[0]
      
      if (b.size > 120):
        print(b.size)
        print("found a red line")
        return True
      return False


    def detect_pedestrain(self, cropped):     
      pre1 = self.preprocess_image(cropped)
      pre2 = self.preprocess_image(self.last_frame)
      diff = cv2.absdiff(pre1, pre2)

      _, thresholded = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
      median = cv2.medianBlur(thresholded, 5)
      kernel = np.ones((6, 6), np.uint8)
      image = cv2.erode(median, kernel) 
      #erode it again to get rid of the tiny blobs as well
      # imageFinal = cv2.erode(image, kernel) 
      # cv2.imshow("eroded", image)
      # cv2.waitKey(1)

      nonZeroEntries = 0
      listNonZero =  cv2.findNonZero(image, nonZeroEntries)
      if (listNonZero is not None):
        nonZeroEntries = len(listNonZero)
        # print(str(nonZeroEntries) + " non zero pixels")

        # nonZeroEntries < 1000
      if (nonZeroEntries > 100):
        print("pedestrian detected")
        return True
      return False


    def preprocess_image(self, image):
      bilateral_filtered_image = cv2.bilateralFilter(image, 7, 150, 150)
      gray_image = cv2.cvtColor(bilateral_filtered_image, cv2.COLOR_BGR2GRAY)
      return gray_image
    

    def detect_turns(self, data, centreIndex):
      img = self.bridge.imgmsg_to_cv2(data, "bgr8")
      #cols start at 360 of the original image so crop less
      horizontalCrop = img[:, 400:880]

      blurred = cv2.GaussianBlur(horizontalCrop, (7, 7), 0)
      lower_hsv = np.array([0,0,80])
      upper_hsv = np.array([255,255,88])
      mask = cv2.inRange(blurred, lower_hsv, upper_hsv)

      binary = cv2.bitwise_not(mask)

      if (np.abs(self.findLineMiddle(binary) - centreIndex) > 300):
        return True
      
      return False
    

    def start_timer(self):
      self.plate_pub.publish(str(self.team_id + "," + self.team_pswd + "," + "0" + "," + 
                                  "XR58"))
    
    def end_timer(self):
      self.plate_pub.publish(str(self.team_id + "," + self.team_pswd + "," + "-1" + "," + 
                                  "XR58"))

def main(args):
  rospy.init_node('robot_navigation', anonymous=True)
  lc = robot_navigation()
  lc.start_timer()

  initialTime = rospy.get_time()
  print(initialTime)

  while(True):
      if (rospy.get_time() - initialTime > 240):
          lc.end_timer()
          print("4Mins timer ended")
          break


  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)

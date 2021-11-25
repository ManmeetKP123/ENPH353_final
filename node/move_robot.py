#! /usr/bin/env python
import sys
import rospy
import cv2
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import matplotlib.pyplot as plt


class robot_navigation:

    def __init__(self):
        self.nav_pub = rospy.Publisher("/cmd_vel",Twist, queue_size=1)
        self.bridge = CvBridge()
        self.nav_sub = rospy.Subscriber("/rrbot/camera1/image_raw",Image,self.callback)
        self.P = 4
        self.linear_speed = 0.5

    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            (rows, cols, channels) = cv_image.shape

            centreRobot = cols/2;
            gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)

            bottom = cv_image[-1, :, 0];
            threshold = int((np.amax(bottom) + np.amin(bottom))/2);
            #print(threshold)
            array = [];

            move = Twist()
            move.linear.x = 0.25
      #move.angular.z = 0.5

            for x in cv_image[-1, :, 0]:
                if(x < threshold):
                    array.append(0); 
                else:
                    array.append(1); 

            centreArr = np.array(array);
            line = np.where(centreArr == 0); 
            centreLine = np.array(line); 
            centreLine = centreLine[0, :];
        
            centreIndex = 0;

            if len(centreLine) != 0:
                centreIndex += (centreLine[0] + centreLine[-1]) // 2; 

            move = Twist()

            move.linear.x = self.linear_speed
            move.angular.z = (float)(centreRobot - centreIndex)/ cols * self.P

            self.nav_pub.publish(move)
                            
       
      #print(centreRobot)
      #print(move.angular.z);      
        except CvBridgeError as e:
            print(e)

        #(rowcv2.imshow("Image window", cv_image)
        #self.nav_pub.publish(move)s,cols,channels) = cv_image.shape

def main(args):
  lc = robot_navigation()
  rospy.init_node('robot_navigation', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)

# Autonomous Parking Violation Detector
This repo was for our final project for ENPH 353, where we used CNN and PID to program a self-driving car that collects parking IDs of the parked
vehicles.
The file structure of the project is as follows: *talk about file structure*
The general overview is as follows: *talk about a very very vague / general idea*

Our general strategy for the competition incorporated the following elements:
- Using HSV thresholding for road detection and finding the centroid of the road
- Using PID for driving and controlling the robot’s motion around the outer and inner loop 
- Detecting movement and the presence of a pedestrian through HSV thresholding and change in pixels in subsequent frames
- Extracting license plates from parked cars through HSV thresholding, contour detection, and perspective transform 
- Reading license plates from extracted images using a CNN
- Generating data points and applying Gaussian blur to better represent simulation conditions 
- Taking data points (pictures of the license plates) from the simulation to train the CNN model 
- Testing the CNN model in the simulation and fixing any accuracy issues by better thresholding the parked cars or feeding the model more data points 

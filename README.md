# Autonomous Parking Violation Detector
This repo was our final project for ENPH 353. The goal was to program a self-driving car to collect the parking IDs of parked vehicles. We did this using PID and a convolutional neural network (CNN).

The file structure of our project is as follows: *talk about file structure*
The general overview is as follows: *talk about a very very vague / general idea*

## General Strategy
### Driving
- Use HSV thresholding for road detection and finding the centroid of the road
- Use PID for driving and controlling the robot’s motion around the outer and inner loop 
- Detect movement and the presence of a pedestrian through HSV thresholding and the change in pixels in subsequent frames

### License Plate Detection
- Extract license plates from parked cars through HSV thresholding, contour detection, and perspective transform 
- Read license plates from extracted images using a CNN

### CNN Model Training
- Generate data points (license plates) and apply Gaussian blur before training to better represent simulation conditions 
- Gather data points from the simulation 
- Improve model accuracy by narrowing down the characters the model struggled with within simulation and feeding the model more data points that include the determined characters

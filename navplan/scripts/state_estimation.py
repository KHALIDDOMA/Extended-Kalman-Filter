#!/usr/bin/env python3
from ast import Sub
import rospy
import numpy as np
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import MultiArrayLayout
from std_msgs.msg import MultiArrayDimension

rospy.init_node("state_estimation")

gps_observation = np.array([0.0,0.0,0.0]) # x, y, theta
steering = 0
velocity = 0

# Subscribe to the gps, steering, and velocity topics named below and update the global variables using callbacks
# /gps
# /car_actions/steer
# /car_actions/vel

def vel_clbck(msg):
    global gps_observation, steering, velocity
    gps_observation = rospy.wait_for_message("/gps", Float64MultiArray)
    steering = rospy.wait_for_message("/car_actions/steer", Float64)
    velocity = msg
    

vel_sub = rospy.Subscriber("/car_actions/vel", Float64, callback=vel_clbck)

# Publisher for the state
state_pub = rospy.Publisher('vehicle_model/state', Float64MultiArray, queue_size=10)

r = rospy.Rate(10)

# Initialize the start values and matrices here

delta_t = 0.1

jacobian = np.array([[1.0,   0, -1 * velocity * delta_t * np.sin(steering)],
                     [  0, 1.0, velocity * delta_t * np.cos(steering)     ],
                     [  0,   0, 1.0                                       ]])

H = np.array([[1.0,  0,   0],
              [  0,1.0,   0],
              [  0,  0, 1.0]])

estimated_uncertainty = np.diag([0.1]*3)

car_state = np.array([0,0,0])

while not rospy.is_shutdown():
    # Create the Kalman Filter here to estimate the vehicle's x, y, and theta

    predicted_state = np.array([car_state[0] + velocity * np.cos(car_state[2]) * delta_t,
                                car_state[1] + velocity * np.sin(car_state[2]) * delta_t,
                                car_state[2] + (velocity * np.tan(steering)/ 4.9) * delta_t])

    predicted_uncertainty = jacobian @ estimated_uncertainty @ jacobian.T
    kalman_gain = predicted_uncertainty @ H.T @ np.linalg.pinv(H @ predicted_uncertainty @ H.T)
    car_state = predicted_state + kalman_gain @ (gps_observation - predicted_state)
    estimated_uncertainty = (np.eye(3) - kalman_gain @ H) @ predicted_uncertainty



    # Create msg to publish#
    current_state = Float64MultiArray()
    layout = MultiArrayLayout()
    dimension = MultiArrayDimension()
    dimension.label = "current_state"
    dimension.size = 3
    dimension.stride = 3
    layout.data_offset = 0
    layout.dim = [dimension]
    current_state.layout = layout
    current_state.data = car_state

    state_pub.publish(current_state)
    r.sleep()
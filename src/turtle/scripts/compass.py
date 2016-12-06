#!/usr/bin/env python

import rospy
import tf
import math
import numpy
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float64
from tf.transformations import quaternion_from_euler

def qv_mult(q1, v1):
    q1 = (q1.x, q1.y, q1.z, q1.w)
    v1 = tf.transformations.unit_vector(v1)
    q2 = list(v1)
    q2.append(0.0)
    return tf.transformations.quaternion_multiply(
        tf.transformations.quaternion_multiply(q1, q2), 
        tf.transformations.quaternion_conjugate(q1)
    )[:3]

northVector = numpy.array([1, 0, 0])

def getCompassValue(robotVector, northVector):
    # If pointing straight up or down, make down vector equal new forward vector
        if robotVector[0] == 0 and robotVector[1] == 0:
            robotVector[0], robotVector[1], robotVector[2] = -robotVector[2], -robotVector[1], robotVector[0]

        # Set z = 0
        robotVector[2] = 0

        # Get 2D angle difference
        signedAngle = math.atan2(robotVector[1], robotVector[0]) - math.atan2(northVector[1], northVector[0])

        return signedAngle

def positionUpdateCallback(msg):
    if "turtle" in msg.name:
        orientation = msg.pose[msg.name.index("turtle")].orientation
        robotVector = qv_mult(orientation, (1, 0, 0))

        compassPub.publish(getCompassValue(robotVector, northVector))
        
positionSub = rospy.Subscriber("/gazebo/model_states", ModelStates, positionUpdateCallback)
compassPub  = rospy.Publisher("compass", Float64, positionUpdateCallback)
rospy.init_node('fitness_evaluator')

rospy.spin()
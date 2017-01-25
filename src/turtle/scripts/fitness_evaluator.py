#!/usr/bin/env python

import rospy
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float64

targetPosition = (-3, -3, 0)
gI = 0

def positionUpdateCallback(msg):
    global gI
    if "turtle" in msg.name:
        pos = msg.pose[msg.name.index("turtle")].position
        distance = ((pos.x-targetPosition[0])**2 + (pos.y-targetPosition[1])**2 + (pos.z-targetPosition[2])**2) ** 0.5
        fitnessPub.publish(1 / distance)
        if gI % 10000 == 0:
            print("Distance: " + str(distance))
        gI = gI + 1

positionSub = rospy.Subscriber("/gazebo/model_states", ModelStates, positionUpdateCallback)
fitnessPub  = rospy.Publisher("fitness_evaluator", Float64, queue_size=10)
rospy.init_node('fitness_evaluator')

rospy.spin()

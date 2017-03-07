#!/usr/bin/env python

import rospy
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float64

def evaluate_fitness(pos):
    targetPosition = (5, 0, 0)
    distance = ((pos.x-targetPosition[0])**2 + (pos.y-targetPosition[1])**2 + (pos.z-targetPosition[2])**2) ** 0.5
    fitnessPub.publish(1.0 / distance)

def positionUpdateCallback(msg):
    if "turtle" in msg.name:
        pos = msg.pose[msg.name.index("turtle")].position
        return evaluate_fitness(pos)

positionSub = rospy.Subscriber("/gazebo/model_states", ModelStates, positionUpdateCallback)
fitnessPub  = rospy.Publisher("fitness_evaluator", Float64, queue_size=0)
rospy.init_node('fitness_evaluator')

rospy.spin()

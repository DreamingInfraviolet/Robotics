#!/usr/bin/env python

import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Vector3

def callback(msg):
    if "turtle" in msg.name:
        pub.publish(msg.pose[msg.name.index("turtle")].position)

sub = rospy.Subscriber("/gazebo/model_states", ModelStates, callback)
pub  = rospy.Publisher("gps", Vector3, queue_size=0)
rospy.init_node('gps')

rospy.spin()

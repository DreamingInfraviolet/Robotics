#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import math

def scanCallback(msg):
    global gRangeAhead
    gRangeAhead = min(msg.ranges)
    if math.isnan(gRangeAhead):
        gRangeAhead = 100

gRangeAhead = 1

cmd_vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
scanSub = rospy.Subscriber("scan", LaserScan, scanCallback)
rospy.init_node('red_light_green_light')

red_light_twist = Twist()
green_light_twist = Twist()
green_light_twist.linear.x = 0.5

rate = rospy.Rate(10)

while not rospy.is_shutdown():
    if gRangeAhead > 0.8:
        print("Moving")
        cmd_vel_pub.publish(green_light_twist)
    else:
        print("Too close: " + str(gRangeAhead))
        cmd_vel_pub.publish(red_light_twist)
    rate.sleep()

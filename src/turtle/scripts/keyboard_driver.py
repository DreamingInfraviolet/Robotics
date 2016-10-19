#!/usr/bin/env python

import sys
import select
import tty
import termios

import rospy
from geometry_msgs.msg import Twist
keyMapping = {
    "w": [1, 0],
    "s": [-1, 0],
    "a": [0, 1],
    "d": [0, -1],
    "z": [0, 0],
}

def lerp(a, b, t):
    return a*(1-t) + b*t

def main():
    keyPublisher = rospy.Publisher("keys", Twist, queue_size=1)
    rospy.init_node("keyboard_driver")

    rate = rospy.Rate(50)

    oldAttr = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    currentVector = [0.0, 0.0]

    while not rospy.is_shutdown():
        
        movementHappened = False
        if select.select([sys.stdin], [], [], 0)[0] == [sys.stdin]:
            key = sys.stdin.read(1)
            print("Got key")
            mapping = keyMapping.get(key)
            if mapping:
                print("Found mapping")
                currentVector[0] = lerp(currentVector[0], mapping[0], 0.8)
                currentVector[1] = lerp(currentVector[1], mapping[1], 0.8)
                movementHappened = True

        if not movementHappened:
            currentVector[0] = lerp(currentVector[0], 0, 0.1)
            currentVector[1] = lerp(currentVector[1], 0, 0.1)
            epsilon = 0.001
            if abs(currentVector[0]) < epsilon:
                currentVector[0] = 0
            if abs(currentVector[1]) < epsilon:
                currentVector[1] = 0

        msg = Twist()
        print(currentVector)
        msg.linear.x = currentVector[0]
        msg.angular.z = currentVector[1]
        keyPublisher.publish(msg)

        rate.sleep()

    termios.tcsetattr(oldAttr)

if __name__=="__main__":
    main()
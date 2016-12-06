#!/usr/bin/env python

import rospy
import sys
from std_msgs.msg import Bool
from gazebo_msgs.msg import ModelState
from gazebo_msgs.msg import ModelStates
import gazebo_ros.gazebo_interface

initialState = None

def resetPositionCallback(msg):
    if msg and initialState:
        print("Resetting model state")
        positionPub.publish(initialState)

def getInitialState():
    global initialState
    rospy.wait_for_service("/gazebo/get_model_state")

    try:
        serv = rospy.ServiceProxy('/gazebo/get_model_state', gazebo_ros.gazebo_interface.GetModelState)
        response = serv(model_name="turtle", relative_entity_name="world")
        if not response.success:
            raise RuntimeError("Failed to get initial state")
        pose = response.pose

        initialState = ModelState()
        initialState.model_name = "turtle"
        initialState.reference_frame = "world"
        initialState.pose = pose

    except rospy.ServiceException, e:
        print("Service call failed: %s" + str(e))
        return False

    return True

if not getInitialState():
    sys.exit(1)

resetSub     = rospy.Subscriber("reset_position", Bool, resetPositionCallback)
positionPub  = rospy.Publisher("/gazebo/set_model_state", ModelState)
rospy.init_node('reset_position')
rospy.spin()
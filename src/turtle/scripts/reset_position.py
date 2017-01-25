#!/usr/bin/env python

import rospy
import sys
import time
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
    retries = 20
    
    try:
        serv = rospy.ServiceProxy('/gazebo/get_model_state', gazebo_ros.gazebo_interface.GetModelState)

        # We need to make a number of attempts in case the robot is not immediately spawned
        for iRetry in range(retries):
            response = serv(model_name="turtle", relative_entity_name="world")
            if not response.success:
                print("Could not find robot. Retrying . . . .")
                time.sleep(0.1)
                continue
            pose = response.pose

            initialState = ModelState()
            initialState.model_name = "turtle"
            initialState.reference_frame = "world"
            initialState.pose = pose

            return True

        # If we are here, we ran out of retries.
        raise RuntimeError("reset_position: Could not find robot")

    except rospy.ServiceException, e:
        print("Service call failed: %s" + str(e))
        return False

    assert False

if not getInitialState():
    sys.exit(1)

resetSub     = rospy.Subscriber("reset_position", Bool, resetPositionCallback)
positionPub  = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)
rospy.init_node('reset_position')
rospy.spin()
#!/usr/bin/env python


import math
import time
import rospy
import sys
from std_msgs.msg import Float64
from std_msgs.msg import Bool
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Vector3
from gazebo_msgs.msg import ModelStates


import tf
import numpy
from gazebo_msgs.msg import ModelStates
from tf.transformations import quaternion_from_euler
import gazebo_ros.gazebo_interface

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


def evaluate_fitness(pos):
    targetPosition = (15, 0, 0)
    distance = ((pos.x-targetPosition[0])**2 + (pos.y-targetPosition[1])**2 + (pos.z-targetPosition[2])**2) ** 0.5
    return 1.0 / distance









class RosInput(object):
    def __init__(self, topic, msgtype, timeout=None, normalisation=None):
        self.sub = rospy.Subscriber(topic, msgtype, self._update)
        self.topic = topic
        self.msgtype = msgtype
        self.values = [msgtype()]
        self.timeout = timeout
        self.normalisation=normalisation

    def _update(self, msg):
        '''Callback to update value'''
        self.values = [msg.data]
        if self.normalisation != None:
            self._normalise()

    def _normalise(self):
        self.values = [min(x, self.normalisation) for x in self.values]
        self.values = [max(x, -self.normalisation) for x in self.values]
        self.values = [x/float(self.normalisation) for x in self.values]
        
    def fetchValues(self):
        '''Wait for a compass message, and return its value'''
        try:
            rospy.wait_for_message(self.topic, self.msgtype, self.timeout)
        except rospy.exceptions.ROSException:
            pass
        return self.values

class LaserInput(RosInput):
    def __init__(self, topic, timeout=None, normalisation=None):
        super(LaserInput, self).__init__(topic, LaserScan, timeout, normalisation)

    def _update(self, msg):
        self.values = [(1000 if x == float("Inf") else x) for x in list(msg.ranges)]
        if self.normalisation:
            self._normalise()

class VectorInput(RosInput):
    def __init__(self, topic, timeout=None, normalisation=None):
        super(VectorInput, self).__init__(topic, Vector3, timeout, normalisation)

    def _update(self, msg):
        self.values = [msg.x, msg.y, msg.z]
        if self.normalisation:
            self._normalise()

def fetchRobotModelPose(serv):
    retries = 20
    try:
        # We need to make a number of attempts in case the robot is not immediately spawned
        for iRetry in range(retries):
            response = serv(model_name="turtle", relative_entity_name="world")
            if not response.success:
                print("Could not find robot. Retrying . . . .")
                time.sleep(0.1)
                continue
            return response.pose
        # If we are here, we ran out of retries.
        raise RuntimeError("reset_position: Could not find robot")
    except rospy.ServiceException, e:
        print("Service call failed: %s" + str(e))
        return None
    assert False

class CompassInput(object):
    def __init__(self):
        rospy.wait_for_service("/gazebo/get_model_state")
        self.serv = rospy.ServiceProxy('/gazebo/get_model_state', gazebo_ros.gazebo_interface.GetModelState)

    def fetchValues(self):
        pose = fetchRobotModelPose(self.serv)
        robotVector = qv_mult(pose.orientation, (1, 0, 0))
        compassValue = getCompassValue(robotVector, northVector)  / math.pi
        # print("compass: " + str(compassValue))
        return [compassValue]
            
class GpsInput(object):
    def __init__(self):
        self.scale = 1.0/20.0
        rospy.wait_for_service("/gazebo/get_model_state")
        self.serv = rospy.ServiceProxy('/gazebo/get_model_state', gazebo_ros.gazebo_interface.GetModelState)
        self.values=[0,0,0]

    def fetchValues(self):
        pose = fetchRobotModelPose(self.serv)
        self.values = [pose.position.x / self.scale, pose.position.y / self.scale, pose.position.z / self.scale]
        # print("gps: " + str(self.values))
        return self.values
            
class FitnessInput(object):
    def __init__(self):
        rospy.wait_for_service("/gazebo/get_model_state")
        self.serv = rospy.ServiceProxy('/gazebo/get_model_state', gazebo_ros.gazebo_interface.GetModelState)
        self.values=[0,0,0]

    def fetchValues(self):
        return [evaluate_fitness(fetchRobotModelPose(self.serv).position)]

class MLController(object):
    def __init__(self):
        self.fitness = 0
        self.actions=[]
        self.inputs=[]
        self.inputCount = -1

    def _finaliseInputSetup(self):
        inputs = self.fetchInputs()
        self.inputCount = len(inputs)

    def getActionCount(self):
        return len(self.actions)

    def performActions(self, n):
        self.actions[n](self, 0.5)

    def getInputCount(self):
        return self.inputCount

    def fetchInputs(self):
        out = []
        for x in self.inputs:
            out = out + x.fetchValues()

        return out

    def fetchFitness(self):
        pass

class TurtlebotMLController(MLController):

    class ResettableVariables(object):
        def __init__(self):
            self.turnDelta = math.pi / 5.0
            self.movementVelocity = 10.0
            self.currentWheelOrientation = 0

    def __init__(self):
        rospy.init_node("ml_controller")
    
        super(TurtlebotMLController, self).__init__()

        self.vars = self.ResettableVariables()

        self.lwPub      = rospy.Publisher("/left_wheel_controller/command", Float64, queue_size=10)
        self.rwPub      = rospy.Publisher("/right_wheel_controller/command", Float64, queue_size=10)
        self.fwPub      = rospy.Publisher("/front_wheel_controller/command", Float64, queue_size=10)
        self.turnPub    = rospy.Publisher("/front_caster_controller/command", Float64, queue_size=10)
        self.resetPub   = rospy.Publisher("/reset_position", Bool, queue_size=10)

        self.gpsInput = GpsInput()
        self.fitnessInput = FitnessInput()

        self.inputs = [
            CompassInput(),
            RosInput("/front_caster_controller/command", Float64, 0.05, normalisation=(math.pi*2)),
            LaserInput("/laser_scan", normalisation=1000.0),
            self.gpsInput,
        ]

        self._finaliseInputSetup()

        self.actions = [
            # Move forward
            lambda instance, n: instance._move(n* instance.vars.movementVelocity),
            # Move backward
            lambda instance, n: instance._move(-n* instance.vars.movementVelocity),
            # Forward left
            lambda instance, n: instance._move_and_turn(n* instance.vars.movementVelocity, -0.5),
            # Forward right
            lambda instance, n: instance._move_and_turn(n* instance.vars.movementVelocity, 0.5),
            # Back left
            lambda instance, n: instance._move_and_turn(-n* instance.vars.movementVelocity, -0.5),
            # Back right
            lambda instance, n: instance._move_and_turn(n* instance.vars.movementVelocity, 0.5),
            # Turn left
            lambda instance, n: instance._move_and_turn(0, -0.5),
            # Turn right
            lambda instance, n: instance._move_and_turn(0, 0.5),
        ]

    def reset(self):
        self.vars = self.ResettableVariables()
        self._move(0)
        self._turn(0)
        self._resetModel()

    def fetchFitness(self):
        return self.fitnessInput.fetchValues()[0]

    def _resetModel(self):
        '''Reset the robot's position and state'''
        #print("Resetting")
        self.resetPub.publish(True)
        
        # Sleeping is slow
        # Instead of sleeping, check until position is reset
        # We first check if it's already at 0,0
        e = 0.1 
        if abs(self.gpsInput.values[0]) < e and abs(self.gpsInput.values[0]) < e:
            pass
        else:
            sys.stdout.write("Waiting until robot is reset")
            countdown = 100
            while countdown > 0:
                sys.stdout.write(".")
                sys.stdout.flush()
                newPos = self.gpsInput.fetchValues()
                if abs(newPos[0]) < e and abs(newPos[0]) < e:
                    sys.stdout.write(" Done!\n")
                    sys.stdout.flush()
                    break
                else:
                    # Rarely reached
                    time.sleep(0.01)
                    countdown = countdown - 1
            if countdown == 0:
                raise EnvironmentError("Robot reset confirmation not received")

    def _move_and_turn(self, speed, angle):
        self._move(speed)
        self._turn(angle)


    def _turn(self, angle):
        '''Set the robot's caster wheel angle'''
        #print("Turning by " + str(angle))
        self.vars.currentWheelOrientation = self.vars.currentWheelOrientation + angle

        if self.vars.currentWheelOrientation < -1.5:
            self.vars.currentWheelOrientation = -1.5
        if self.vars.currentWheelOrientation > 1.5:
            self.vars.currentWheelOrientation = 1.5

        self.turnPub.publish(self.vars.currentWheelOrientation)

    def _move(self, amount):
        ''' Set the robot's forward motion by @amount'''
        self.lwPub.publish(amount)
        self.rwPub.publish(amount)
        self.fwPub.publish(amount)

import math
import time
import rospy
from std_msgs.msg import Float64
from std_msgs.msg import Bool
from sensor_msgs.msg import LaserScan

class RosInput(object):
    def __init__(self, topic, msgtype, timeout=None):
        self.sub = rospy.Subscriber(topic, msgtype, self._update)
        self.topic = topic
        self.msgtype = msgtype
        self.values = [msgtype()]
        self.timeout = timeout

    def _update(self, msg):
        '''Callback to update value'''
        self.values = [msg.data]

    def fetchValues(self):
        '''Wait for a compass message, and return its value'''
        try:
            rospy.wait_for_message(self.topic, self.msgtype, self.timeout)
        except rospy.exceptions.ROSException:
            pass
        return self.values

class LaserInput(RosInput):
    def __init__(self, topic, timeout=None):
        super(LaserInput, self).__init__(topic, LaserScan, timeout)
        self.values=[]

    def _update(self, msg):
        self.values = list(msg.ranges)
        self.values[self.values==float("Inf")] = 1000

class MLController(object):
    def __init__(self):
        self.fitness = 0
        self.actions=[]
        self.inputs=[]
        self.fitnessInput = RosInput("fitness_evaluator", Float64)
        self.inputCount = -1

    def _finaliseInputSetup(self):
        self.inputCount = len(self.fetchInputs())

    def _updateFitness(self, msg):
        self.fitness = msg.data

    def getActionCount(self):
        return len(self.actions)

    def performAction(self, n):
        self.actions[n](self)

    def getInputCount(self):
        return self.inputCount

    def fetchInputs(self):
        out = []
        for x in self.inputs:
            out = out + x.fetchValues()

        return out

    def fetchFitness(self):
        return self.fitnessInput.fetchValues()[0]

class TurtlebotMLController(MLController):

    class ResettableVariables(object):
        def __init__(self):
            self.turnDelta = math.pi / 5
            self.movementVelocity = 5.0
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

        self.inputs = [
            RosInput("compass", Float64),
            RosInput("/front_caster_controller/command", Float64, 0.05),
            LaserInput("/laser_scan")
        ]

        self._finaliseInputSetup()

        self.actions = [
            # Move forward
            lambda instance: instance._move(instance.vars.movementVelocity),
            # Move backward
            lambda instance: instance._move(-instance.vars.movementVelocity),
            # Turn right
            lambda instance: instance._turn(instance.vars.turnDelta),
            # Turn left
            lambda instance: instance._turn(-instance.vars.turnDelta),
        ]

    def reset(self):
        self.vars = self.ResettableVariables()
        self._move(0)
        self._turn(0)
        self._resetModel()

    def _resetModel(self):
        '''Reset the robot's position and state'''
        #print("Resetting")
        self.resetPub.publish(True)
        time.sleep(0.05)

    def _turn(self, angle):
        '''Set the robot's caster wheel angle'''
        #print("Turning by " + str(angle))
        newOrientation = self.vars.currentWheelOrientation + angle
        if abs(newOrientation) > 1:
            newOrientation = self.vars.currentWheelOrientation
        self.vars.currentWheelOrientation = newOrientation
        self.turnPub.publish(self.vars.currentWheelOrientation)

    def _move(self, amount):
        ''' Set the robot's forward motion by @amount'''
        #print("Moving by " + str(amount))
        self.lwPub.publish(amount)
        self.rwPub.publish(amount)
        self.fwPub.publish(amount)

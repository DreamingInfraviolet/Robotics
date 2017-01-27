import math
import time
import rospy
from std_msgs.msg import Float64
from std_msgs.msg import Bool

class RosInput(object):
    def __init__(self, topic, msgtype):
        self.sub = rospy.Subscriber(topic, msgtype, self._update)
        self.topic = topic
        self.msgtype = msgtype
        self.value = msgtype()

    def _update(self, msg):
        '''Callback to update value'''
        self.value = msg.data

    def fetchValue(self):
        '''Wait for a compass message, and return its value'''
        rospy.wait_for_message(self.topic, self.msgtype)
        return self.value

class MLController(object):
    def __init__(self):
        self.fitness = 0
        self.actions=[]
        self.inputs=[]
        self.fitnessInput = RosInput("fitness_evaluator", Float64)

    def _updateFitness(self, msg):
        self.fitness = msg.data

    def getActionCount(self):
        return len(self.actions)

    def performAction(self, n):
        self.actions[n](self)

    def getInputCount(self):
        return len(self.inputs)

    def fetchInputs(self):
        return [x.fetchValue() for x in self.inputs]

    def fetchFitness(self):
        return self.fitnessInput.fetchValue()

class TurtlebotMLController(MLController):

    class ResettableVariables(object):
        def __init__(self):
            self.turnDelta = math.pi / 5
            self.movementVelocity = 5.0
            self.currentWheelOrientation = 0

    def __init__(self):
        super(TurtlebotMLController, self).__init__()

        self.vars = self.ResettableVariables()

        self.lwPub      = rospy.Publisher("/left_wheel_controller/command", Float64, queue_size=10)
        self.rwPub      = rospy.Publisher("/right_wheel_controller/command", Float64, queue_size=10)
        self.fwPub      = rospy.Publisher("/front_wheel_controller/command", Float64, queue_size=10)
        self.turnPub    = rospy.Publisher("/front_caster_controller/command", Float64, queue_size=10)
        self.resetPub   = rospy.Publisher("/reset_position", Bool, queue_size=10)

        rospy.init_node("ml_controller")

        self.inputs = [
            RosInput("compass", Float64),
        ]

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
        self.vars.currentWheelOrientation = self.vars.currentWheelOrientation + angle
        self.turnPub.publish(self.vars.currentWheelOrientation)

    def _move(self, amount):
        ''' Set the robot's forward motion by @amount'''
        #print("Moving by " + str(amount))
        self.lwPub.publish(amount)
        self.rwPub.publish(amount)
        self.fwPub.publish(amount)

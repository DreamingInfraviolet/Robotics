import math
import rospy
from std_msgs.msg import Float64
from std_msgs.msg import Bool

class MLController(object):
    def __init__(self):
        pass

    def _updateFitness(self, msg):
        self.fitness = msg.data

    def getActionCount(self):
        return 0

    def performAction(self, n):
        pass

class TurtlebotMLController(MLController):
    def __init__(self, fitness_node="fitness_evaluator", compass_node="compass",
                 leftWheelCmdNode="/left_wheel_controller/command",
                 rightWheelCmdNode="/right_wheel_controller/command",
                 frontWheelCmdNode="/front_caster_controller/command",
                 frontWheelCasterCmdNode="/front_caster_controller/command",
                 resetNode="/reset_position"):
        super(TurtlebotMLController, self)
        self.turnDelta = math.pi / 5
        self.movementVelocity = 5.0
        self.desiredWheelOrientation = 0

        self.fitnessSub = rospy.Subscriber(fitness_node, Float64, self._updateFitness)
        self.compassSub = rospy.Subscriber(compass_node, Float64, self._updateCompass)
        self.lwPub      = rospy.Publisher(leftWheelCmdNode, Float64, queue_size=10)
        self.rwPub      = rospy.Publisher(rightWheelCmdNode, Float64, queue_size=10)
        self.fwPub      = rospy.Publisher(frontWheelCmdNode, Float64, queue_size=10)
        self.turnPub    = rospy.Publisher(frontWheelCasterCmdNode, Float64, queue_size=10)
        self.resetPub   = rospy.Publisher(resetNode, Bool, queue_size=10)

        rospy.init_node("ml_controller")

        self.actions = [
            # Move forward
            lambda instance: instance._move(instance.movementVelocity),
            # Move backward
            lambda instance: instance._move(-instance.movementVelocity),
            # Turn right
            lambda instance: instance._turn(instance.turnDelta),
            # Turn left
            lambda instance: instance._turn(-instance.turnDelta),
        ]

    def getActionCount(self):
        return len(self.actions)

    def performAction(self, n):
        self.actions[n](self)

    def getFitnessFromNode(self):
        '''Get the robot's fitness and return its value'''
        rospy.wait_for_message("fitness_evaluator", Float64)
        return self.fitness

    def getCompass(self):
        '''Wait for a compass message, and return its value'''
        rospy.wait_for_message("compass", Float64)
        return self.compass

    def reset(self):
        self._move(0)
        self._turn(0)
        self._resetModel()


    def _updateCompass(self, msg):
        '''Callback to update compass'''
        self.compass = msg.data

    def _resetModel(self):
        '''Reset the robot's position and state'''
        print("Resetting")
        self.resetPub.publish(True)

    def _turn(self, angle):
        '''Set the robot's caster wheel angle'''
        print("Turning by " + str(angle))
        self.desiredWheelOrientation = self.desiredWheelOrientation + angle
        self.turnPub.publish(self.desiredWheelOrientation)

    def _move(self, amount):
        ''' Set the robot's forward motion by @amount'''
        print("Moving by " + str(amount))
        self.lwPub.publish(amount)
        self.rwPub.publish(amount)
        self.fwPub.publish(amount)
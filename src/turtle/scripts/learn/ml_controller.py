import math
import time
import rospy
import sys
from std_msgs.msg import Float64
from std_msgs.msg import Bool
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Vector3

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
            
class MLController(object):
    def __init__(self):
        self.fitness = 0
        self.actions=[]
        self.inputs=[]
        self.fitnessInput = RosInput("fitness_evaluator", Float64)
        self.inputCount = -1

    def _finaliseInputSetup(self):
        inputs = self.fetchInputs()
        self.inputCount = len(inputs)

    def _updateFitness(self, msg):
        self.fitness = msg.data

    def getActionCount(self):
        return len(self.actions)

    def performActions(self, n):
        for i in range(len(n)):
            self.actions[i](self, n[i])

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
            self.turnDelta = math.pi / 5.0
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

        self.gpsInput = VectorInput("/gps", normalisation=1000.0)

        self.inputs = [
            RosInput("compass", Float64, normalisation=(math.pi*2)),
            # RosInput("/front_caster_controller/command", Float64, 0.05, normalisation=(math.pi*2)),
            LaserInput("/laser_scan", normalisation=1000.0),
            self.gpsInput,
        ]

        self._finaliseInputSetup()

        self.actions = [
            # Move forward
            lambda instance, n: instance._move(n* instance.vars.movementVelocity),
            # Move backward
            # lambda instance, n: instance._move(-n* instance.vars.movementVelocity),
            # Turn right
            lambda instance, n: instance._turn(n),
            # Turn left
            # lambda instance, n: instance._turn(-n),
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

    def _turn(self, angle):
        '''Set the robot's caster wheel angle'''
        #print("Turning by " + str(angle))
        self.vars.currentWheelOrientation = angle
        self.turnPub.publish(self.vars.currentWheelOrientation)

    def _move(self, amount):
        ''' Set the robot's forward motion by @amount'''
        #print("Moving by " + str(amount))
        self.lwPub.publish(amount)
        self.rwPub.publish(amount)
        self.fwPub.publish(amount)

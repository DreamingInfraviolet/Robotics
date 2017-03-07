import time
import rospy

class MLProgressRewardAlgorithm(object):
    ''' This class must be updated continuously with the robot's fitness, and returns an appropriate reward when asked.
        The reason we need this is that we may want to award +1 when progress was made, or -1 when fitness worsened. '''
    def __init__(self, improvementRewardThreshold=1.018):
        self.startingFitness = 0
        self.bestFitness = 0
        self.lastFitness = 0
        self.currentReward = 0
        self.improvementRewardThreshold = improvementRewardThreshold
        self.startAlreadyInitialised = False

    def updateFitness(self, score):
        if score > self.bestFitness * self.improvementRewardThreshold:
            self.currentReward = 1
            self.bestFitness = score
        elif score < self.startingFitness:
            self.currentReward = -1
        else:
            self.currentReward = 0

        self.lastFitness = score

    def getRewardValue(self):
        return self.currentReward

    def resetForEpoch(self, startingFitness):
        if not self.startAlreadyInitialised:
            self.startingFitness = startingFitness
            self.startAlreadyInitialised = True
        self.bestFitness = self.startingFitness
        self.lastFitness = self.startingFitness
        self.currentReward = 0

class MLDirectFitnessRewardAlgorithm(object):
    ''' This class must be updated continuously with the robot's fitness, and returns an appropriate reward when asked.
        The reason we need this is that we may want to award +1 when progress was made, or -1 when fitness worsened. '''
    def __init__(self):
        self.fitness = 0

    def updateFitness(self, score):
        self.fitness = score

    def getRewardValue(self):
        return self.fitness

    def resetForEpoch(self, startingFitness):
        self.fitness = 0


class MLSimulationDriver(object):
    ''' This class is meant to abstract away the ros-specific code needed to train an algorithm.
        It relies on a ML Controller of choice to control the robot. The main idea is that the
        inputs/outputs and the overall process structure is abstracted away. '''

    def __init__(self, controller, algorithm, rewardAlgorithm, epochs=100, decisionsPerEpoch=20, decisionSimTimeRateHz=1):
        self.controller = controller
        self.algorithm = algorithm
        self.epochs = epochs
        self.decisionsPerEpoch = decisionsPerEpoch
        self.decisionSimTimeRateHz = decisionSimTimeRateHz
        self.rewardAlgorithm = rewardAlgorithm

    def learn(self, skipStartLearningCall = False):
        
        if not skipStartLearningCall:
            self.algorithm.startLearning(self.controller.getInputCount(), self.controller.getActionCount())

        totalLearningReward = 0
        for iEpoch in range(self.epochs):

            totalEpochReward = 0
            totalEpochLearningLoss = 0
            startTime = int(round(time.time() * 1000))
            decisionRate = rospy.Rate(self.decisionSimTimeRateHz)
            self.controller.reset()
            self.rewardAlgorithm.resetForEpoch(self.controller.fetchFitness())

        
            # ///
            self.algorithm.startEpoch()
            
            for iDecision in range(self.decisionsPerEpoch):
                # Decide on next move using algorithm
                decisions = self.algorithm.decideOnAction(self.controller.fetchInputs())
                # Act out decision
                self.controller.performActions(decisions)

                # Give the decision some time to have effect
                # NOTE: Bottleneck. A lot of time is spent sleeping and not doing anything
                # Might be impossible to solve, and this is essentially waiting for the simulation
                decisionRate.sleep()

                # Get reward
                reward = self.fetchReward()
                totalEpochReward = totalEpochReward + reward

                # Get new inputs
                newInputs = self.controller.fetchInputs()
                # React to its effect
                isFinalDecision = (iEpoch == self.epochs - 1 and iDecision == self.decisionsPerEpoch - 1)
                loss = self.algorithm.reactToDecision(reward, newInputs, isFinalDecision)
                totalEpochLearningLoss = totalEpochLearningLoss + loss

            self.algorithm.endEpoch()
            # \\\

            totalLearningReward = totalLearningReward + totalEpochReward

            elapsedTime = int(round(time.time() * 1000)) - startTime
            print("epoch " + str(iEpoch) + "/" + str(self.epochs) + " | "
                 +"avg. reward: " + str(totalEpochReward) + " (" + str(self.decisionsPerEpoch) + " max) | " + str(elapsedTime) + "ms"
            +" | avg. loss: " + str(totalEpochLearningLoss / self.decisionsPerEpoch))

        return float(totalLearningReward) / (self.epochs * self.decisionsPerEpoch)

    def fetchReward(self):
        fitness = self.controller.fetchFitness()
        self.rewardAlgorithm.updateFitness(fitness)
        return self.rewardAlgorithm.getRewardValue()
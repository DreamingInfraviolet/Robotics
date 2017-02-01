import keras
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd

class MLAlgorithm(object):
    pass

class ExperienceReplay(object):
    def __init__(self, max_memory=100, discount=.9):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, game_over):
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    # Bottleneck in predict()
    def get_batch(self, model, batch_size=10):
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]
        env_dim = self.memory[0][0][0].shape[1]
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]

            inputs[i:i+1] = state_t
            # There should be no target values for actions not taken.
            # Thou shalt not correct actions not taken #deep
            targets[i] = model.predict(state_t)[0]
            Q_sa = np.max(model.predict(state_tp1)[0])
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets


class ToyQlearningAlgorithm(MLAlgorithm):

    class LearningData(object):
        def __init__(self):
            self.model = None
            self.experienceReplay = None
            self.actionCount = 0

    class EpochData(object):
        def __init__(self):
            self.winCount = 0
            self.loss = 0.0
            self.initialInputs = None
            self.actionTaken = -1

    def __init__(self, explorationEpsilon, maxMemory, hiddenSize, batchSize):
        self.epsilon    = explorationEpsilon
        self.maxMemory  = maxMemory
        self.hiddenSize = hiddenSize
        self.batchSize  = batchSize

        self.epochData = None
        self.learningData = None

    def startLearning(self, inputCount, actionCount):
        self.learningData = self.LearningData()

        self.learningData.actionCount = actionCount

        self.learningData.model = Sequential()
        self.learningData.model.add(Dense(self.hiddenSize, input_shape=(inputCount,), activation='relu'))
        self.learningData.model.add(Dense(self.hiddenSize, activation='relu'))
        self.learningData.model.add(Dense(self.learningData.actionCount))
        self.learningData.model.compile(sgd(lr=.2), "mse")

        self.learningData.experienceReplay = ExperienceReplay(max_memory=self.maxMemory)

        # If you want to continue training from a previous model, just uncomment the line bellow
        # model.load_weights("model.h5")

    def startEpoch(self):
        self.epochData = self.EpochData()

    def decideOnAction(self, currentInputs):
        currentInputs = np.array(currentInputs).reshape((1, -1))

        # get next action
        if np.random.rand() <= self.epsilon:
            # Perform random action, explore!
            action = np.random.randint(0, self.learningData.actionCount, size=1)
        else:
            # Predict best action to take
            q = self.learningData.model.predict(currentInputs)
            action = np.argmax(q[0])

        self.epochData.initialInputs = currentInputs
        self.epochData.actionTaken = action

        return action

    def reactToDecision(self, reward, newInputs, finalDecision):
        self.learningData.experienceReplay.remember([self.epochData.initialInputs, self.epochData.actionTaken, reward, np.array(newInputs).reshape((1, -1))], finalDecision)
        inputs, targets = self.learningData.experienceReplay.get_batch(self.learningData.model, batch_size=self.batchSize)
        self.learningData.model.train_on_batch(inputs, targets)


    def endEpoch(self):
        pass
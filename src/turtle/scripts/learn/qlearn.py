#!/usr/bin/env python

import json
import numpy as np
import rospy
import math
import time
import sys
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd
import tensorflow
from std_msgs.msg import Float64
from std_msgs.msg import Bool

print("Started learning system")

sess = tensorflow.Session(config=tensorflow.ConfigProto(log_device_placement=True))

gFitness = 0
gCompass = 0

def updateFitness(msg):
    global gFitness
    gFitness = msg.data

def updateCompass(msg):
    global gCompass
    gCompass = msg.data

fitnessSub = rospy.Subscriber("fitness_evaluator", Float64, updateFitness)
compassSub = rospy.Subscriber("compass", Float64, updateCompass)
lwPub      = rospy.Publisher("/left_wheel_controller/command", Float64)
rwPub      = rospy.Publisher("/right_wheel_controller/command", Float64)
fwPub      = rospy.Publisher("/front_wheel_controller/command", Float64)
turnPub    = rospy.Publisher("/front_caster_controller/command", Float64)
resetPub   = rospy.Publisher("/reset_position", Bool)

rospy.init_node("learner")

def getCurrentTimeMs():
    return int(round(time.time() * 1000))

class RosSimulation(object):
    def __init__(self):
        self.turnDelta = math.pi / 5
        self.movementVelocity = 5.0
        self.runForMs = 10000
        self.reset()

    def _update_state(self, action):
        """
        Updates the state by performing the actions and updates the internal state attribute.
        Input: action and states
        """

        if action == 0:  # left
            self._turn(-self.turnDelta)
        elif action == 1: # right
            self._turn(self.turnDelta)
        elif action == 2: # forward
            self._move(self.movementVelocity)
        elif action == 3: # backward
            self._move(-self.movementVelocity)
        # else: # stay
        #     self._move(0)
        else:
            print("Invalid action")
            sys.exit(2)
            
        self.state = np.array([self._getCompass(), self.desiredWheelOrientation])

    def _get_reward(self):
        """ Returns the reward in the range [-1, 0, 1] """
        newFitness = self._getFitnessFromNode()

        if newFitness > self.bestScore * 1.0001:
            reward = 1
            self.bestScore = newFitness
        elif newFitness < self.initialScore * 0.9999:
            reward = -1
        else:
            reward = 0

        print("Reward " + str(reward))
        return reward

    def _getFitnessFromNode(self):
        rospy.wait_for_message("fitness_evaluator", Float64)
        return gFitness

    def _is_over(self):
        return getCurrentTimeMs() > self.startTime + self.runForMs

    def _getCompass(self):
        rospy.wait_for_message("compass", Float64)
        return gCompass

    def _resetModel(self):
        print("Resetting")
        resetPub.publish(True)

    def _turn(self, angle):
        print("Turning by " + str(angle))
        self.desiredWheelOrientation = self.desiredWheelOrientation + angle
        turnPub.publish(self.desiredWheelOrientation)

    def _move(self, amount):
        print("Moving by " + str(amount))
        lwPub.publish(amount)
        rwPub.publish(amount)
        fwPub.publish(amount)

    def observe(self):
        """ Return the current state """
        return self.state.reshape((1, -1))

    def act(self, action):
        self.rate.sleep()
        self._update_state(action)
        reward = self._get_reward()
        game_over = self._is_over()
        return self.observe(), reward, game_over

    def reset(self):
        self.desiredWheelOrientation = 0
        self.rate = rospy.Rate(10)
        self.state = np.array([0, self.desiredWheelOrientation])
        
        self._move(0)
        self._turn(0)
        self._resetModel()
        
        self.initialScore = self._getFitnessFromNode()
        self.bestScore = self.initialScore

        self.startTime = getCurrentTimeMs()

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


if __name__ == "__main__":
    # parameters
    epsilon = .1  # exploration
    num_actions = 4 # [turn_left, turn_right, move_forward, move_backward]
    epoch = 1000
    max_memory = 500
    hidden_size = 50
    batch_size = 50

    # Define environment/game
    env = RosSimulation()

    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(len(env.state),), activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(sgd(lr=.2), "mse")

    # If you want to continue training from a previous model, just uncomment the line bellow
    # model.load_weights("model.h5")

    # Initialize experience replay object
    exp_replay = ExperienceReplay(max_memory=max_memory)

    # Train
    win_cnt = 0
    for e in range(epoch):
        loss = 0.
        env.reset()
        game_over = False
        # get initial input
        input_t = env.observe()

        while not game_over:
            input_tm1 = input_t
            # get next action
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, num_actions, size=1)
            else:
                q = model.predict(input_tm1)
                action = np.argmax(q[0])

            # apply action, get rewards and new state
            input_t, reward, game_over = env.act(action)
            if reward == 1:
                win_cnt += 1

            # store experience
            exp_replay.remember([input_tm1, action, reward, input_t], game_over)

            # adapt model
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

            loss += model.train_on_batch(inputs, targets)
        print("Epoch {:03d}/999 | Loss {:.4f} | Win count {}".format(e, loss, win_cnt))

    # Save trained model weights and architecture, this will be used by the visualization code
    model.save_weights("model.h5", overwrite=True)
    with open("model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)

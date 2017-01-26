#!/usr/bin/env python

import json
import numpy as np
import time
import keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd
import tensorflow
import ml_controller
import os
import rospy

# os.environ["CUDA_VISIBLE_DEVICES"]="0"

print("Started learning system")

config = tensorflow.ConfigProto(
        device_count = {'GPU': 0}, # Disable GPU
        log_device_placement=True # Write detailed logs
    )
sess = tensorflow.Session(config=config)


def getCurrentTimeMs():
    return int(round(time.time() * 1000))

class RosSimulation(object):
    def __init__(self, controller):
        self.controller = controller
        self.runForMs = 10000
        self.state = np.array([0, 0, 0])
        self.reset()
        self.rate = rospy.Rate(1)

    def _update_state(self, action):
        """
        Updates the state by performing the actions and updates the internal state attribute.
        Input: action and states
        """

        self.controller.performAction(action)
        self.state = np.array(self.controller.fetchInputs())

    def _get_reward(self):
        ''' Returns the reward in the range [-1, 0, 1] '''
        newFitness = self.controller.fetchFitness()

        if newFitness > self.bestScore * 1.01:
            reward = 1
            self.bestScore = newFitness
        elif newFitness < self.initialScore * 0.99:
            reward = -1
        else:
            reward = 0

        print("Reward " + str(reward))
        return reward

    def _is_over(self):
        return getCurrentTimeMs() > self.startTime + self.runForMs

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
        self.controller.reset()
        self.state = np.array(self.controller.fetchInputs())
        
        # Wait a bit for model to reset properly
        time.sleep(0.5)
        self.initialScore = self.controller.fetchFitness()
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


if __name__ == "__main__":
    # parameters
    epsilon = .1  # exploration
    epoch = 100
    max_memory = 500
    hidden_size = 50
    batch_size = 50

    # Define environment/game
    controller = ml_controller.TurtlebotMLController()
    env = RosSimulation(controller)

    num_actions = controller.getActionCount()

    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(controller.getInputCount(),), activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(sgd(lr=.2), "mse")

    # If you want to continue training from a previous model, just uncomment the line bellow
    # model.load_weights("model.h5")

    # Initialize experience replay object
    exp_replay = ExperienceReplay(max_memory=max_memory)

    # Train

    for e in range(epoch):
        startTime = getCurrentTimeMs()
        win_cnt = 0
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
            elapsedTime = getCurrentTimeMs() - startTime
        print("Epoch {:03d}/999 | Loss {:.4f} | Win count {} | took {:.4f} ms".format(e, loss, win_cnt, elapsedTime))

    # Save trained model weights and architecture, this will be used by the visualization code
    model.save_weights("model.h5", overwrite=True)
    with open("model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)

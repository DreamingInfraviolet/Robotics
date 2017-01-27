#!/usr/bin/env python

import json
import numpy as np
import time
import tensorflow
import os
import rospy

import ml_simulation_driver
import ml_controller
import ml_algorithm

os.environ["CUDA_VISIBLE_DEVICES"]="0"

print("Started learning system")

config = tensorflow.ConfigProto(
        device_count         = {'GPU': 0}, # Disable GPU
        log_device_placement = True # Write detailed logs
    )
sess = tensorflow.Session(config=config)

if __name__ == "__main__":

    controller = ml_controller.TurtlebotMLController()
    algorithm  = ml_algorithm.ToyQlearningAlgorithm(explorationEpsilon = 0.1, maxMemory = 1000, hiddenSize = 50, batchSize = 50)
    rewardAlg  = ml_simulation_driver.MLProgressRewardAlgorithm()
    driver     = ml_simulation_driver.MLSimulationDriver(controller, algorithm, rewardAlg, epochs=100, decisionsPerEpoch=20, decisionSimTimeRateHz=1)

    driver.learn()

    # Save trained model weights and architecture, this will be used by the visualization code
    algorithm.learningData.model.save_weights("model.h5", overwrite=True)
    with open("model.json", "w") as outfile:
        json.dump(algorithm.learningData.model.to_json(), outfile)

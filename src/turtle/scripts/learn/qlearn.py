#!/usr/bin/env python

import json
import tensorflow
import time
import os

import ml_simulation_driver
import ml_controller
import ml_algorithm
from scipy.optimize import minimize
import numpy as np

#os.environ["CUDA_VISIBLE_DEVICES"]="0"

print("Started learning system")

config = tensorflow.ConfigProto(
        # device_count         = {'GPU': 0}, # Disable GPU
        log_device_placement = True # Write detailed logs
    )
sess = tensorflow.Session(config=config)

# Learning metafunction. Returns a function to optimise
def learn(epochs, decisionsPerEpoch, decisionSimTimeRateHz, maxMemory, controller):
    # This function includes parameters that should be optimised
    def _learn(x):
        explorationEpsilon = x[0]
        hiddenSize         = int(x[1])
        batchSize          = int(x[2])

        print("Running with: " + str(x))

        np.random.seed(0)

        # Populate algorithms

        algorithm  = ml_algorithm.RealQlearningAlgorithm(explorationEpsilon=explorationEpsilon,
                                                         maxMemory=maxMemory,
                                                         hiddenSize=hiddenSize,
                                                         batchSize=batchSize)

        rewardAlg  = ml_simulation_driver.MLProgressRewardAlgorithm()

        driver     = ml_simulation_driver.MLSimulationDriver(controller,
                                                             algorithm,
                                                             rewardAlg, 
                                                             epochs=epochs,
                                                             decisionsPerEpoch=decisionsPerEpoch,
                                                             decisionSimTimeRateHz=decisionSimTimeRateHz)

        #Learn, and record how many seconds it took    
        startTime = time.time()
        score = driver.learn()
        elapsedTime = time.time() - startTime
    
        # Prepare to calculate final score
        timeDiscountInfluence = 0.0001
        # Original score should be in range [-1, 1].
        # We should penalise the algorithm for taking too long!
        finalScore = score - timeDiscountInfluence * elapsedTime

        print("final score = " + str(finalScore)
        + " | original score = " + str(score)
        + " | elapsed: " + str(elapsedTime))

        return -finalScore

    return _learn
        

if __name__ == "__main__":

    controller = ml_controller.TurtlebotMLController()

    learningFunction = learn(10, 10, 1, 1000, controller)
    initialGuesses = (0.1, 100, 50)
    bounds = ((0.0001, 0.8), (1, 500), (1, 200))
    options = {"disp" : True}
    # Powell for noisy measurements.
    # See http://www.scipy-lectures.org/advanced/mathematical_optimization/#practical-guide-to-optimization-with-scipy
    method="Nelder-Mead"
    minimize(learningFunction, x0=initialGuesses, bounds=bounds, options=options, method=method)

    # Save trained model weights and architecture, this will be used by the visualization code
    # algorithm.learningData.model.save_weights("model.h5", overwrite=True)
    # with open("model.json", "w") as outfile:
    #     json.dump(algorithm.learningData.model.to_json(), outfile)

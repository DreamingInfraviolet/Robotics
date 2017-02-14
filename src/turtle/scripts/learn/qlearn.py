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
        # log_device_placement = True # Write detailed logs
    )
sess = tensorflow.Session(config=config)


#Tensorflow GPU optimization
config.gpu_options.allow_growth = True
from keras import backend as K
K.set_session(sess)

# # Learning metafunction. Returns a function to optimise
# def learn(epochs, decisionsPerEpoch, decisionSimTimeRateHz, maxMemory, controller):
#     # This function includes parameters that should be optimised
#     def _learn(x):
#         explorationEpsilon = x[0]
#         hiddenSize         = int(x[1])
#         batchSize          = int(x[2])
#         epsilonDecay       = x[3]
#         temporalDiscount   = x[4]

#         print("Running with: " + str(x))

#         np.random.seed(0)

#         # Populate algorithms

#         algorithm  = ml_algorithm.RealQlearningAlgorithm(explorationEpsilon=explorationEpsilon,
#                                                          epsilonDecay=epsilonDecay,
#                                                          maxMemory=maxMemory,
#                                                          hiddenSize=hiddenSize,
#                                                          batchSize=batchSize,
#                                                          temporalDiscount=temporalDiscount)

#         rewardAlg  = ml_simulation_driver.MLProgressRewardAlgorithm()

#         driver     = ml_simulation_driver.MLSimulationDriver(controller,
#                                                              algorithm,
#                                                              rewardAlg, 
#                                                              epochs=epochs,
#                                                              decisionsPerEpoch=decisionsPerEpoch,
#                                                              decisionSimTimeRateHz=decisionSimTimeRateHz)
    
#         # We need to do this now so that learning data is available
#         algorithm.startLearning(controller.getInputCount(), controller.getActionCount())
#         # Set discount to 0. We should initialise network with good unitial values
#         algorithm.learningData.experienceReplay.discount = 0

    
#         # Train algorithm on recorded data
#         algorithm.learningData.experienceReplay.memory = np.load("learning-history.npy").tolist()
#         print("Training on existing history (first pass) (" + str(len(algorithm.learningData.experienceReplay.memory)) + " entries)")
#         # for i in range(len(algorithm.learningData.experienceReplay.memory)):
#         for i in range(500):
#             inputs, targets = algorithm.learningData.experienceReplay.get_batch(algorithm.learningData.model, batch_size=algorithm.batchSize)
#             loss = algorithm.learningData.model.train_on_batch(inputs, targets)
#             print(str(i) + ": loss = " + str(loss))

        
#         # Train algorithm on recorded data
#         algorithm.learningData.experienceReplay.discount = temporalDiscount
#         print("Training on existing history (second pass) (" + str(len(algorithm.learningData.experienceReplay.memory)) + " entries)")
#         # for i in range(len(algorithm.learningData.experienceReplay.memory)):
#         for i in range(1000):
#             inputs, targets = algorithm.learningData.experienceReplay.get_batch(algorithm.learningData.model, batch_size=algorithm.batchSize)
#             loss = algorithm.learningData.model.train_on_batch(inputs, targets)
#             print(str(i) + ": loss = " + str(loss))

#         print("Training done. Moving to simulation")

#         # Reset history to start learning from good data
#         algorithm.learningData.experienceReplay.memory = []

#         # Lower history to speed up learning
#         algorithm.learningData.experienceReplay.max_memory = batchSize * 5

#         #Learn, and record how many seconds it took    
#         startTime = time.time()                                                             
#         score = driver.learn(skipStartLearningCall = True)
#         elapsedTime = time.time() - startTime
    
#         # Prepare to calculate final score
#         timeDiscountInfluence = 0.0001
#         # Original score should be in range [-1, 1].
#         # We should penalise the algorithm for taking too long!
#         finalScore = score - timeDiscountInfluence * elapsedTime

#         print("final score = " + str(finalScore)
#         + " | original score = " + str(score)
#         + " | elapsed: " + str(elapsedTime))


#         # np.save("learning-history", algorithm.learningData.experienceReplay.memory)

#         return -finalScore

#     return _learn
        


# Learning metafunction. Returns a function to optimise
def learn(epochs, decisionsPerEpoch, decisionSimTimeRateHz, maxMemory, controller):
    # This function includes parameters that should be optimised
    def _learn(x):
        explorationEpsilon = x[0]
        hiddenSize         = int(x[1])
        batchSize          = int(x[2])
        epsilonDecay       = x[3]
        temporalDiscount   = x[4]

        print("Running with: " + str(x))

        np.random.seed(0)

        # Populate algorithms

        global sess
        algorithm  = ml_algorithm.PolicyGradient(sess)

        rewardAlg  = ml_simulation_driver.MLProgressRewardAlgorithm()

        driver     = ml_simulation_driver.MLSimulationDriver(controller,
                                                             algorithm,
                                                             rewardAlg, 
                                                             epochs=epochs,
                                                             decisionsPerEpoch=decisionsPerEpoch,
                                                             decisionSimTimeRateHz=decisionSimTimeRateHz)
    

        #Learn, and record how many seconds it took    
        startTime = time.time()                                
        score = driver.learn(skipStartLearningCall=False)
        elapsedTime = time.time() - startTime
    
        # Prepare to calculate final score
        timeDiscountInfluence = 0.0001
        # Original score should be in range [-1, 1].
        # We should penalise the algorithm for taking too long!
        finalScore = score - timeDiscountInfluence * elapsedTime

        print("final score = " + str(finalScore)
        + " | original score = " + str(score)
        + " | elapsed: " + str(elapsedTime))


        # np.save("learning-history", algorithm.learningData.experienceReplay.memory)

        return -finalScore

    return _learn
        






if __name__ == "__main__":

    controller = ml_controller.TurtlebotMLController()

    learningFunction = learn(500, 50, 1, 25000, controller)
    initialGuesses = (0.8, 128, 100, 0.5, 0.9)
    bounds = ((0.0001, 0.8), (1, 500), (1, 200), (0, 1), (0, 1))
    options = {"disp" : True}
    # Powell for noisy measurements.
    # See http://www.scipy-lectures.org/advanced/mathematical_optimization/#practical-guide-to-optimization-with-scipy
    method="Nelder-Mead"
    # minimize(learningFunction, x0=initialGuesses, bounds=bounds, options=options, method=method)
    learningFunction(initialGuesses)

    # Save trained model weights and architecture, this will be used by the visualization code
    # algorithm.learningData.model.save_weights("model.h5", overwrite=True)
    # with open("model.json", "w") as outfile:
    #     json.dump(algorithm.learningData.model.to_json(), outfile)

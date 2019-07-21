Images in the test folder with predict_thresholded (without number) in the name are the result of a previosu test with the threshold of 0.5, and # epochs of 10.

Images that containg a number are the results of a new test with thresholds 0.4, 0.5 and 0.6, and # epochs of 3. Ideally, 0_predict_thresholded.png and 0_predict_thresholded_0.5.png should be the same, but they are not (predict_thresholded_0.5.png is completely white) Maybe running for longer epochs is necessary for the network in order to be more sure that lines are a real signal.

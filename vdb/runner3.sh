#!/bin/bash

PS=3 TP=SPK ./process_data.py timit TRAIN mfcc --overwrite true
PS=3 TP=SPK ./process_data.py timit TEST mfcc --overwrite true

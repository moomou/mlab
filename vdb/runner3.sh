#!/bin/bash

TP=SPK ./process_data.py timit TRAIN mfcc --overwrite true
TP=SPK ./process_data.py timit TEST mfcc --overwrite true

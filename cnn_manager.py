#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
from time import time

import predictor_facade
import image_data_loader

# names
image_state_name = "state"
dataset_dir = "dataset"
predictive_service = "predictive_service"
current_model = "model_actual.h5"
new_model = "model.h5"

# paths
current_script_path = os.path.dirname(os.path.realpath(__file__))
current_model = os.path.join(current_script_path,current_model)
current_logs_dir = os.path.join(current_script_path, "logs", format(time()))

# model
image_size = 128

if __name__ == "__main__":

    #sys.argv = [sys.argv[0], predictive_service]
    #sys.argv = [sys.argv[0], "predict", r"C:\Users\Stefan\source\repos\anion0278\cnn_hand_pose\real_time\state100-15-2-23-25_img94_date30-10-2019_11#45#32.jpeg"]
    #sys.argv = [sys.argv[0], "train"]
    print(sys.argv) 

    img_loader = image_data_loader.ImageDataLoader(current_script_path, dataset_dir, image_state_name, image_size)
    predictor = predictor_facade.PredictorFacade(img_loader, current_logs_dir, image_size)

    if (len(sys.argv) == 1):
        print("No arguments provided. See help (-h).")
        sys.exit(0)

    if (sys.argv[1] == "-h"):
        print("    -h - this help")
        print("    -train - train model")
        print("    -predict *fileName* - predict result for file with name")
        sys.exit(0)

    if (sys.argv[1] == "train"):
        print("Training...")
        model = predictor.train_model()
        model.save(new_model)
        sys.exit(0)

    if (sys.argv[1] == "continue"):
        print("Continue training...")
        predictor.train_model(current_model)
        sys.exit(0)

    if (sys.argv[1] == "predict" and not(sys.argv[2].isspace())):
        print("Predicting %s" % sys.argv[2])
        predictor.predict_single_img(sys.argv[2], current_model)
        sys.exit(0)
   
    if (len(sys.argv) == 3 and sys.argv[1] == predictive_service):
        print("Setting up service mode")
        predictor.load_model(sys.argv[2])
        predictor.service_mode()
        sys.exit(0)

    if (sys.argv[1] == predictive_service):
        print("Setting up service mode")
        predictor.load_model(current_model)
        predictor.service_mode()
        sys.exit(0)
   
    print("Unknown argument(s)")
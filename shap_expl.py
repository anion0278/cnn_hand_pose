import sys
import os
from time import time
import tensorflow
print(tensorflow.__version__)
import numpy as np
import predictor_facade
import image_data_loader
import shap
import tensorflow.compat.v1.keras.backend as K

tensorflow.compat.v1.disable_eager_execution() # for SHAP!!!

# names
image_state_name = "state"
dataset_dir = "dataset-shap"
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
    sys.argv = [sys.argv[0], "predict", r"C:\Users\Stefan\source\repos\anion0278\cnn_hand_pose\dataset\state0-0-6-8-14_img37536_date06-02-2020_13#02#28.jpeg"]
    #sys.argv = [sys.argv[0], "train"]
    print(sys.argv) 

    img_loader = image_data_loader.ImageDataLoader(current_script_path, dataset_dir, image_state_name, image_size)
    predictor = predictor_facade.PredictorFacade(img_loader, current_logs_dir, image_size)

    x, y = img_loader.get_train_data()
    x_test = x[85:86]
    background = x[np.random.choice(x.shape[0], 100, replace=False)]


    predictor.load_model("model_no_norm.h5")
    model = predictor.loaded_model.model

    def map2layer(x, layer):
        feed_dict = dict(zip([model.layers[0].input], [x.copy()]))
        return K.get_session().run(model.layers[layer].input, feed_dict)

    # slower
    # uses entire dataset as background
    print(predictor.loaded_model.model.predict(x_test))

    layer = 1
    if True:
        e = shap.GradientExplainer((model.layers[layer].input, model.layers[-1].output), map2layer(x.copy(), layer))
        shap_values, indexes = e.shap_values(map2layer(x_test, layer), ranked_outputs=99) ## 99 does not matter because takes max classes (5)
        #labels = np.expand_dims(np.array(["5", "4", "2", "1", "3"]), axis=0) # fake numbers
        names = np.array(["1", "2", "3", "4", "5"])
        labels = np.vectorize(lambda x: names[x])(indexes)
        # indexes give the order
        shap.image_plot(shap_values, x_test, labels) # order depends on the predicted value for each outlet

    # explain only one output "outlet"
    outlet = 4
    output = tensorflow.slice(model.layers[-1].output[0], [outlet], [1]) #xstart, ystart, xlen, y len
    e = shap.GradientExplainer((model.layers[layer].input, output), map2layer(x.copy(), layer))
    shap_values = e.shap_values(map2layer(x_test, layer), ranked_outputs=99) ## 99 does not matter because takes max classes (5)
    #labels = np.array(["5"]) # bug in shapley
    shap.image_plot(shap_values, x_test) 

    shap.force_plot(e.expected_value[0], shap_values[0][0])

    # requires update to TF2.2
    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(x_test)
    shap.image_plot(shap_values, x_test) # correct order (0,1,2,3,4)

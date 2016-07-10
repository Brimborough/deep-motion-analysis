import numpy as np
import keras

def save_weights(model, file_name):
    all_weights = []
    for layer in model.layers:
       w = layer.get_weights()
       all_weights.append(w)

    all_weights = np.array(all_weights)
    np.save('../../weights/'+str(file_name)+'.npy', all_weights)
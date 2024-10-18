import sctp, socket
import time
from datetime import datetime
from log import *
import threading
from keras.utils import  img_to_array
from tensorflow import keras
import tensorflow as tf
from PIL import Image
import numpy as np
from ricsdl.syncstorage import SyncStorage
from ricsdl.exceptions import RejectedByBackend, NotConnected, BackendError
import redis


SAMPLING_RATE = 7.68e6

spectrogram_time = 0.010  # 10 ms
num_of_samples = SAMPLING_RATE * spectrogram_time
SPEC_SIZE = num_of_samples * 8  # size in bytes, where 8 bytes is the size of one sample (complex64)

PROTOCOL = 'SCTP'
ENABLE_DEBUG = False


# create the sdl client
# ns = "Spectrograms"
# sdl3= SyncStorage()
sdl3 = redis.Redis(host='localhost', port=6379, db=0)
should_send = True


perturbed_result = None


def entry(self):
    global server, should_send, model
    """  Read from interface in an infinite loop and run prediction every second
      TODO: do training as needed in the future
    """
    # sdl3.subscribe_channel(ns, new_spectrogram_cb, 'channel')
    model = load_model()
    while True:
        prev_raw_bytes = get_bytes()
        while True:
            curr_raw_bytes = get_bytes()
            if curr_raw_bytes!= prev_raw_bytes:
                perturbed_result = run_prediction(curr_raw_bytes)
                # save_prediction(original_predicted_label, perturbed_result)
                prev_raw_bytes = curr_raw_bytes
                log_info(f"Perturbed Result is {perturbed_result}")


# This function gets the bytes from the SDL
def get_bytes():
    data_dict = sdl3.get('new_spec')
    return data_dict

# This function writes bytes back to the SDL
def write_bytes(perturbed_image):
    perturbed_bytes = perturbed_image.numpy().tobytes()
    sdl3.set('new_spec', perturbed_bytes)
    return perturbed_bytes

def run_prediction(raw_bytes):
    start_time = time.perf_counter()
    # sample.show()
    if ENABLE_DEBUG:
        log_debug(f"Total time for Iraw bytes conversion: {time.perf_counter() - start_time}")

    sample = process_image(raw_bytes)
    # print(type(sample))

    start_time = time.perf_counter()
    perturbed_sample = pgd_attack(sample,0.02)
    write_bytes(perturbed_sample)
    result = predict_newdata(perturbed_sample)
    log_debug(f"Time for prediction: {time.perf_counter() - start_time}")
    # log_debug(self,sub_mgr)
    if ENABLE_DEBUG:
        log_debug(f"Total time for prediction: {time.perf_counter() - start_time}")

        log_info(f"Prediction result: {result}")

    return result


#Load model
def load_model():
    best_model = keras.models.load_model('/home/aganiyu/pranshav-work/AML-main/icxApp/student.h5')
    return best_model




# Process the image for appropriate shape to be fed into the model
def process_image(img):
    processed_image = np.frombuffer(img, dtype=np.float32).reshape((1,128,128,1))
    # processed_image = np.expand_dims(processed_image, axis=0)
    return processed_image


def predict_newdata(sample):
    
    prob_pred = model(sample)
    original_predicted_label = np.argmax(prob_pred, axis=1)
    original_predicted_label = original_predicted_label[0]
    if original_predicted_label == 0:
        original_predicted_label = 'soi'
    elif original_predicted_label == 1:
        original_predicted_label = 'soi+cwi'
    elif original_predicted_label == 2:
        original_predicted_label = 'soi+ci'
    return original_predicted_label



#PGD ATTACK


def pgd_attack(image, epsilon, iters=5):
    global original_predicted_label
    model = load_model()
    image = tf.convert_to_tensor(image)
    adv_image = tf.identity(image)
    target_label = np.array([[1.,0.,0.]])
    alpha = 0.01
    for i in range(iters):
        with tf.GradientTape() as tape:
            tape.watch(adv_image)
            original_prediction = model(adv_image)
            original_predicted_label = np.argmax(original_prediction, axis=1)
            original_predicted_label = original_predicted_label[0]
            if original_predicted_label == 0:
                original_predicted_label = 'soi'
            elif original_predicted_label == 1:
                original_predicted_label = 'soi+cwi'
            elif original_predicted_label == 2:
                original_predicted_label = 'soi+ci'
            log_info("Original prediction is " + original_predicted_label)
            loss = -1*tf.keras.losses.sparse_categorical_crossentropy(0, original_prediction)
        gradient = tape.gradient(loss, adv_image)
        # print(type(gradient))
        perturbation = alpha*tf.sign(gradient)
        adv_image_perturbed = adv_image + perturbation
        eta = tf.clip_by_value(adv_image_perturbed-image, -epsilon, epsilon)
        # print(eta)
        adv_image = adv_image+eta
        adv_image = tf.clip_by_value(adv_image, 0, 1)
        return adv_image



# def save_prediction(original_prediction, perturbed_prediction): 

#     log_data_path = "/home/sdhungel/AML-metrics-enb/cwi/45db/pgd/eps0.02-attack/log.csv"
#     signal = "soi+cwi"
#     with open(log_data_path, "a") as log:
#         log.write(f"{signal},{original_prediction},{perturbed_prediction}\n")



def start(thread=False):
    entry(None)


if __name__ == '__main__':
    start()


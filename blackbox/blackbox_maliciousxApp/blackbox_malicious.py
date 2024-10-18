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
import json
import io
import redis
import PIL
import matplotlib.pyplot as plt


SAMPLING_RATE = 7.68e6

spectrogram_time = 0.010  # 10 ms
num_of_samples = SAMPLING_RATE * spectrogram_time
SPEC_SIZE = num_of_samples * 8  # size in bytes, where 8 bytes is the size of one sample (complex64)

PROTOCOL = 'SCTP'
ENABLE_DEBUG = False


# create the sdl client
ns = "spectrograms1"
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

    log_data_path = "log1.csv"
    with open(log_data_path, "w") as log:
        log.write(f"original,perturbed\n")

    while True:
        prev_raw_bytes = get_bytes()
        global prev_attacked_bytes
        prev_attacked_bytes = None
        old_iq_time = time.perf_counter()
        while True:
            curr_raw_bytes = get_bytes()
            # print(prev_bytes == None)
               
            if curr_raw_bytes != prev_raw_bytes and curr_raw_bytes != prev_attacked_bytes:
                # perturbed_result, original_result = run_prediction(curr_raw_bytes)
                print('='*80)
                perturbed_result = run_prediction(curr_raw_bytes)
                prev_raw_bytes = curr_raw_bytes
                log_info(f"Perturbed Result is {perturbed_result}")
                print(f"Time to retrieve new data from DB is {time.perf_counter() - old_iq_time}")
                old_iq_time = time.perf_counter()
            # print("Raw bytes from DB remains unchanged")

    # while True:
    #     prev_raw_bytes = get_bytes()
    #     while True:
    #         curr_raw_bytes = get_bytes()
    #         if curr_raw_bytes!= prev_raw_bytes:
    #             perturbed_result = run_prediction(curr_raw_bytes)
    #             # save_prediction(original_predicted_label, perturbed_result)
    #             prev_raw_bytes = curr_raw_bytes
    #             log_info(f"Perturbed Result is {perturbed_result}")


# This function gets the bytes from the SDL
def get_bytes():
    raw_bytes = sdl3.get("new_spec")

    return raw_bytes


def process_image(new_img):
    image_width = 128
    image_height = 128
    crop_size= (80,60,557,425)
    new_img = new_img.convert('L')
    processed_image = new_img
    processed_image = processed_image.crop(crop_size)
    processed_image = processed_image.resize((image_width, image_height))
    processed_image = np.array(processed_image)
    return processed_image
 

# This function writes bytes back to the SDL
def write_bytes(perturbed_image):
    # numpy_arr = perturbed_image
    # print(numpy_arr.shape, numpy_arr.dtype)
    # numpy_arr = numpy_arr.reshape((128, 128, 3))
    # processed_img = Image.fromarray(np.uint8(numpy_arr), 'RGB')
    # processed_imgtobytes = io.BytesIO()
    # prc = processed_img
    # prc.save(processed_imgtobytes,format="PNG")
    # perturbed_bytes = processed_imgtobytes.getvalue()
    # sdl3.set('new_spec', perturbed_bytes)
    # return perturbed_bytes

    global prev_attacked_bytes

    numpy_arr = perturbed_image.numpy()
    print(numpy_arr.shape, numpy_arr.dtype)
    numpy_arr = numpy_arr.squeeze()
    print("shape after squeezing",numpy_arr.shape)
    prc = Image.fromarray((numpy_arr * 255).astype(np.uint8), 'RGB')
    processed_imgtobytes = io.BytesIO()
    prc.save(processed_imgtobytes,format="PNG")
    perturbed_bytes = processed_imgtobytes.getvalue()
    sdl3.set('new_spec', perturbed_bytes)
    plt.imshow(prc)
    # plt.title("attacked image")
    # plt.show()
    # plt.close()
    prev_attacked_bytes = perturbed_bytes
    return perturbed_bytes


def run_prediction(raw_bytes):
    start_time = time.perf_counter()
    # sample.show()
    if ENABLE_DEBUG:
        log_debug(f"Total time for Iraw bytes conversion: {time.perf_counter() - start_time}")
   

    sample = raw_bytes_to_image(raw_bytes)
    if sample is not None:
        sample = np.expand_dims(sample, axis=0)

        start_time = time.perf_counter()
        
        perturbed_sample = fgsm_attack(sample, 0.2)
        write_bytes(perturbed_sample)
        result = predict_newdata(perturbed_sample)
        log_debug(f"Time for attack and storing back in DB: {time.perf_counter() - start_time}")
        # log_debug(self,sub_mgr)
        if ENABLE_DEBUG:
            log_debug(f"Total time for prediction: {time.perf_counter() - start_time}")

            log_info(f"Prediction result: {result}")

        return result
    else:
        log_error("No raw bytes retreived from the database")


#Load model
def load_model():
    # previous ic model
    best_model = keras.models.load_model('/home/aganiyu/pranshav-work/AML-main/blackbox/mob_final_complete.h5')
    # new ic model
    # best_model = keras.models.load_model('/home/aganiyu/pranshav-work/AML-main/blackbox/blackbox_maliciousxApp/mobile_final.h5')
    return best_model



def raw_bytes_to_image(raw_bytes):
    # Read as a PIL image and return
    try:
        ret = io.BytesIO(raw_bytes)
        image = Image.open(ret)
        image = image.convert('RGB')
        image = image.resize((128,128))
        arr = np.array(image)/255.0
        arr = arr.reshape((128,128,3))
        # print("Shape of image is",arr.shape)
        # print("Plotting image")
        plt.imshow(arr)
        # plt.title("Before attack")
        # plt.show()
        plt.close()
        
        return arr
    except PIL.UnidentifiedImageError as e:
        return None


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


# Function for FGSM ATTACK
def fgsm_attack(image, epsilon):
    start = time.time()
    global original_predicted_label
    model = load_model()
    image = tf.convert_to_tensor(image)
    target_label = np.array([[1.,0.,0.]])
    with tf.GradientTape() as tape:
        tape.watch(image)
        original_prediction = model(image)
        original_predicted_label = np.argmax(original_prediction, axis=1)
        original_predicted_label = original_predicted_label[0]
        if original_predicted_label == 0:
            original_predicted_label = 'soi'
        elif original_predicted_label == 1:
            original_predicted_label = 'soi+cwi'
        elif original_predicted_label == 2:
            original_predicted_label = 'soi+ci'
        
        try:
            log_info("Original prediction is " + original_predicted_label)
        except TypeError:
            print("type not string", original_predicted_label.dtype)
        loss = -1*tf.keras.losses.sparse_categorical_crossentropy(0, original_prediction)
    gradient = tape.gradient(loss, image)
    perturbed_image = image + epsilon*tf.math.sign(gradient)
    perturbed_image = tf.clip_by_value(perturbed_image,0,1)
    print("Total time for attack ================================ ", time.time() - start)
    return perturbed_image


def start(thread=False):
    entry(None)


if __name__ == '__main__':
    start()


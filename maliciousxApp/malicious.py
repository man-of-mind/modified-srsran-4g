# import sctp, socket
# import time
# from datetime import datetime
# from log import *
# import threading
# from keras.utils import  img_to_array
# from tensorflow import keras
# import tensorflow as tf
# from PIL import Image
# import numpy as np
# from ricsdl.syncstorage import SyncStorage
# from ricsdl.exceptions import RejectedByBackend, NotConnected, BackendError
# import json
# import redis
# import io


# SAMPLING_RATE = 7.68e6

# spectrogram_time = 0.010  # 10 ms
# num_of_samples = SAMPLING_RATE * spectrogram_time
# SPEC_SIZE = num_of_samples * 8  # size in bytes, where 8 bytes is the size of one sample (complex64)

# PROTOCOL = 'SCTP'
# ENABLE_DEBUG = False


# # create the sdl client
# # ns = "Spectrograms"
# # sdl3= SyncStorage()
# sdl3 = redis.Redis(host='localhost', port=6379, db=0)

# should_send = True


# perturbed_result = None


# def entry(self):
#     global server, should_send, model
#     """  Read from interface in an infinite loop and run prediction every second
#       TODO: do training as needed in the future
#     """
#     # sdl3.subscribe_channel(ns, new_spectrogram_cb, 'channel')
#     model = load_model()

#     log_data_path = "log1.csv"
#     with open(log_data_path, "w") as log:
#         log.write(f"original,perturbed\n")

#     while True:
#         prev_raw_bytes = get_bytes()
#         while True:
#             curr_raw_bytes = get_bytes()
#             if curr_raw_bytes!= prev_raw_bytes:
#                 perturbed_result = run_prediction(curr_raw_bytes)
#                 # save_prediction(original_predicted_label, perturbed_result)
#                 prev_raw_bytes = curr_raw_bytes
#                 log_info(f"Perturbed Result is {perturbed_result}")


# # This function gets the bytes from the SDL
# def get_bytes():
#     # data_dict = sdl3.get(ns, {'new_spec'})
#     # raw_bytes = None
#     raw_bytes = sdl3.get("new_spec")
#     # for key, val in data_dict.items():
#     #     raw_bytes = val
#     return raw_bytes

# # This function writes bytes back to the SDL
# def write_bytes(perturbed_image):
#     perturbed_bytes = perturbed_image.numpy().tobytes()
#     sdl3.set('new_spec', perturbed_bytes)
#     return perturbed_bytes

# def run_prediction(raw_bytes):
#     start_time = time.perf_counter()
#     # sample.show()
#     if ENABLE_DEBUG:
#         log_debug(f"Total time for Iraw bytes conversion: {time.perf_counter() - start_time}")

#     # sample = process_image(raw_bytes)
#     sample = raw_bytes_to_image(raw_bytes)

#     start_time = time.perf_counter()
#     perturbed_sample = fgsm_attack(sample, 0.1)
#     write_bytes(perturbed_sample)
#     result = predict_newdata(perturbed_sample)
#     log_debug(f"Time for attack and storing back in DB: {time.perf_counter() - start_time}")
#     # log_debug(self,sub_mgr)
#     if ENABLE_DEBUG:
#         log_debug(f"Total time for prediction: {time.perf_counter() - start_time}")

#         log_info(f"Prediction result: {result}")

#     return result


# #Load model
# def load_model():
#     best_model = keras.models.load_model('/home/aganiyu/pranshav-work/AML-main/icxApp/student.h5')
#     return best_model


# # Process the image for appropriate shape to be fed into the model
# # def process_image(img):
# #     processed_image = np.frombuffer(img, dtype=np.float32).reshape(128,128,1)
# #     processed_image = np.expand_dims(processed_image, axis=0)
# #     return processed_image


# def raw_bytes_to_image(raw_bytes):
#     # Read as a PIL image and return
#     ret = io.BytesIO(raw_bytes)
#     image = Image.open(ret)
#     image = image.convert('L')
#     image = image.resize((128,128))
#     arr = np.array(image)/255.0
#     arr = arr.reshape((128,128,1))

    
#     return arr  


# def predict_newdata(sample):
    
#     prob_pred = model(sample)
#     original_predicted_label = np.argmax(prob_pred, axis=1)
#     original_predicted_label = original_predicted_label[0]
#     if original_predicted_label == 0:
#         original_predicted_label = 'soi'
#     elif original_predicted_label == 1:
#         original_predicted_label = 'soi+cwi'
#     elif original_predicted_label == 2:
#         original_predicted_label = 'soi+ci'
#     return original_predicted_label


# # Function for FGSM ATTACK
# def fgsm_attack(image, epsilon):
#     global original_predicted_label
#     model = load_model()
#     image = tf.convert_to_tensor(image)
#     target_label = np.array([[1.,0.,0.]])
#     with tf.GradientTape() as tape:
#         tape.watch(image)
#         original_prediction = model(image)
#         original_predicted_label = np.argmax(original_prediction, axis=1)
#         original_predicted_label = original_predicted_label[0]
#         if original_predicted_label == 0:
#             original_predicted_label = 'soi'
#         elif original_predicted_label == 1:
#             original_predicted_label = 'soi+cwi'
#         elif original_predicted_label == 2:
#             original_predicted_label = 'soi+ci'
#         log_info("Original prediction is " + original_predicted_label)
#         print("Original prediction is ", original_predicted_label)
#         loss = -1*tf.keras.losses.sparse_categorical_crossentropy(0, original_prediction)
#     gradient = tape.gradient(loss, image)
#     perturbed_image = image + epsilon*tf.math.sign(gradient)
#     perturbed_image = tf.clip_by_value(perturbed_image,0,1)
#     return perturbed_image


# # def save_prediction(original_prediction, perturbed_prediction): 

# #     log_data_path = "/home/sdhungel/AML-metrics-enb/cwi/45db/fgsm/eps0.02-attack/log.csv"
# #     signal = "soi+cwi"
# #     with open(log_data_path, "a") as log:
# #         log.write(f"{signal},{original_prediction},{perturbed_prediction}\n")


# def start(thread=False):
#     entry(None)


# if __name__ == '__main__':
#     start()


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
import json, base64


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
        while True:
            curr_raw_bytes = get_bytes()
            if curr_raw_bytes!= prev_raw_bytes:
                perturbed_result = run_prediction(curr_raw_bytes)
                # save_prediction(original_predicted_label, perturbed_result)
                prev_raw_bytes = curr_raw_bytes
                log_info(f"Perturbed Result is {perturbed_result}")


# This function gets the bytes from the SDL
def get_bytes():
    # data_dict = sdl3.get(ns, {'new_spec'})
    # raw_bytes = None
    # for key, val in data_dict.items():
    #     raw_bytes = val
    # return raw_bytes
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
    # perturbed_bytes = perturbed_image.numpy().tobytes()
    # sdl3.set('new_spec', perturbed_bytes)
    # return perturbed_bytes
    numpy_arr = perturbed_image.numpy()
    print(numpy_arr.shape, numpy_arr.dtype)
    numpy_arr = numpy_arr.squeeze()
    # numpy_arr = numpy_arr.reshape((128, 128, 1))
    # processed_img = Image.fromarray(np.uint8(numpy_arr), 'L')
    # processed_imgtobytes = io.BytesIO()
    # prc = processed_img
    # numpy_arr = numpy_arr[0]
    # if numpy_arr.shape[-1] == 1:
    #     numpy_arr = numpy_arr.squeeze(-1)
    # else:
    #     raise ValueError("Expected a single channel grayscale image.")
    print("shape after squeezing",numpy_arr.shape)
    prc = Image.fromarray((numpy_arr * 255).astype(np.uint8), 'L')
    # prc = Image.fromarray(np.uint8(numpy_arr), 'L')
    processed_imgtobytes = io.BytesIO()
    prc.save(processed_imgtobytes,format="PNG")
    perturbed_bytes = processed_imgtobytes.getvalue()
    sdl3.set('new_spec', perturbed_bytes)
    # print(type(perturbed_bytes), len(perturbed_bytes), perturbed_bytes)
    # plt.imshow(prc, cmap='gray')
    # plt.title("attacked image")
    # plt.show()
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
        perturbed_sample = fgsm_attack(sample, 0.1)
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
    best_model = keras.models.load_model('/home/aganiyu/pranshav-work/AML-main/icxApp/student.h5')
    # best_model = keras.models.load_model('/home/aganiyu/pranshav-work/AML-main/icxApp/icmodel.h5')
    return best_model


# Process the image for appropriate shape to be fed into the model
# def process_image(img):
#     processed_image = np.frombuffer(img, dtype=np.float32).reshape(128,128,1)
#     processed_image = np.expand_dims(processed_image, axis=0)
#     return processed_image



def raw_bytes_to_image(raw_bytes):
    # Read as a PIL image and return
    try:
        ret = io.BytesIO(raw_bytes)
        image = Image.open(ret)
        print("---------", np.array(image).shape)
        image = image.convert('L')
        image = image.resize((128,128))
        arr = np.array(image)/255.0
        arr = arr.reshape((128,128,1))
        print("Shape of image is",arr.shape)
        # print("raw bytessssssssssssssssssssssssssssssssss", len(raw_bytes), raw_bytes, arr.shape)
        # print("Plotting image")
        # plt.imshow(arr,cmap='gray')
        # plt.show()
        # plt.close()
        
        return arr
    except PIL.UnidentifiedImageError as e:
        return None


def predict_newdata(sample):
    
    prob_pred = model(sample)
    original_predicted_label = np.argmax(prob_pred, axis=1)
    original_predicted_label = original_predicted_label[0]
    print("================>>>>>>>",original_predicted_label)
    if original_predicted_label == 0:
        original_predicted_label = 'soi'
    elif original_predicted_label == 1:
        original_predicted_label = 'soi+cwi'
    elif original_predicted_label == 2:
        original_predicted_label = 'soi+ci'
    return original_predicted_label


# Function for FGSM ATTACK
def fgsm_attack(image, epsilon):
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
            log_error(f"Original prediction label returned is {original_predicted_label}, something is wrong...")
        loss = -1*tf.keras.losses.sparse_categorical_crossentropy(0, original_prediction)
    gradient = tape.gradient(loss, image)
    perturbed_image = image + epsilon*tf.math.sign(gradient)
    perturbed_image = tf.clip_by_value(perturbed_image,0,1)
    return perturbed_image


# def save_prediction(original_prediction, perturbed_prediction): 

#     log_data_path = "/home/sdhungel/AML-metrics-enb/cwi/45db/fgsm/eps0.02-attack/log.csv"
#     signal = "soi+cwi"
#     with open(log_data_path, "a") as log:
#         log.write(f"{signal},{original_prediction},{perturbed_prediction}\n")


def start(thread=False):
    entry(None)


if __name__ == '__main__':
    start()


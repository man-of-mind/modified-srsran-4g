# import sctp, socket
# import time
# from datetime import datetime
# from log import *
# import threading
# from keras.utils import  img_to_array
# from tensorflow import keras
# from PIL import Image
# import numpy as np
# from ricsdl.syncstorage import SyncStorage
# from ricsdl.exceptions import RejectedByBackend, NotConnected, BackendError
# import tensorflow as tf
# import io
# import matplotlib.pyplot as plt
# import redis

# SAMPLING_RATE = 7.68e6

# spectrogram_time = 0.010  # 10 ms
# num_of_samples = SAMPLING_RATE * spectrogram_time
# SPEC_SIZE = num_of_samples * 8  # size in bytes, where 8 bytes is the size of one sample (complex64)

# PROTOCOL = 'SCTP'
# ENABLE_DEBUG = False
# cmds = {
#     'DYNAMIC_SCHEDULING_ON': b'1',
#     'DYNAMIC_SCHEDULING_OFF': b'0'
# }
# server = None

# # create the sdl client
# # ns = "spectrograms1"
# # sdl2= SyncStorage()
# sdl2 = redis.Redis(host='localhost', port=6379, db=0)

# should_send = True

# def post_init(self):
#     global server

#     ip_addr = socket.gethostbyname(socket.gethostname())
#     port_icxApp = 5002

#     log_info(f"connecting using SCTP on {ip_addr}")
#     server = sctp.sctpsocket_tcp(socket.AF_INET)
#     server.bind((ip_addr, port_icxApp)) 
#     server.listen()

#     log_info('Server started')

# interference_result = None


# def entry(self):
#     global server, should_send, model
#     """  Read from interface in an infinite loop and run prediction every second
#       TODO: do training as needed in the future
#     """
#     post_init(self)
#     # sdl2.subscribe_channel(ns, new_spectrogram_cb, 'channel')
#     model = load_model()
#     while True:
#         try:
#             conn, addr = server.accept() 
#             log_info(f'Connected by {addr}')
#             # prev_raw_bytes = get_bytes()
#             prev_raw_bytes = None
#             average1 = 0
#             count1 = []
#             while True:
#                 curr_raw_bytes = get_bytes()
#                 if curr_raw_bytes!= prev_raw_bytes:
#                     # Prediction is run here
#                     start_time = time.perf_counter()
#                     interference_result = run_prediction(curr_raw_bytes)
#                     end_time = time.perf_counter()
#                     print("Total time for model inference", end_time-start_time)
#                     count1.append(end_time-start_time)
#                     # print(count)
                   
#                     prev_raw_bytes = curr_raw_bytes
#                     if len(count1) == 100:
#                         average1= sum(count1)/len(count1)
#                     log_info(f"Average time to make prediction {average1}")
            
#                     if interference_result == 'soi+ci' or interference_result == "soi+cwi":
#                         log_info("SOI+CWI or SOI+CI detected, sending control message to use adaptive MCS")
#                         conn.send(cmds['DYNAMIC_SCHEDULING_ON'])
#                         # time.sleep(5)
#                     elif interference_result == "soi":
#                         log_info("SOI detected, sending control message to to use Fixed MCS")
#                         conn.send(cmds['DYNAMIC_SCHEDULING_OFF'])
#                         # should_send = False
#                 # print(spec, f'result')
                
                

#         except OSError as e:
#             log_error(e)


# # This function gets the bytes from the database
# def get_bytes():
#     # print(raw_bytes,"THIS is raw bytes")
#     raw_bytes = sdl2.get("new_spec")
    
#     return raw_bytes

# # This converts the raw bytes back to an image
# def raw_bytes_to_image(raw_bytes):
#     # Read as a PIL image and return
#     ret = io.BytesIO(raw_bytes)
#     image = Image.open(ret)
#     image = image.convert('L')
#     image = image.resize((128,128))
#     arr = np.array(image)/255.0
#     arr = arr.reshape((128,128,1))
#     # print("Shape of image is",arr.shape)
#     # print("Plotting image")
#     # plt.imshow(arr,cmap='gray')
#     # plt.show()
#     # plt.close()
    
#     return image

# # This runs the prediction
# def run_prediction(raw_bytes):
#     start_time = time.perf_counter()
#     sample = raw_bytes_to_image(raw_bytes)
#     # sample.show()
#     if ENABLE_DEBUG:
#         log_debug(f"Total time for raw bytes conversion: {time.perf_counter() - start_time}")

#     # sample = process_image(raw_bytes)

#     start_time = time.perf_counter()
#     result = predict_newdata(sample)
#     # result = "Just fake prediction"
#     log_debug(f"Time for prediction: {time.perf_counter() - start_time}")
#     # log_debug(self,sub_mgr)
#     if ENABLE_DEBUG:
#         log_debug(f"Total time for prediction: {time.perf_counter() - start_time}")

#         log_info(f"Prediction result: {result}")

#     return result


# #Load model
# def load_model():
#     best_model = keras.models.load_model('/home/aganiyu/pranshav-work/AML-main/icxApp/student.h5') 
#     return best_model


# def predict_newdata(sample):
#     sample = np.expand_dims(sample, axis=0)
#     prob_pred = model.predict(sample)
#     predicted_label = np.argmax(prob_pred, axis=1)
#     predicted_label = predicted_label[0]

#     if predicted_label == 0:
#         predicted_label = 'soi'
#     elif predicted_label == 1:
#         predicted_label = 'soi+cwi'
#     else:
#         predicted_label = 'soi+ci'
#     return predicted_label


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
from PIL import Image
import numpy as np
from ricsdl.syncstorage import SyncStorage
from ricsdl.exceptions import RejectedByBackend, NotConnected, BackendError
import tensorflow as tf
import io
import matplotlib.pyplot as plt
import redis
import PIL

SAMPLING_RATE = 7.68e6

spectrogram_time = 0.010  # 10 ms
num_of_samples = SAMPLING_RATE * spectrogram_time
SPEC_SIZE = num_of_samples * 8  # size in bytes, where 8 bytes is the size of one sample (complex64)

PROTOCOL = 'SCTP'
ENABLE_DEBUG = False
cmds = {
    'DYNAMIC_SCHEDULING_ON': b'1',
    'DYNAMIC_SCHEDULING_OFF': b'0'
}
server = None

# create the sdl client
# ns = "spectrograms1"
# sdl2= SyncStorage()
sdl2 = redis.Redis(host='localhost', port=6379, db=0)

should_send = True

def post_init(self):
    global server

    ip_addr = socket.gethostbyname(socket.gethostname())
    port_icxApp = 5002

    log_info(f"connecting using SCTP on {ip_addr}")
    server = sctp.sctpsocket_tcp(socket.AF_INET)
    server.bind((ip_addr, port_icxApp)) 
    server.listen()

    log_info('Server started')

interference_result = None


def entry(self):
    global server, should_send, model
    """  Read from interface in an infinite loop and run prediction every second
      TODO: do training as needed in the future
    """
    post_init(self)
    # sdl2.subscribe_channel(ns, new_spectrogram_cb, 'channel')
    model = load_model()
    while True:
        try:
            conn, addr = server.accept() 
            log_info(f'Connected by {addr}')
            # prev_raw_bytes = get_bytes()
            prev_raw_bytes = None
            average1 = 0
            count1 = []
            old_iq_time = time.perf_counter()
            while True:
                curr_raw_bytes = get_bytes()
                if curr_raw_bytes!= prev_raw_bytes:
                    print(f"Time to read new data is {time.perf_counter() - old_iq_time}")
                    old_iq_time = time.perf_counter()
                    # Prediction is run here
                    start_time = time.perf_counter()
                    interference_result = run_prediction(curr_raw_bytes)
                    end_time = time.perf_counter()
                    print("Total time for model inference", end_time-start_time)
                    count1.append(end_time-start_time)
                    # print(count)
                   
                    prev_raw_bytes = curr_raw_bytes
                    if len(count1) == 100:
                        average1= sum(count1)/len(count1)
                    log_info(f"Average time to make prediction {average1}")
            
                    # if interference_result == 'soi+ci' or interference_result == "soi+cwi":
                    #     log_info("SOI+CWI or SOI+CI detected, sending control message to use adaptive MCS")
                    #     conn.send(cmds['DYNAMIC_SCHEDULING_ON'])
                    if interference_result == "soi+cwi":
                        log_info("SOI+CWI, sending control message to use adaptive MCS")
                        conn.send(cmds['DYNAMIC_SCHEDULING_ON'])
                        # time.sleep(5)
                    elif interference_result == "soi":
                        log_info("SOI detected, sending control message to to use Fixed MCS")
                        conn.send(cmds['DYNAMIC_SCHEDULING_OFF'])
                    elif interference_result == 'soi+ci':
                        log_info("SOI+CI detected, sending control message to use adaptive MCS==============================")
                        conn.send(cmds['DYNAMIC_SCHEDULING_ON'])
                        # should_send = False
                # print(spec, f'result')
                
                

        except OSError as e:
            log_error(e)


# This function gets the bytes from the database
def get_bytes():
    # data_dict = sdl2.get(ns, {'new_spec'})
    # raw_bytes = None
    # for key, val in data_dict.items():
    #     raw_bytes = val
    # # print(raw_bytes,"THIS is raw bytes")
    # return raw_bytes
    raw_bytes = sdl2.get("new_spec")
    
    return raw_bytes

# This converts the raw bytes back to an image
def raw_bytes_to_image(raw_bytes):
    try: 
        # Read as a PIL image and return
        ret = io.BytesIO(raw_bytes)
        image = Image.open(ret)
        image = image.convert('RGB')
        image = image.resize((128,128))
        arr = np.array(image)/255.0
        arr = arr.reshape((128,128,3))
        # print("Shape of image is",arr.shape)
        # print("Plotting image")
        # plt.imshow(image)
        # plt.title("Victim image as read from the database")
        # plt.show()
        # plt.close()

        # return arr
        return image
    except PIL.UnidentifiedImageError:
        return None

# This runs the prediction
def run_prediction(raw_bytes):
    start_time = time.perf_counter()
    sample = raw_bytes_to_image(raw_bytes)
    if sample is not None:
        # sample.show()
        if ENABLE_DEBUG:
            log_debug(f"Total time for raw bytes conversion: {time.perf_counter() - start_time}")

        # sample = process_image(raw_bytes)

        start_time = time.perf_counter()
        result = predict_newdata(sample)
        # result = "Just fake prediction"
        log_debug(f"Time for prediction: {time.perf_counter() - start_time}")
        # log_debug(self,sub_mgr)
        if ENABLE_DEBUG:
            log_debug(f"Total time for prediction: {time.perf_counter() - start_time}")

            log_info(f"Prediction result: {result}")

        return result
    else:
        log_error("Error retrieving IQ raw bytes from database")


#Load model
def load_model():
    # new model
    # best_model = keras.models.load_model('/home/aganiyu/pranshav-work/AML-main/blackbox/victim_xApp/victim_full.h5')
    # previous model
    best_model = keras.models.load_model('/home/aganiyu/pranshav-work/AML-main/blackbox/victim_xApp/victim_complete.h5')
    return best_model


def predict_newdata(sample):
    sample = np.expand_dims(sample, axis=0)
    prob_pred = model.predict(sample)
    predicted_label = np.argmax(prob_pred, axis=1)
    predicted_label = predicted_label[0]

    if predicted_label == 0:
        predicted_label = 'soi'
    elif predicted_label == 1:
        predicted_label = 'soi+cwi'
    else:
        predicted_label = 'soi+ci'
    return predicted_label


def start(thread=False):
    entry(None)


if __name__ == '__main__':
    start()

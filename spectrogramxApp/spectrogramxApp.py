# ==================================================================================
#       Copyright (c) 2020 China Mobile Technology (USA) Inc. Intellectual Property.
#       Copyright (c) 2022 NextG Wireless Lab Intellectual Property.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
# ==================================================================================
"""
ss entrypoint module

RMR Messages
 #define TS_UE_LIST 30000
for now re-use the 30000 to receive a UEID for prediction
"""


ENABLE_DEBUG = False

print("STARTUP")

from os import getenv
import time
from log import *
import sctp, socket
from keras.utils import  img_to_array
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  
import numpy as np
from PIL import Image
from ricsdl.syncstorage import SyncStorage

import json

print("STAGE 1")

sampling_rates = [7.68e6, 15.36e6]

SAMPLING_RATE = sampling_rates[1]

spectrogram_time = 0.010  # 10 ms
num_of_samples = SAMPLING_RATE * spectrogram_time
SPEC_SIZE = num_of_samples * 8  # size in bytes, where 8 bytes is the size of one sample (complex64)

iq_samples = {}


# create the sdl client
ns = "Spectrograms"
sdl1 = SyncStorage()

def post_init(self):
    global spec_server

    ip_addr = socket.gethostbyname(socket.gethostname())
    port_specxApp = 5001
    log_info(self,f"connecting using SCTP on {ip_addr}")
    spec_server = sctp.sctpsocket_tcp(socket.AF_INET)

    spec_server.bind((ip_addr, port_specxApp))
    spec_server.listen()

    log_info(self, 'spec_Server started')



def entry(self):
    global current_iq_data, spec_server, iq_samples
    
    post_init(self)
    i = 0

    while True:
        try:
            conn, addr = spec_server.accept()
            #conn.setblocking(0)
            log_info(self, f'Connected by {addr}')
            average1 = 0
            count1 = []
            count2 = 0
            while True:
                spec_start = time.perf_counter()
                data = conn.recv(10000)
                # log_info(self, f"Receiving I/Q data...")
                while len(data) < SPEC_SIZE:
                    data += conn.recv(10000)
                spec_end = time.perf_counter()
                recv_ts = time.time()
                log_info(self, f"Received buffer size {len(data)} with ts {data[0:data.find(b'______')].decode() if data.find(b'______') >= 0 else 'not found'}, received at ts {recv_ts}")
                log_info(self, f"Finished receiving message, processing")
                log_info(self,f"Time to receive I/Q samples {spec_end-spec_start}")
                count1.append(spec_end-spec_start)
                # print(count)
                if len(count1) == 100:
                    average1= sum(count1)/len(count1)
                log_info(self, f"Average time {average1}")
                current_iq_data = data  
                #save_iq = bytearray(current_iq_data)
                #with open(f"/mnt/tmp/iq_samples/iq_sample_{time.time()}.bin", "wb") as f:
                #    f.write(save_iq)
                complex_data = np.frombuffer(current_iq_data, dtype=np.complex64)
                # print(complex_data)
                complex_data = complex_data.tolist()
                complex_data = [str(x) for x in complex_data]

                entry = {}
                entry["timestamp"] = time.time()
                entry["data"] = complex_data

                iq_samples[count2] = entry

                

                spec = iq_handler(self, i)
                print(spec)
                
                i += 1
                count2 += 1


        except OSError as e:
            log_error(self, e)

        except KeyboardInterrupt:
            log_info(self, "Saving JSON data...")
            with open("/mnt/tmp/iq_samples/iq_sample.json", "w") as f:
                    json.dump(iq_samples, f)            
            exit(1)
        
def process_image(new_img):
    image_width = 128
    image_height = 128
    crop_size= (80,60,557,425)
    new_img = new_img.convert('L')
    processed_image = new_img
    processed_image = processed_image.crop(crop_size)
    processed_image = processed_image.resize((image_width, image_height))
    processed_image = np.array(processed_image)
    processed_image = processed_image/255.0
    return processed_image

def iq_to_spectrogram(iq_data, sampling_rate=SAMPLING_RATE) -> Image:
    """Convert I/Q data in 1-dimensional array into a spectrogram image as a numpy array
    """
    # The I/Q data is in [I,Q,I,Q,...] format
    # Each one is a 32-bit float so we can combine them easily by reading the array
    # as complex64 (made of two 32-bit floats)
    #complex_data = iq_data.view(np.complex64)
    complex_data = np.frombuffer(iq_data, dtype=np.complex64)

    # print(complex_data)
    # print(len(complex_data), 'length of IQ data')
    
    # Create new matplotlib figure
    fig = plt.figure()

    # Create spectrogram from data
    plt.specgram(complex_data, Fs=sampling_rate)
    # Manually update the canvas
    fig.canvas.draw()

    w, h = [int(i) for i in fig.canvas.get_renderer().get_canvas_width_height()]
    plt.close()
    #print(fig.canvas.tostring_rgb()[2000:3000])
    # Convert image to bytes, then read as a PIL image and return
    img = Image.frombytes('RGB', (w, h), fig.canvas.tostring_rgb())
    return img, process_image(img).astype(np.float32).tobytes()

# Converts I/Q data to spectrogram and saves it
def iq_handler(self, i):
    global current_iq_data
    """Read the latest cell_meas sample from influxDB and run it by the model inference
    """
    # get i/q sample data
    average2 = 0
    count2 = []
    start_time = time.perf_counter()
    sample, image_bytes= iq_to_spectrogram(current_iq_data)
    
    
    # save I/Q samples somewhere
    sample.save(f'samples/{i}.png')
    # save I/Q to database
    # sdl1.set_and_publish(ns, {'channel': 'new spectrogram'}, {'new_spec': image_bytes})
    sdl1.set(ns, {'new_spec': image_bytes})
    end_time = time.perf_counter()
    log_info(self, f"Total time for I/Q data conversion and storing to database: { end_time- start_time}")
    count2.append(end_time-start_time)
    # print(count)
    if len(count2) == 50:
        average2= sum(count2)/50
    log_info(self, f"Average time sending to spec xApp {average2}")
    return "Completed spectrogram"


def start(thread=False):
        
    entry(None)

if __name__ == '__main__':
    # ai_model = load_model_parameter()
    entry(None)

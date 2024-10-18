import sctp, socket
import time
from datetime import datetime
from log import *
import threading
import os
import numpy as np
import struct
import array
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  
import numpy as np
from PIL import Image
# from influxdb import InfluxDBClient
from datetime import datetime
# from ricsdl.syncstorage import SyncStorage
import io
import redis
import json



sampling_rates = [7.68e6, 15.36e6]
SAMPLING_RATE = sampling_rates[0]


spectrogram_time = 0.010 # 10 ms
num_of_samples = SAMPLING_RATE * spectrogram_time
spectrogram_size = num_of_samples * 8  # size in bytes, where 8 bytes is the size of one sample (complex64)

# This is for redis
ns1 = "spectrograms1"
# ns2 = "spectrograms2"
# sdl1 = SyncStorage()
sdl1 = redis.Redis(host='localhost', port=6379, db=0)


server = None
command = b'0'
prev_iq_data = None
command_prev = None
iq_samples = {}
lock = threading.Lock()

def post_init(self):
    global server, spec_conn, ic_conn

    # This will automatically find a correct IP address to use, and works well in the RIC.
    ip_addr = socket.gethostbyname(socket.gethostname())
    port_srsRAN = 5000 # local port to enable connection to srsRAN
    port_specxApp = 5001
    port_icxApp = 5002
    # port_malxApp = 5003

    log_info( f"E2-like enabled, connecting using SCTP on {ip_addr}")
    server = sctp.sctpsocket_tcp(socket.AF_INET)
    server.bind((ip_addr, port_srsRAN)) 
    server.listen()

    log_info('Server started')

    # Create client connection
    spec_conn = sctp.sctpsocket_tcp(socket.AF_INET)
    ic_conn = sctp.sctpsocket_tcp(socket.AF_INET)

    # while True:
    #     try:
    #         spec_conn.connect((os.getenv('SPEC_ADDR', ip_addr), port_specxApp))
    #         break
    #     except:
    #         log_info("Waiting for specxApp to start...")
    #         time.sleep(1)

    
    while True:
        try:
            ic_conn.connect((os.getenv('IC_ADDR', ip_addr), port_icxApp))
            break
        except:
            log_info("Waiting for icxApp to start...")
            time.sleep(1)

    log_info("Client connection established to icxApp")


def handle_clients(client_soc, addr, thread_id):
    # This is used to interact with all client base stations
    global prev_iq_data
    try:
        j = 0
        k = 0
        while True:
            client_soc.send(command)
            print("Sending command to base station", command)
            print(f"[{thread_id}]    Sending control to srsRAN through e2 lite interface by port {addr[1]}")
            iqdata = client_soc.recv(10000) #conn.recv(10000)
            print(f"[{thread_id}]    Receiving I/Q iqdata... from port {addr[1]}")
            i = 0
            while i < spectrogram_size-10000:
                if spectrogram_size - i > 10000:
                    size = 10000
                    curr_data= client_soc.recv(size) #conn.recv(size)
                    # spec_conn.sctp_send(curr_data)
                    iqdata+=curr_data
                else:
                    size = spectrogram_size - i
                    curr_data=client_soc.recv(size) #conn.recv(size)
                    # spec_conn.sctp_send(curr_data)
                    iqdata+=curr_data
                i+=10000
            print(f"[{thread_id}]    Length of received iq buffer representing 10ms frame {len(iqdata)} {addr[1]}")
            current_iq_data = iqdata 
            if current_iq_data!=prev_iq_data:
                if addr[1] == 38071:  
                    # I want to save the I/Q samples into a json file here
                    spec = iq_handler(addr[1],j,current_iq_data)
                    # print(spec)
                    j += 1
                    

                # elif addr[1] == 38072:
                #     spec = iq_handler(addr[1],k)
                #     print(spec)
                #     k += 1
                prev_iq_data = current_iq_data
                time.sleep(0.1)
            else:
                print("IQ DATA FROM RAN UNCHANGED")
    except OSError as e:
        log_error(e)
                
def entry(self):
    global server , conn, iq_samples
    # Initialize the E2-like interface
    post_init(self)

    clients = []

    # Accept number of connections depending on the number of base stations you have
    try:
        # Loop through for any number of connected base stations
        for i in range(1):
            conn, addr = server.accept()
            log_info(f'Connected to RAN {i+1} by {addr}')
            clients.append([conn,addr, i])
            conn.send(f"E2-lite request at {datetime.now().strftime('%H:%M:%S')}".encode('utf-8'))
            print( "Sent E2-lite request to port", addr[1])
        # Start a thread for the icxApp
        ic_recvthread = threading.Thread(target=handle_icclient, args=(ic_conn,))
        ic_recvthread.start()

        # Start threads for any number of base stations
        threads = [threading.Thread(target=handle_clients, args=(conn[0], conn[1], conn[2])) for conn in clients]
        # Start threads
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        print("Base stations have been served.")

    except KeyboardInterrupt:
        for key in iq_samples:
            with open(f'{key}.json') as f:
                json.dump(iq_samples[key], f)

    except OSError as e:
        log_error(e)

# For threading purposes to receive the ic xApp result concurrently.
def handle_icclient(connection):
    global command
    while True:
        command = connection.recv(1)
        if command:
            command_decoded = command.decode()
            print(f"Received command {command_decoded}")

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

def iq_to_spectrogram(iq_data, i,sampling_rate=SAMPLING_RATE)-> Image:
    """Convert I/Q data in 1-dimensional array into a spectrogram image as a numpy array
    """
    global iq_samples
    complex_data = np.frombuffer(iq_data, dtype=np.complex64)
    iq_samples[str(i)] = {
        'data': complex_data
    }
    # print(iq_samples)
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
    # print("canvas", fig.canvas.tostring_rgb())
    
    # Convert image to bytes, then read as a PIL image and return
    img = Image.frombytes('RGB', (w, h), fig.canvas.tostring_rgb())
    processed_img = process_image(img)
    processed_img = Image.fromarray(np.uint8(processed_img))
    processed_imgtobytes = io.BytesIO()
    prc = processed_img
    prc.save(processed_imgtobytes,format="PNG")
    processed_bytes = processed_imgtobytes.getvalue()

    
    return img, processed_bytes  


def iq_handler(ran_id,i,current_iq_data):
    with lock:
        sample, image_bytes= iq_to_spectrogram(current_iq_data,i)
    # save images somewhere on device
    if ran_id == 38071:
        sample.save(f'samples/{i}.png')
        # print(image_bytes, "These are bytes")
        sdl1.set('new_spec',image_bytes)
        # print(sdl1.get(ns1, {'new_spec'}), "This is me checking if they did save")
    # elif ran_id == 38072:
    #     sample.save(f'samples_ran2/{i}.png')
    #     sdl1.set(ns2, {'new_spec2': image_bytes})
    # print(image_bytes)
    # save just latest image to redis database
    
    print("successfully saved to Redis database")
    return "Completed IQ HANDLER"

def save_iq_tojson(filename, data):
    formatted_data = {}
    required_keys = list(data.keys())[-10:]
    for key in required_keys:
        value = data[key]
        formatted_data[key] = {
            'data': [f'({sample.real}+{sample.imag}j)' for sample in value['data']]
        }
    # Write to JSON file
    with open(filename, 'w') as file:
        json.dump(formatted_data, file, indent=4)

    return "IQ samples saved to json file"

def save_kpms_tojson():
    return "KPMs saved to json file"

def start(thread=False):
    entry(None)


if __name__ == '__main__':
    start()

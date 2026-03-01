import cv2
import socket
import numpy as np
import time

# ======================= USER CONFIGURATION =======================
LISTEN_IP = "0.0.0.0"  
LISTEN_PORT = 1234    
# =================================================================
global frame 

def start_receiver():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
    sock.bind((LISTEN_IP, LISTEN_PORT))

    print(f"UDP Receiver active on port {LISTEN_PORT}...")
   
    raw_buffer = bytearray()
    prev_time = 0

    while True:
        try:
            packet, addr = sock.recvfrom(2048)
            raw_buffer.extend(packet)

            if len(raw_buffer) > 2 and raw_buffer[-2:] == b'\xff\xd9':
                start_idx = raw_buffer.rfind(b'\xff\xd8')
               
                if start_idx != -1:
                    frame_data = raw_buffer[start_idx:]
                    img_array = np.frombuffer(frame_data, dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    cv2.imshow("ESP32-CAM Stream", frame)
                    #return frame
                    raw_buffer.clear()
                    
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if len(raw_buffer) > 250000:
                raw_buffer.clear()


        except Exception as e:
            print(f"Error: {e}")
            raw_buffer.clear()

    cv2.destroyAllWindows()
    sock.close()
    return frame
if __name__ == "__main__":
    start_receiver()


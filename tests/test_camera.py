import cv2
import urllib.request
import numpy as np
import time

URL = 'http://192.168.53.56:81/stream'
CHUNK_SIZE = 64
MAX_BUFFER_BYTES = 512_000
REPORT_EVERY = 10

def pop_latest_complete_jpeg(buf):
    """Drain complete JPEGs and return only the newest one to minimize latency."""
    newest = None
    while True:
        start = buf.find(b'\xff\xd8')
        if start < 0:
            break

        end = buf.find(b'\xff\xd9', start + 2)
        if end < 0:
            if start > 0:
                del buf[:start]
            break

        newest = bytes(buf[start:end + 2])
        del buf[:end + 2]

    return newest


stream = urllib.request.urlopen(URL, timeout=5)
buffer = bytearray()
sum_ms = 0.0
count = 0

try:
    while True:
        t = time.perf_counter()
        chunk = stream.read(CHUNK_SIZE)
        if not chunk:
            continue

        buffer.extend(chunk)
        if len(buffer) > MAX_BUFFER_BYTES:
            del buffer[:-MAX_BUFFER_BYTES]

        jpg = pop_latest_complete_jpeg(buffer)
        if jpg is None:
            continue

        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue

        elapsed_ms = (time.perf_counter() - t) * 1000.0
        sum_ms += elapsed_ms
        count += 1

        #cv2.imshow('i', frame)
        # key = cv2.waitKey(1) & 0xFF
        # if key == 27 or key == ord('q'):
        #     break

        if count >= REPORT_EVERY:
            print(f"avg read+decode over {count} frames: {sum_ms / count:.2f} ms", flush=True)
            sum_ms = 0.0
            count = 0
finally:
    stream.close()
    cv2.destroyAllWindows()
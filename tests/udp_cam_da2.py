import numpy as np
import torch
import cv2
import socket
from depth_anything_v2.dpt import DepthAnythingV2

# ======================= UDP CONFIG =======================
LISTEN_IP = "0.0.0.0"
LISTEN_PORT = 1234
MAX_UDP_SIZE = 2048
# =========================================================

# ======================= MODEL CONFIG =====================
INPUT_SIZE = 384            # lower = faster
ENCODER = 'vits'
CHECKPOINT = "checkpoints/depth_anything_v2_metric_hypersim_vits.pth"
MAX_DEPTH = 20
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# =========================================================

# -----------------------------
# LOAD MODEL
# -----------------------------
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
}

model = DepthAnythingV2(
    **{**model_configs[ENCODER], 'max_depth': MAX_DEPTH}
)
model.load_state_dict(torch.load(CHECKPOINT, map_location='cpu'))
model = model.to(DEVICE).eval()

# -----------------------------
# CAMERA INTRINSICS (OPTIONAL)
# -----------------------------
K = np.array([[294.67, 0, 170.88],
              [0, 303.12, 116.93],
              [0, 0, 1]])
dist = np.zeros(5)

def undistort(frame):
    h, w = frame.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1)
    map1, map2 = cv2.initUndistortRectifyMap(
        K, dist, None, new_K, (w, h), cv2.CV_32FC1
    )
    return cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

# -----------------------------
# UDP SOCKET
# -----------------------------
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
sock.bind((LISTEN_IP, LISTEN_PORT))
sock.settimeout(0.2)

print(f"[INFO] Listening UDP on {LISTEN_PORT}")

buffer = bytearray()

def get_udp_frame():
    global buffer
    try:
        packet, _ = sock.recvfrom(MAX_UDP_SIZE)
        buffer.extend(packet)

        if buffer[-2:] == b'\xff\xd9':   # JPEG end
            start = buffer.find(b'\xff\xd8')
            if start != -1:
                jpg = buffer[start:]
                buffer.clear()
                return cv2.imdecode(
                    np.frombuffer(jpg, np.uint8),
                    cv2.IMREAD_COLOR
                )
            buffer.clear()

        if len(buffer) > 300_000:
            buffer.clear()

    except socket.timeout:
        return None
    except Exception:
        buffer.clear()
        return None

    return None

# -----------------------------
# MAIN LOOP
# -----------------------------
frame_id = 0

while True:
    frame = get_udp_frame()
    if frame is None:
        continue

    frame = undistort(frame)

    # -------- DEPTH --------
    depth = model.infer_image(frame, INPUT_SIZE)
    depth = np.clip(depth, 0.1, MAX_DEPTH)

    nearest = np.percentile(depth, 2)

    dmin, dmax = depth.min(), depth.max()
    if dmax - dmin < 1e-6:
        continue

    depth_norm = ((depth - dmin) / (dmax - dmin) * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)

    cv2.putText(
        depth_vis,
        f"{nearest:.2f} m",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 255),
        2
    )

    cv2.imshow("UDP Metric Depth", depth_vis)
    frame_id += 1

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
sock.close()


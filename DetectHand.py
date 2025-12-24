import os
import cv2
import numpy as np
import tensorflow as tf
from collections import deque

# =========================
# CONFIG
# =========================
MODEL_FILENAME = "./model/hand_gesture_cnn_model_4_gestures_final.h5"
CLASS_NAMES = ['0', '1', '2', '3']  
IMG_WIDTH, IMG_HEIGHT = 100, 100

CONF_THRESH = 0.75
SMOOTH_N = 8        
STABLE_K = 4        

ROI_SIZE = 220


PRED_EVERY = 1      


UNKNOWN_HOLD_FRAMES = 12


USE_BLUR = False


print("Dang tai mo hinh da huan luyen...")
if not os.path.exists(MODEL_FILENAME):
    raise FileNotFoundError(f"Khong tim thay model: {MODEL_FILENAME}")

model = tf.keras.models.load_model(MODEL_FILENAME)
print("Da tai thanh cong!")


dummy = tf.zeros([1, IMG_HEIGHT, IMG_WIDTH, 3], dtype=tf.float32)
_ = model(dummy, training=False)


@tf.function
def infer(x):
    y = model(x, training=False)
    return y


print("\nDang khoi dong camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Khong the mo camera. Kiem tra webcam/quyen truy cap!")


cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Camera da san sang. Nhan 'q' de thoat.")


prob_buf = deque(maxlen=SMOOTH_N)
stable_count = 0
current_label = "Unknown"
current_conf = 0.0
unknown_count = 0

frame_i = 0
last_probs = np.ones((len(CLASS_NAMES),), dtype=np.float32) / len(CLASS_NAMES)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Khong nhan duoc khung hinh. Dang thoat...")
        break

    frame_i += 1
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]


    cx, cy = w // 2, h // 2
    x1 = max(0, cx - ROI_SIZE // 2)
    y1 = max(0, cy - ROI_SIZE // 2)
    x2 = min(w, cx + ROI_SIZE // 2)
    y2 = min(h, cy + ROI_SIZE // 2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    roi = frame[y1:y2, x1:x2]


    if roi.size == 0:
        cv2.putText(frame, "dua tay vao vi tri du doan", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Hand Gesture - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue


    roi_resized = cv2.resize(roi, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
    roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)

    if USE_BLUR:
        roi_rgb = cv2.GaussianBlur(roi_rgb, (3, 3), 0)

    img = roi_rgb.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  

   
    if frame_i % PRED_EVERY == 0:
        probs_tf = infer(tf.convert_to_tensor(img, dtype=tf.float32))
        probs = probs_tf[0].numpy()
        last_probs = probs
    else:
        probs = last_probs

    prob_buf.append(probs)

   
    avg_probs = np.mean(prob_buf, axis=0)
    pred_idx = int(np.argmax(avg_probs))
    conf = float(np.max(avg_probs))

  
    if conf >= CONF_THRESH:
        stable_count += 1
        unknown_count = 0
    else:
        stable_count = 0
        unknown_count += 1

    if stable_count >= STABLE_K:
        current_label = CLASS_NAMES[pred_idx]
        current_conf = conf
        stable_count = STABLE_K
    else:
       
        if unknown_count >= UNKNOWN_HOLD_FRAMES:
            current_label = "HINT"
            current_conf = conf

   
    if current_label == "HINT":
        text = "dua tay vao vi tri du doan"
        cv2.putText(frame, text, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        text = f"{current_label} ({current_conf*100:.1f}%)"
        cv2.putText(frame, text, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    
        top2 = avg_probs.argsort()[-2:][::-1]
        debug = " | ".join([f"{CLASS_NAMES[i]}:{avg_probs[i]*100:.0f}%" for i in top2])
        cv2.putText(frame, debug, (30, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

   
    cv2.imshow("Hand Gesture - press q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Da tat")

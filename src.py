import os
import pygame
import numpy as np
import cv2
from datetime import datetime

# Force eager mode so Keras .fit(...) works on single samples
import tensorflow as tf
tf.config.run_functions_eagerly(True)

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# ---- Configuration ----
WIDTH, HEIGHT       = 280, 280
MODEL_FILE          = r"D:\python\mnist_cnn.h5"
CUSTOM_DATA_DIR     = r"D:\python\custom_data"
FINE_TUNE_EPOCHS    = 3    # epochs on full custom set at startup
INCREMENTAL_EPOCHS  = 1    # epochs per new sample
INCREMENTAL_BATCH   = 1

# Prepare folders 0–9 under custom_data
for lbl in range(10):
    os.makedirs(os.path.join(CUSTOM_DATA_DIR, str(lbl)), exist_ok=True)

# ---- Load existing custom samples ----
def load_custom_data():
    Xc, yc = [], []
    for lbl in range(10):
        folder = os.path.join(CUSTOM_DATA_DIR, str(lbl))
        for fn in os.listdir(folder):
            if fn.lower().endswith(".png"):
                img = cv2.imread(os.path.join(folder, fn), cv2.IMREAD_GRAYSCALE)
                img = img.astype("float32")/255.0
                Xc.append(img.reshape(28,28,1))
                yc.append(lbl)
    if Xc:
        return np.stack(Xc), np.array(yc)
    return None, None

# ---- Build model architecture ----
def build_model():
    m = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax'),
    ])
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return m

# ---- Train on MNIST or load & fine-tune on custom ----
def train_or_load_model():
    if os.path.exists(MODEL_FILE):
        model = load_model(MODEL_FILE)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("🔄 Loaded CNN and recompiled.")
    else:
        # Initial MNIST train
        (x_train, y_train), _ = mnist.load_data()
        x_train = x_train.reshape(-1,28,28,1)/255.0
        y_train = to_categorical(y_train,10)

        model = build_model()
        print("🚀 Training on MNIST (5 epochs)...")
        model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)
        model.save(MODEL_FILE)

    # Fine-tune on full custom set at startup
    Xc, yc = load_custom_data()
    if Xc is not None:
        yc_cat = to_categorical(yc,10)
        print(f"✏️  Fine-tuning on {len(Xc)} custom samples ({FINE_TUNE_EPOCHS} epochs)...")
        model.fit(Xc, yc_cat, epochs=FINE_TUNE_EPOCHS, batch_size=min(16,len(Xc)))
        model.save(MODEL_FILE)
    else:
        print("ℹ️  No custom data to fine-tune on.")

    return model

model = train_or_load_model()

# # In trọng số Dense 128
# dense_layer = model.get_layer(index=-2)
# weights, biases = dense_layer.get_weights()
# print("Weights shape:", weights.shape)
# print("Biases shape:", biases.shape)
# print("Sample weights:\n", weights[:5, :3])
# print("Sample biases:\n", biases[:3])

# ---- Pygame setup ----
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Draw digit + CNN + immediate feedback")
font = pygame.font.SysFont("Arial", 32)
canvas = pygame.Surface((WIDTH, HEIGHT))
canvas.fill((0,0,0))

running           = True
drawing           = False
predicted_label   = None
last_raw          = None
awaiting_feedback = False
awaiting_label    = False

# ---- Helpers ----
def surface_to_array(surf):
    arr3 = pygame.surfarray.array3d(surf)
    gray = np.dot(arr3[...,:3], [0.2989,0.5870,0.1140])
    return np.transpose(gray)

def preprocess_drawing(surf):
    gray = surface_to_array(surf)
    inv  = 255 - gray
    _, thresh = cv2.threshold(inv.astype(np.uint8), 100,255,cv2.THRESH_BINARY)
    if thresh.sum() < 1000: return None
    cnts,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    x,y,w,h = cv2.boundingRect(max(cnts,key=cv2.contourArea))
    pad = 20
    x0,y0 = max(x-pad,0), max(y-pad,0)
    x1,y1 = min(x+w+pad,WIDTH), min(y+h+pad,HEIGHT)
    roi = thresh[y0:y1, x0:x1]
    raw28 = cv2.resize(roi,(28,28), interpolation=cv2.INTER_AREA)
    inp   = raw28.astype("float32")/255.0
    inp   = inp.reshape(1,28,28,1)
    return inp, raw28

def save_and_learn(label, raw_img):
    # 1) Save PNG
    folder = os.path.join(CUSTOM_DATA_DIR,str(label))
    fname  = datetime.now().strftime("%Y%m%d%H%M%S%f")+".png"
    path   = os.path.join(folder,fname)
    cv2.imwrite(path, raw_img)
    print(f"[DEBUG] Saved sample → {path}")

    # 2) Immediate incremental fine-tune
    Xi = raw_img.astype("float32")/255.0
    Xi = Xi.reshape(1,28,28,1)
    yi = to_categorical([label],10)
    print(f"[DEBUG] Incremental training on label {label} (1 epoch)...")
    model.fit(Xi, yi, epochs=INCREMENTAL_EPOCHS, batch_size=INCREMENTAL_BATCH, verbose=0)
    model.save(MODEL_FILE)
    print(f"[DEBUG] Model weights updated and saved.")

# ---- Main loop ----
while running:
    for evt in pygame.event.get():
        if evt.type == pygame.QUIT:
            running = False

        # drawing
        elif evt.type == pygame.MOUSEBUTTONDOWN and evt.button == 1:
            drawing = True
        elif evt.type == pygame.MOUSEBUTTONUP and evt.button == 1:
            drawing = False
        elif evt.type == pygame.MOUSEMOTION and drawing:
            pygame.draw.circle(canvas,(255,255,255),evt.pos,6)

        # keys
        elif evt.type == pygame.KEYDOWN:
            # ENTER → predict
            if evt.key==pygame.K_RETURN and not awaiting_feedback:
                result = preprocess_drawing(canvas)
                if result:
                    inp, raw28 = result
                    pl = int(np.argmax(model.predict(inp)))
                    predicted_label, last_raw = pl, raw28
                    awaiting_feedback            = True
                    print(f"Predicted: {pl}  (Y=correct, N=wrong)")
                else:
                    print("Nothing to recognize!")

            # Y → save under predicted + learn
            elif awaiting_feedback and evt.key==pygame.K_y:
                save_and_learn(predicted_label, last_raw)
                awaiting_feedback = False

            # N → next key is true label
            elif awaiting_feedback and evt.key==pygame.K_n:
                awaiting_feedback = False
                awaiting_label    = True
                print("Press the correct digit (0–9) now.")

            # capture correct label
            elif awaiting_label and pygame.K_0<=evt.key<=pygame.K_9:
                correct = evt.key - pygame.K_0
                save_and_learn(correct, last_raw)
                awaiting_label = False
                print(f"Saved & learned as {correct}")

            # C → clear
            elif evt.key==pygame.K_c:
                canvas.fill((0,0,0))
                predicted_label= None
                awaiting_feedback= awaiting_label= False

            # ESC → quit
            elif evt.key==pygame.K_ESCAPE:
                running = False

    # render
    screen.blit(canvas,(0,0))
    if predicted_label is not None:
        msg = f"Predicted: {predicted_label}"
        if awaiting_feedback:
            msg += " [Y/N]?"
        surf = font.render(msg, True, (255,0,0))
        screen.blit(surf,(10,10))

    pygame.display.flip()

pygame.quit()

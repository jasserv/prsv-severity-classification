import tkinter as tk
from tkinter import Label, Button, Frame, Toplevel, messagebox
import numpy as np
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2
from PIL import Image, ImageTk
import os
import time
import csv
from libcamera import Transform

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (320, 440)}, 
    transform=Transform(hflip=1)
)
picam2.configure(config)
picam2.start()
picam2.set_controls({"AfMode": 2})

interpreter = tflite.Interpreter(model_path="/home/amir/Documents/THESIS/effnet_calibrated.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

save_dir = "/home/amir/Documents/THESIS/captured_images"
expert_dir = "/home/amir/Documents/THESIS/expert_annotations"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(expert_dir, exist_ok=True)

csv_file = os.path.join(save_dir, time.strftime("classification_results_%Y%m%d_%H%M%S.csv"))
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Trial No.", "Predicted Classification", "True Classification", "Confidence Level", "Processing Time (ms)", "Result"])

test_counter = 1
is_expert_annotation = None
true_severity_value = None

def ask_annotation():
    global is_expert_annotation
    answer = messagebox.askyesno("Annotation Mode", "Will this image be annotated by an expert?")
    is_expert_annotation = answer

ask_annotation()

root = tk.Tk()
root.title("PRSV Severity Classification")
root.geometry("320x440")

main_frame = Frame(root, width=320, height=440)
main_frame.pack_propagate(False)
main_frame.pack()

camera_label = Label(main_frame)
camera_label.place(x=0, y=0, width=320, height=440)

prediction_label = Label(main_frame, text="Prediction: None", font=("Arial", 10), bg="black", fg="white")
prediction_label.place(x=10, y=10, width=300, height=30)

confidence_label = Label(main_frame, text="Confidence: N/A", font=("Arial", 10), bg="black", fg="white")
confidence_label.place(x=10, y=40, width=300, height=30)

def update_camera():
    frame = picam2.capture_array()
    frame = Image.fromarray(frame).transpose(Image.FLIP_TOP_BOTTOM).resize((320, 440))
    img_tk = ImageTk.PhotoImage(frame)
    camera_label.img_tk = img_tk
    camera_label.config(image=img_tk)
    root.after(50, update_camera)

def capture_and_process():
    global test_counter
    img_dir = expert_dir if is_expert_annotation else save_dir
    img_path = os.path.join(img_dir, f"test_{test_counter}.jpg")
    picam2.capture_file(img_path)
    time.sleep(1)

    if os.path.exists(img_path):
        if is_expert_annotation:
            review_for_expert_annotation(img_path)
        else:
            review_image(img_path)
    else:
        prediction_label.config(text="Error: Image not captured!")

def review_for_expert_annotation(img_path):
    review_window = Toplevel(root)
    review_window.title("Review Image for Expert Annotation")
    review_window.geometry("320x440")
    review_window.grab_set()

    img = Image.open(img_path).transpose(Image.FLIP_TOP_BOTTOM).resize((320, 440))
    img_tk = ImageTk.PhotoImage(img)
    img_label = Label(review_window, image=img_tk)
    img_label.image = img_tk
    img_label.place(x=0, y=0, width=320, height=440)

    severity_values = [0, 1, 3, 5, 7, 9]
    for i, severity in enumerate(severity_values):
        Button(review_window, text=str(severity),
               command=lambda s=severity: [set_severity(s, img_path), review_window.destroy()],
               bg="blue", fg="white").place(x=10 + i*50, y=380, width=40, height=40)

def review_image(img_path):
    review_window = Toplevel(root)
    review_window.title("Review Image")
    review_window.geometry("320x440")
    review_window.grab_set()

    img = Image.open(img_path).transpose(Image.FLIP_TOP_BOTTOM).resize((320, 440))
    img_tk = ImageTk.PhotoImage(img)
    img_label = Label(review_window, image=img_tk)
    img_label.image = img_tk
    img_label.place(x=0, y=0, width=320, height=440)

    Button(review_window, text="Accept", command=lambda: [review_window.destroy(), classify_and_rename(img_path)], bg="green", fg="white").place(x=60, y=380, width=100, height=40)
    Button(review_window, text="Retake", command=review_window.destroy, bg="red", fg="white").place(x=180, y=380, width=100, height=40)

def set_severity(severity, img_path):
    global true_severity_value
    true_severity_value = severity
    classify_and_rename(img_path, true_severity_value)

def classify_and_rename(img_path, true_value=None):
    global test_counter
    start_time = time.time()

    img = Image.open(img_path).convert("RGB").resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data, axis=1)
    confidence = np.max(output_data) * 100
    processing_time = (time.time() - start_time) * 1000

    class_labels = {0: "Null", 1: "Severity 1", 2: "Severity 3", 3: "Severity 5", 4: "Severity 7", 5: "Severity 9"}
    predicted_label = class_labels.get(predicted_class[0], "Unknown")

    img_rotated = img.rotate(180)

    predicted_severity = predicted_label.split()[-1] if predicted_label != "Null" else "0"
    true_severity = str(true_value) if true_value is not None else None

    if true_value is not None:
        result = "Correct" if predicted_severity == true_severity else "Incorrect"
        labeled_img_path = os.path.join(save_dir, f"test_{test_counter}_Pred_{predicted_severity}_True_{true_severity}.jpg")
        img_rotated.save(labeled_img_path)
    else:
        labeled_img_path = os.path.join(save_dir, f"test_{test_counter}_Pred_{predicted_severity}.jpg")
        img_rotated.save(labeled_img_path)
        result = "Not Annotated"

    prediction_label.config(text=f"Prediction: {predicted_label}")
    confidence_label.config(text=f"Confidence: {confidence:.2f}%")

    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([test_counter, predicted_label, true_severity if true_severity else "Not Annotated", f"{confidence:.2f}%", f"{processing_time:.2f}", result])

    test_counter += 1

capture_button = Button(main_frame, text="Capture & Process", command=capture_and_process, font=("Arial", 10), bg="red", fg="white")
capture_button.place(x=60, y=380, width=200, height=40)

update_camera()
root.mainloop()

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 20
CLASSES_LIST = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace", "HorseRiding", "PullUps"]

# Load the trained model
model = load_model('human_action_recognition_model.h5')

# Frame extraction function
def frames_extraction(video_path):
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)
    for frame_counter in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
        frames_list.append(normalized_frame)
    video_reader.release()
    return frames_list

# Prediction function
def recognize_video(video_path):
    print(f"Processing video: {video_path}")
    frames = frames_extraction(video_path)
    print("Input shape to model:", np.array(frames).shape)  # Debug print
    if len(frames) == SEQUENCE_LENGTH:
        input_data = np.expand_dims(frames, axis=0)
        predictions = model.predict(input_data)
        predicted_class_index = np.argmax(predictions)
        predicted_class_name = CLASSES_LIST[predicted_class_index]
        print(f"\nPredicted Activity: {predicted_class_name}\n")

        # Print all class probabilities
        for idx, prob in enumerate(predictions[0]):
            print(f"{CLASSES_LIST[idx]}: {prob:.4f}")

        # Display a sample of the extracted frames
        for i, frame in enumerate(frames):
            if i % 5 == 0:
                plt.imshow(frame)
                plt.axis('off')
                plt.title(f"Prediction: {predicted_class_name}")
                plt.show()
    else:
        print("Video does not have enough frames.")

# Run prediction on a sample video (update filename if needed)
recognize_video("./random/.mp4")

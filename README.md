🏃‍♂️ Human Activity Recognition from Video using CNN

This project demonstrates a deep learning approach to classify human activities from video using Convolutional Neural Networks (CNN). It covers the full pipeline including video preprocessing, model training, and real-time inference.

🎯 Objective

To build a model that can recognize common human actions (e.g., waving, clapping, walking) from video clips using only spatial features extracted from frames.

🚀 Features

Extracts and preprocesses video frames for model training

Builds and trains a CNN using TensorFlow/Keras

Classifies actions into predefined categories

Predicts activities on unseen videos in real time using OpenCV

Saves trained model for future inference

🧠 Tech Stack

Python

TensorFlow / Keras

OpenCV

NumPy

Matplotlib

🧪 How It Works

Videos are loaded and split into frame sequences

Frames are resized, normalized, and labeled

A CNN model is trained on the processed frame data

Saved model is used for real-time action recognition from new videos

📁 Project Structure

bash
Copy
Edit
Human_Activity_Recognition/
│
├── train_model.py               # Script for training CNN model
├── predict_video.py             # Real-time prediction on test videos
├── Human_Action_Recognition.ipynb # Training and evaluation notebook
├── model.h5                     # Trained Keras model
├── random/                      # Folder with video samples
└── Figure_1.png                 # Sample result visualization

⚙️ How to Run

Clone the repository

Place labeled videos in the input directory

Run train_model.py to train the model

Run predict_video.py to classify actions in test videos

📈 Results

Achieved reliable performance on multiple action classes

Verified predictions visually via overlaid results on sample videos

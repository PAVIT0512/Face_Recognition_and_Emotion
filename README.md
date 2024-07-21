# Emotion Detection and Face Recognition System 😊😢😡😱

Hello Viewers,

This is a trained CNN classifier using Keras with a TensorFlow backend, capable of successfully predicting various human emotions 😊😢😡😱. The classifier can distinguish between multiple emotional states such as happiness, sadness, anger, and fear. After training the model, I integrated it with OpenCV and the face_recognition library to detect faces in real-time video streams.

Once a face is detected, it is passed to the classifier, which then predicts the person's emotion. This combination of deep learning and real-time face detection allows for an interactive and dynamic application, capable of understanding human emotions on the fly. The system can draw bounding boxes around detected faces and display the predicted emotions using text labels. This project demonstrates the powerful synergy between machine learning, computer vision, and real-time image processing technologies 🖥️📸🤖.

## 🎥 Demonstration

You can view the demonstration in the provided video link:
[Watch the Demo](https://drive.google.com/file/d/1MoRPzgczFvYnTK6EWHoqH-pw_ZCpfrRf/view?usp=sharing)

## 🛠️ Installation

To run the program in your local environment, install the necessary packages using the following command:
```bash
pip install opencv-python face_recognition imutils matplotlib pandas tqdm tensorflow kaggle
```
## 📁 Face Recognition

To build your dataset for Face Recognition:

1. Create a folder with your name inside the **dataset** folder.
2. Add your images to this folder, or use the provided Python script **build_face_dataset.py** to capture photos from your local system 📷.

To use this script, open build_face_dataset.py and replace line 49:
```sh
p = os.path.sep.join(["dataset/pavit", "{}.png".format(str(total).zfill(5))])
```

with:

```sh
p = os.path.sep.join(["dataset/{Your Name}", "{}.png".format(str(total).zfill(5))])
```

Run the script. Press "k" to capture images and "q" to quit. The photos will be saved in the folder you created 📁.

3. After capturing images, encode them by running **encode_faces.py**. This will create an **encodings.pickle** file that stores the encodings of the images. Once done, run **pi_face_recognition.py** and enjoy the results 🎉.

## 😊 Emotion Detection

To start from scratch:

1. To get a Kaggle API key, follow these steps:
   - Go to your Kaggle account settings.
   - Scroll down to the "API" section.
   - Click the "Create New API Token" button.
   - A file called `kaggle.json` will be downloaded to your computer. This file contains your API key.
   - Place the `kaggle.json` file in the following directory:
     - Windows: `C:\Users\<Windows-username>\.kaggle\kaggle.json`

2. Prepare the dataset by running **dataset_prepare.py**.

3. Generate the **model.h5** file by running **emotion_model_builder.py**.

4. Run **emotions.py** to experience the emotion detection functionality 😊.

   
## 🤖 Emotion + Face Recognition

For a combined experience of emotion detection and face recognition, run the **emotion+face_recognition.py** script.

Feel free to explore the repository and enjoy the experience!










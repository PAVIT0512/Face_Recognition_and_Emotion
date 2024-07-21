Hello Viewers<br><br>
This is a trained CNN classifier using Keras with a TensorFlow backend, capable of successfully predicting various human emotions 😊😢😡😱. The classifier can distinguish between multiple emotional states such as happiness, sadness, anger, and fear. After training the model, I integrated it with OpenCV and the face_recognition library to detect faces in real-time video streams.<br> Once a face is detected, it is passed to the classifier, which then predicts the person's emotion. This combination of deep learning and real-time face detection allows for an interactive and dynamic application, capable of understanding human emotions on the fly. <br>The system can draw bounding boxes around detected faces and display the predicted emotions using both text labels. This project demonstrates the powerful synergy between machine learning, computer vision, and real-time image processing technologies 🖥️📸🤖.<br><br>
You can view the demonstration in the provided video link 🎥:<br>
https://drive.google.com/file/d/1MoRPzgczFvYnTK6EWHoqH-pw_ZCpfrRf/view?usp=sharing<br><br>
First make sure you have entered the virtual python environment.To enter venv use the command:<br>
**.\myenv\Scripts\activate**<br><br>
If you want to run the program in local environment then kindly install these packages using the following command:<br>
**pip install opencv-python face_recognition imutils matplotlib pandas tqdm tensorflow**<br><br>

Note: I recommend to use the virtual environment as it will help you run the code without any errors.<br><br>

**Face Recognition**<br>
To build your dataset for Face Recognition, first create your own folder with your name inside the dataset folder. You can add your own images to the folder, or use the provided Python script called **build_face_dataset.py** to take photos from the local system. 📷<br>
To use this script, open build_face_dataset.py and replace line 49,**p = os.path.sep.join(["dataset/pavit", "{}.png".format(str(total).zfill(5))])**, with **p = os.path.sep.join(["dataset/{Your Name}", "{}.png".format(str(total).zfill(5))])**, then run the code. While running the script, press **"k"** to capture images and **"q"** to quit. This will save the photos in the folder you created. 📁<br><br>
After capturing the images, encode them by running **encode_faces.py**. This will create an encodings.pickle file, storing the encodings of the images. Once this is done, run **pi_face_recognition.py** and enjoy the experience. 🎉<br><br>
**Emotion**<br>
To run this from scratch, first prepare the dataset by running the **dataset_prepare.py** file. To generate the **model.h5** file, run **emotion_model_builder.py**. After that, run the **emotions.py** script and enjoy the experience. 😊.<br><br>
**Emotion+Face Recognition**<br>
For a combined experience, run the **emotion+face_recognition.py** script and enjoy the seamless integration of emotion detection and face recognition. 🤖









# Gender Prediction

This code is a Python script that uses OpenCV and Caffe models to predict the gender of faces detected in images or live camer.

[Gender Detection using OpenCV in Python](https://www.thepythoncode.com/article/gender-detection-using-opencv-in-python)

## Gender Prediction Results

![gender_predicted.jpg](output/live-output-result.jpg)

## Prerequisites

Before running this code, make sure you have the following installed:

- Python
- OpenCV
- Caffe

You will also need the following files:

- "deploy_gender.prototxt" (gender model architecture)
- "gender_net.caffemodel" (pre-trained gender model weights)
- "deploy.prototxt.txt" (face model architecture)
- "res10_300x300_ssd_iter_140000_fp16.caffemodel" (pre-trained face model weights)

- `pip3 install -r requirements.txt`
- Download [face detection](https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel) and [gender detection](https://drive.google.com/open?id=1W_moLzMlGiELyPxWiYQJ9KFaXroQ_NFQ) models and put them in the `weights` folder. Check [the tutorial](https://www.thepythoncode.com/article/gender-detection-using-opencv-in-python) for more on how to set it up.
- Run the program:

```bash
python.exe .\images-gender-detect.py .\images\dna.jpg
```

## Usage

1. Import the required libraries: cv2 (OpenCV) and numpy.

2. Set the paths for the gender model architecture, gender model weights, face model architecture, face model weights, and the mean values for image channels.

3. Load the face and gender models using cv2.dnn.readNetFromCaffe().

4. Define the frame dimensions for resizing.

5. Define a function, "get_faces()", to detect faces in the frame using the face model.

6. Define a function, "get_optimal_font_scale()", to determine the optimal font scale based on the frame width.

7. Define a function, "image_resize()", to resize an image.

8. Define the main function, "predict_gender()", which predicts the gender of faces in live video.

9. Inside "predict_gender()", initialize a new camera capture.

10. Read the frame from the camera capture.

11. Make a copy of the frame and resize it if necessary.

12. Use the "get_faces()" function to get the detected faces in the frame.

13. Initialize male and female counts.

14. Loop through the detected faces.

15. Extract the face image from the frame and create a blob for input to the gender model.

16. Use the gender model to predict the gender of the face.

17. Draw a rectangle around the face and label it with the predicted gender and confidence score.

18. Increment the male or female count based on the predicted gender.

19. Display the processed frame with the predicted genders.

20. Press 'q' to exit the prediction.

21. Uncomment the line to save the processed frame as an image.

22. Clean up and close any open windows.

23. Call the "predict_gender()" function if the script is executed directly.

## Conclusion

This script demonstrates how to use OpenCV and Caffe models to perform gender prediction on faces detected in images or live video. The code can be used as a starting point for implementing gender prediction in various applications.

import mediapipe as mp
import cv2
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import pandas as pd
import threading
import tensorflow as tf



n_time_steps = 27
lm_list = []


BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

_result = None
label = "..."

def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global _result
    _result = result

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

cap = cv2.VideoCapture(0)

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
        annotated_image,
        hand_landmarks_proto,
        solutions.hands.HAND_CONNECTIONS,
        solutions.drawing_styles.get_default_hand_landmarks_style(),
        solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

def make_landmark_timestep(results):
    if len(results.hand_landmarks) > 0:
        c_lm = []
        for id, lm in enumerate(results.hand_landmarks[0]):
            c_lm.append(lm.x)
            c_lm.append(lm.y)
            c_lm.append(lm.z)
            c_lm.append(lm.visibility)
        return c_lm
    else:
        return None

model = tf.keras.models.load_model("model_hand1.h5")

def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (450, 30)
    fontScale = 1
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img

def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    #print(lm_list.shape)
    results = model.predict(lm_list)
    #print(results)
    if results[0][0] > 0.5:
        label = "UP"
    else:
        label = "DOWN"
    return label

def main():
    with GestureRecognizer.create_from_options(options) as recognizer:
        warmup_frames = 60
        i = 0
        while True:
            ret, numpy_frame_from_opencv = cap.read()
            numpy_frame_from_opencv = cv2.flip(numpy_frame_from_opencv, 1)
            i += 1
            if i >= warmup_frames:
                #print("Start detect....")
                if ret:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)
                    i += 1
                    recognizer.recognize_async(mp_image, i)
                    if (_result != None):
                        numpy_frame_from_opencv = draw_landmarks_on_image(numpy_frame_from_opencv,_result)
                        lm = make_landmark_timestep(_result)
                        # if lm == None:
                        #     lm_list = []
                        if lm != None:
                            lm_list.append(lm)
                        if len(lm_list) == n_time_steps:
                            t1 = threading.Thread(target=detect, args=(model, lm_list,))
                            t1.start()
                            lm_list = []
                    #print(len(lm_list))
                numpy_frame_from_opencv = draw_class_on_image(label, numpy_frame_from_opencv)
                cv2.imshow("Image", cv2.cvtColor(numpy_frame_from_opencv, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
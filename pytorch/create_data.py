import mediapipe as mp
import cv2
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import pandas as pd


lm_list = []
LABEL = "LIKE"
N_DATA = 15
_result = None
data = []


BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode


def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global _result
    _result = result

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='../data/gesture_recognizer.task'),
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
        for id, lm in enumerate(results.hand_landmarks[0]):
            lm_list.append(lm.x)
            lm_list.append(lm.y)
            lm_list.append(lm.z)
            lm_list.append(lm.visibility)
        return True
    else:
        return False

with GestureRecognizer.create_from_options(options) as recognizer:
    i = 0
    while len(data) < N_DATA:
        ret, numpy_frame_from_opencv = cap.read()
        numpy_frame_from_opencv = cv2.flip(numpy_frame_from_opencv, 1)
        if ret:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)
            i += 1
            recognizer.recognize_async(mp_image, i)
            if (_result != None):
                numpy_frame_from_opencv = draw_landmarks_on_image(numpy_frame_from_opencv,_result)
                check = make_landmark_timestep(_result)

                if check == False and lm_list != []:
                    data.append(lm_list)
                    print("Done: " + str(len(data)))
                    lm_list = []

            cv2.imshow("Image", cv2.cvtColor(numpy_frame_from_opencv, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) == ord('q'):
                break

new_data = []
for i in range(len(data)):
    if (len(data[i]) / 84) > 10:
        new_data.append(data[i])

with open('../data/' + LABEL + '.txt', 'w') as fp:
    for item in new_data:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Done')
cap.release()
cv2.destroyAllWindows()
import mediapipe as mp
import cv2
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import pandas as pd
import threading
import tensorflow as tf
from selenium import webdriver
import time
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from model import Model
import torch
from selenium.webdriver import ActionChains

def setup_selenium():
    options = webdriver.ChromeOptions()
    options.add_experimental_option("detach", True)
    options.add_experimental_option("excludeSwitches", ['enable-automation'])
    options.add_argument("disable-extensions")
    options.add_argument("--start-maximized")
    options.add_argument(r"--user-data-dir=C:\\Users\\trnth\\AppData\\Local\\Google\\Chrome\\User Data")
    options.add_argument('--profile-directory=Default')

    return webdriver.Chrome(options=options)

# Go to tiktok and click any video to maximize video session
ff = setup_selenium()
ff.get("https://www.tiktok.com/")
ff.find_element(By.XPATH, "//div[contains(@class, 'DivVideoPlayerContainer')]").click()

lm_list = []

# mediapipe hand landmarks setup
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

# Set up model
model = Model(input_dim=84, num_classes=3, lstm_layer=8)
model.load('pretrain/model3.pt')
_result = None
label = "WARM UP ..."

#Function to draw landmarks
def draw_landmarks_on_image(rgb_image, detection_result):
    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
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

def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (470, 30)
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

def detect(model,):
    global label
    global lm_list
    list = lm_list.copy()
    list = np.array(list)
    list = np.expand_dims(list, axis=0)
    model.reset_hidden()
    #print(lm_list.shape)
    output = model.forward(torch.Tensor(list).squeeze(0))
    result = output[0].max(0)[1]
    if result == 1:
        label = "UP"
    elif result == 0:
        label = "DOWN"
    elif result == 2:
        label = "LIKE"
    print("----------------------" + label  + "---------------------")
    return label



def main():
    with GestureRecognizer.create_from_options(options) as recognizer:
        warmup_frames = 60
        i = 0
        cap = cv2.VideoCapture(0)
        global lm_list
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
                        if lm != None:
                            lm_list.append(lm)
                        if lm == None and len(lm_list) > 20:
                            t1 = threading.Thread(target=detect, args=(model,))
                            t1.start()
                            t1.join()
                            lm_list = []
                            #time.sleep(1)
                            if label == "UP":
                                ff.find_element(By.TAG_NAME, 'body').send_keys(Keys.UP)
                            if label == "DOWN":
                                ff.find_element(By.TAG_NAME, 'body').send_keys(Keys.DOWN)
                            if label == "LIKE":
                                action = ActionChains(ff)
                                element = ff.find_element(By.XPATH, "//div[contains(@class, 'DivVideoPlayerContainer')]")
                                action.double_click(element).perform()
                            
                        if lm != None:
                            lm_list.append(lm)
                        
                    #print(len(lm_list))
                numpy_frame_from_opencv = draw_class_on_image(label, numpy_frame_from_opencv)
                cv2.imshow("Image", cv2.cvtColor(numpy_frame_from_opencv, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
import cv2
import time
import mediapipe as mp
import utils
import pyautogui
from pynput.mouse import Button, Controller
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses info and warnings, only shows errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import absl.logging
absl.logging.set_verbosity('error')  # This reduces absl warnings


screen_width, screen_height = pyautogui.size()
mouse = Controller()
cap = cv2.VideoCapture(0)

# import and set up the hand model
mpHands = mp.solutions.hands
hand = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=2
)
# ham ve hand landmark
draw = mp.solutions.drawing_utils

# tìm đầu ngón trỏ
def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]
        return hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
    return None

def mouse_move(index_finger_tip):
    if index_finger_tip is not None:
        x = int(index_finger_tip.x * screen_width)
        y = int(index_finger_tip.y * screen_height)
    pyautogui.moveTo(x,y)

def left_click(landmark_list, thumb_index_dist):
    return (
            thumb_index_dist > 50
            and utils.get_angle(landmark_list[8], landmark_list[6], landmark_list[5]) < 90
            and utils.get_angle(landmark_list[12],landmark_list[10],landmark_list[9]) > 90
    )

def right_click(landmark_list, thumb_index_dist):
    return (
            thumb_index_dist > 50
            and utils.get_angle(landmark_list[8], landmark_list[6], landmark_list[5]) > 90
            and utils.get_angle(landmark_list[12],landmark_list[10],landmark_list[9]) < 90
    )
def double_click(landmark_list, thumb_index_dist):
    return (
            thumb_index_dist > 50
            and utils.get_angle(landmark_list[8], landmark_list[6], landmark_list[5]) < 90
            and utils.get_angle(landmark_list[12],landmark_list[10],landmark_list[9]) < 90
    )
def screen_shot(landmark_list, thumb_index_dist):
    return (
            thumb_index_dist < 50
            and utils.get_angle(landmark_list[8], landmark_list[6], landmark_list[5]) < 90
            and utils.get_angle(landmark_list[12],landmark_list[10],landmark_list[9]) < 90
    )

def detect_gestures(frame, landmark_list, processed):
    if len(landmark_list) >=21:
        index_finger_tip = find_finger_tip(processed)
        thumb_index_distant = utils.get_distance([landmark_list[4],landmark_list[5]])

        # move the mouse
        if thumb_index_distant < 50 and utils.get_angle(landmark_list[8],landmark_list[6],landmark_list[5])>90:
            mouse_move(index_finger_tip)
        # LEFT CLICK
        elif left_click(landmark_list, thumb_index_distant):
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame,"Left Click",(100,50),cv2.FONT_ITALIC, 1, (0,255,0),2)
        #RIGHT CLICK
        elif right_click(landmark_list, thumb_index_distant):
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame,"Right Click",(100,50),cv2.FONT_ITALIC, 1, (0,0,255),2)
        #DOUBLE CLICK
        elif double_click(landmark_list, thumb_index_distant):
            pyautogui.doubleClick()
            cv2.putText(frame, "Double Click", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        # SCREEN SHOT
        elif screen_shot(landmark_list, thumb_index_distant):
            im1 = pyautogui.screenshot()
            label = np.random.randint(1000, size=1)
            im1.save(f'my_screenshot_{label}.png')
            cv2.putText(frame, "Screenshot Taken", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

def main():
    preTime = 0
    while cap.isOpened():
        success,img = cap.read()
        img = cv2.flip(img,1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if not success:
            print("Không thể nhận frame từ camera")
            break
        # xu ly frame

        processed = hand.process(imgRGB)

        landmark_list = []

        # LAY VA VE HAND LANDMARKS TU ANH DA XU LY
        if processed.multi_hand_landmarks:
            hand_landmarks = processed.multi_hand_landmarks[0]
            draw.draw_landmarks(img,hand_landmarks, mpHands.HAND_CONNECTIONS)

            for lm in hand_landmarks.landmark:
                landmark_list.append((lm.x,lm.y))
            # print(utils.get_distance([landmark_list[4],landmark_list[5]]))
            # print(utils.get_angle(landmark_list[8],landmark_list[6],landmark_list[5]))

        # PHAT HIEN CU CHI
        detect_gestures(img, landmark_list, processed)
        cTime = time.time()
        fps = int(1/(cTime-preTime))
        preTime = cTime
        cv2.putText(img,"fps {}".format(fps), (30,50),cv2.FONT_ITALIC, 1, (0,255,0),2)
        cv2.imshow("Img",img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
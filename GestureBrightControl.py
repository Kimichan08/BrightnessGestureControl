# Import các thư viện 
import cv2 
import mediapipe as mp  
from math import hypot 
import screen_brightness_control as sbc 
import numpy as np 

mpHands = mp.solutions.hands 
hands = mpHands.Hands( 
    static_image_mode=False, 
    model_complexity=1, 
    min_detection_confidence=0.75, 
    min_tracking_confidence=0.75, 
    max_num_hands=2
) 

Draw = mp.solutions.drawing_utils 

# Bắt đầu quay video từ webcam
cap = cv2.VideoCapture(0) 

while True: 
    _, frame = cap.read() 

    frame = cv2.flip(frame, 1) 

    # Chuyển đổi màu sắc và xử lý phát hiện bàn tay
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 

    Process = hands.process(frameRGB) 

    # Xử lý các điểm đặc trưng của bàn tay
    landmarkList = [] 
   
    if Process.multi_hand_landmarks: 
        # Phát hiện dấu tay
        for handlm in Process.multi_hand_landmarks: 
            for _id, landmarks in enumerate(handlm.landmark): 
                height, width, color_channels = frame.shape 

                x, y = int(landmarks.x*width), int(landmarks.y*height) 
                landmarkList.append([_id, x, y]) 
            # Vẽ các điểm đặc trưng lên khung hình
            Draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS) 

    if landmarkList != []: 
        
        x_1, y_1 = landmarkList[4][1], landmarkList[4][2] 
        x_2, y_2 = landmarkList[20][1], landmarkList[20][2] 

        # vẽ hình tròn ở đầu ngón cái và ngón út
        cv2.circle(frame, (x_1, y_1), 7, (0, 255, 0), cv2.FILLED) 
        cv2.circle(frame, (x_2, y_2), 7, (0, 255, 0), cv2.FILLED) 

        cv2.line(frame, (x_1, y_1), (x_2, y_2), (0, 255, 0), 3) 

        # Tính khoảng cách Euclidean giữa đầu ngón tay cái và đầu ngón tay út
        L = hypot(x_2-x_1, y_2-y_1) 
# tính tỷ lệ độ dài của tay
        #  Sử dụng nội suy tuyến tính để chuyển đổi khoảng cách sang mức độ sáng.
        b_level = np.interp(L, [15, 220], [0, 100]) 
        cv2.putText(frame,f"{int(b_level)}%",(10,40),cv2.FONT_ITALIC,1,(0, 255, 98),3)
        # Đặt độ sáng màn hình dựa trên mức độ sáng tính toán được
        sbc.set_brightness(int(b_level)) 

    cv2.imshow('Image', frame) 
    if cv2.waitKey(1) & 0xff == ord('q'): 
        break
    
cap.release()        
cv2.destroyAllWindows() 
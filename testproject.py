import mediapipe as mp
import cv2
import numpy as np
import csv

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

buffer = [0 for element in range(60)]
buffer1 = []
buffer1.extend(buffer)
tocsv = [1]
overall_count = 0
flag = False

#initiate holistic model
with mp_holistic.Holistic(min_detection_confidence = 0.5,min_tracking_confidence = 0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        #recoloring image since frame is in BGR
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #make detections
        results = holistic.process(image)
        image_height, image_width, _ = image.shape
        #convert image back to BGR to process it
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            # Calculate angles
            angle = int(calculate_angle(shoulder, elbow, wrist))
            angle1 = int(calculate_angle(hip, knee, ankle))
            
            # Visualize required angle

            cv2.putText(image, "Knee angle: " + str(angle1),(100,100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
            
            #When the knee angle dips under 70 degrees, the buffer starts recording values

            if(angle1<120 or flag):
                if(overall_count==len(buffer)):
                    print("STOP")
                    minimum = min(buffer)
                    pos = buffer.index(minimum)
                    tocsv[0] = angle
                    with open('angles.csv', 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(tocsv)
                    overall_count=0
                    flag = False
                else:
                    print("HI")
                    flag = True
                    buffer[overall_count] = hip[1]
                    buffer[overall_count] = angle
                    overall_count+=1


        except:
            pass

        #Body Drawings
        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS)
        

        cv2.imshow('Webcam Footage',image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

=======
import mediapipe as mp
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)

#a

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

#initiate holistic model
with mp_holistic.Holistic(min_detection_confidence = 0.5,min_tracking_confidence = 0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        #recoloring image since frame is in BGR
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #make detections
        results = holistic.process(image)
        image_height, image_width, _ = image.shape
        #convert image back to BGR to process it
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)        

        try:
            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # Calculate angle
            angle = int(calculate_angle(shoulder, elbow, wrist))
            
            # Visualize angle
            cv2.putText(image, "Elbow angle: " + str(angle),(100,100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        except:
            pass

        #Body Drawings
        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS)
        

        cv2.imshow('Webcam Footage',image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


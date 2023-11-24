import cv2
import dlib
import imutils
import winsound
from imutils import face_utils
from scipy.spatial import distance

shapePredictorModel = "shape_predictor_68_face_landmarks.dat"
shapePredictor = dlib.shape_predictor(shapePredictorModel)
faceDetector = dlib.get_frontal_face_detector()

cam = cv2.VideoCapture(0)

if cam.isOpened() == False:
    cam.open()

#get the co-ord of left and right eye
(lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# print(lstart, lend)
# print("Right")
# print(rstart, rend)

def eyeAspectRatio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    EAR = (A+B) / (2.0*C)
    return EAR

count = 0
earThresh = 0.3 #Distance between the vertical eye cordinates threshold
earFrames = 48 #Frames for eye closures

#For Beeping
frequency = 2500
duration = 1000

while True:
    ret, frame = cam.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceDetector(gray,0)

    for face in faces:
        #Determine the facial landmarks for the face region
        facialLandmarks = shapePredictor(gray, face)

        #Convert the facial landmark (x,y) coordinates to a numpy array
        facialLandmarks = face_utils.shape_to_np(facialLandmarks)

        leftEye = facialLandmarks[lstart:lend]
        rightEye = facialLandmarks[rstart:rend]
        leftEAR = eyeAspectRatio(leftEye)
        rightEAR = eyeAspectRatio(rightEye)

        ear = (leftEAR + rightEAR)/ 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        cv2.drawContours(frame, [leftEyeHull], -1, (0,0,255), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0,0,255), 1)

        if ear < earThresh:
            count += 1

            if count >= earFrames:
                cv2.putText(frame, "Drowsiness Detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                winsound.Beep(frequency, duration)
        else:
            count = 0

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
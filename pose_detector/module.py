import cv2
import mediapipe as mp

class PoseEstimator():
    def __init__(self, static_image_mode=False,
               model_complexity=1,
               smooth_landmarks=True,
               enable_segmentation=False,
               smooth_segmentation=True,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5) -> None:
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=static_image_mode,
               model_complexity=model_complexity,
               smooth_landmarks=smooth_landmarks,
               enable_segmentation=enable_segmentation,
               smooth_segmentation=smooth_segmentation,
               min_detection_confidence=min_detection_confidence,
               min_tracking_confidence=min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils
    def findPoses(self, img, points=[11, 12], draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        landmarks = self.results.pose_landmarks
        if landmarks:
            self.mpDraw.draw_landmarks(img, landmarks, self.mpPose.POSE_CONNECTIONS)
            for id, point in enumerate(landmarks.landmark):
                h, w, c = img.shape
                cw, cy = int(point.x * w), int(point.y * h)
                if draw and id in points :
                    cv2.circle(img, (cw, cy), 20, (255, 0, 0), cv2.FILLED)
                
        return img
    def findPositions(self, img):
        lmList = []
        landmarks = self.results.pose_landmarks
        if landmarks:
            for id, point in enumerate(landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(point.x * w), int(point.y * h)
                lmList.append([id, cx, cy])
                        
        return lmList

    
def main():
    cap = cv2.VideoCapture(0)
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils
    while True: 
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        landmarks = results.pose_landmarks
        if landmarks:
            mpDraw.draw_landmarks(img, landmarks, mpPose.POSE_CONNECTIONS)
            for id, point in enumerate(landmarks.landmark):
                h, w, c = img.shape
                cw, cy = int(point.x * w), int(point.y * h)
                if id == 12 or id==11:
                    cv2.circle(img, (cw, cy), 20, (255, 0, 0), cv2.FILLED)


        cv2.imshow('Image', img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
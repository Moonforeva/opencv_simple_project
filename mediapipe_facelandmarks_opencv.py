"""
Simple OpenCV project that use Mediapipe's Face Landmarks detector
"""

import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import cv2
import time 
import numpy as np

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"), # download at https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
    running_mode=VisionRunningMode.VIDEO)

def draw_landmarks_on_image(rgb_frame, detection_result: mp.tasks.vision.FaceLandmarkerResult):
    face_landmarks_list = detection_result.face_landmarks
    annotated_frame = np.copy(rgb_frame)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_frame,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
    
    return annotated_frame

def main(face_video_path:str):
    cap = cv2.VideoCapture(filename=face_video_path)
    with FaceLandmarker.create_from_options(options=options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is False:
                break
            mp_frame = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame)
            face_landmarker_result = landmarker.detect_for_video(mp_frame, timestamp_ms=int(time.time() * 1000))
            resulted_frame = draw_landmarks_on_image(mp_frame.numpy_view(), face_landmarker_result)

            cv2.imshow("Face Landmarks", resulted_frame)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main(face_video_path="xxx.mp4") # add video path that contain face here
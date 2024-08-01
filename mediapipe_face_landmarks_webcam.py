import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import cv2
import time 


mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult

# Global variable
DETECTION_RESULT = None

def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global DETECTION_RESULT
    DETECTION_RESULT = result


options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"), # download at https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

custom_connections = [
    (468, 473), (468, 4), (473, 4), 
    (4, 61), (4,291),(61,291),
    (152,4), (152, 468), (152,473),
    (33, 152), (33, 263), (33,4),
    (263, 4),(263,152)
]

custom_draw = mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=1)

cap = cv2.VideoCapture(0)
with FaceLandmarker.create_from_options(options=options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break

        frame = cv2.flip(frame, 1)

        img_h, img_w, img_c = frame.shape

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_frame = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame)
        face_landmarker_result = landmarker.detect_async(mp_frame, timestamp_ms=time.time_ns() // 1000000)

        current_frame = frame

        if DETECTION_RESULT:
            # Draw landmarks.
            for face_landmarks in DETECTION_RESULT.face_landmarks: 
                face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                face_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x,
                                                    y=landmark.y,
                                                    z=landmark.z) for
                    landmark in
                    face_landmarks
                ])
                mp_drawing.draw_landmarks(
                    image=current_frame,
                    landmark_list=face_landmarks_proto,
                    connections=custom_connections,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=custom_draw)

        cv2.imshow("Face Landmarks", current_frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
    
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
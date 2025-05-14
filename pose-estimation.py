import json
from pathlib import Path
import mediapipe as mp
from PIL import Image, ImageDraw
import numpy as np
import cv2

POSE_LANDMARKS = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index"
}

frame_output_dir = "frames"
pose_output_dir = "poses"
composite_output_dir = "final"

for dir in [frame_output_dir, pose_output_dir, composite_output_dir]:
    Path(dir).mkdir(parents=True, exist_ok=True)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)


def draw_landmarks(image, results):
    # Convert OpenCV image to PIL Image if needed
    if isinstance(image, np.ndarray):
        vector_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        vector_img = Image.new("RGBA", image.size)

    # Create drawing object
    draw = ImageDraw.Draw(vector_img)

    dot_size = 4
    border_size = 2

    # Draw the connections first (lines)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        h, w = vector_img.height, vector_img.width

        # Draw connections
        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]

            start_point = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
            end_point = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))

            draw.line([start_point, end_point], fill='white', width=2)

        # Draw landmarks (nodes)
        for idx, landmark in enumerate(landmarks):
            px = int(landmark.x * w)
            py = int(landmark.y * h)

            if (idx) == 0:
                dot = dot_size * 5.0
                border = (dot_size * 5.0) + border_size
                dot_color = "#ff9900"
            else:
                dot = dot_size
                border = (dot_size + border_size)
                dot_color = '#50D9EF'

            draw.ellipse([(px - border, py - border), (px + border, py + border)], fill='white')
            draw.ellipse([(px - dot, py - dot), (px + dot, py + dot)], fill=dot_color)

            # Draw landmark index
            draw.text((px - 3, py - 3), f'#{idx}', fill='black')

    return vector_img


def extract_frames(video_path, interval=1.5):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(frame_rate * interval)

    frame_count = 0
    saved_frames = 0

    # Extract and save frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = Path(frame_output_dir) / f"frame_{saved_frames:03}.jpg"
            cv2.imwrite(frame_filename, frame)
            saved_frames += 1

        frame_count += 1

    cap.release()


def extract_pose(image):
    np_img = np.array(image)

    results = pose.process(np_img)
    landmarks = []

    vector_img = draw_landmarks(image, results)

    if results.pose_landmarks:
        index = 0
        for landmark in results.pose_landmarks.landmark:
            landmarks.append({
                'landmark': POSE_LANDMARKS[index],
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            })
            index += 1

    pose_object = {
        'landmarks': landmarks
    }

    return vector_img, pose_object


def process_frames_to_poses():
    for frame_file in Path(frame_output_dir).glob("*.jpg"):
        frame_image = Image.open(frame_file).convert("RGB")
        pose_image, pose_object = extract_pose(frame_image)

        pose_output_path = Path(pose_output_dir) / f"{frame_file.stem}_pose.png"
        pose_image.save(pose_output_path)

        pose_json_filename = Path(pose_output_dir) / f"{frame_file.stem}_pose.json"
        with open(pose_json_filename, 'w') as f:
            json.dump(pose_object, f)

        composite_image = Image.alpha_composite(frame_image.convert('RGBA'), pose_image).convert('RGB')

        composite_output_path = Path(composite_output_dir) / f"{frame_file.stem}_composite.jpg"
        composite_image.save(composite_output_path)


video_path = "data/BbXfvgDl-0D.mp4"

print('Extracting Frames')
extract_frames(video_path, 0.5)  # every 0.5 seconds

print('Extracting Poses')
process_frames_to_poses()

print('Done')

from flask import Flask, render_template, request, send_file, redirect
import cv2
import numpy as np
from ultralytics import YOLO
from moviepy.editor import VideoFileClip
from tqdm import tqdm
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = YOLO('yolov8s.pt')
crosswalk_region = None  # global region

def detect_crosswalk(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=80,
                            minLineLength=80, maxLineGap=20)

    if lines is None:
        h, w = frame.shape[:2]
        return ((int(w*0.3), int(h*0.6)), (int(w*0.7), int(h*0.9)))

    # filter parallel (horizontal/vertical) lines
    vertical_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        if abs(angle) < 20 or abs(angle) > 160:
            vertical_lines.append((x1, y1, x2, y2))

    if len(vertical_lines) < 3:
        h, w = frame.shape[:2]
        return ((int(w*0.3), int(h*0.6)), (int(w*0.7), int(h*0.9)))

    xs, ys = [], []
    for x1, y1, x2, y2 in vertical_lines:
        xs.extend([x1, x2])
        ys.extend([y1, y2])

    min_x, max_x = max(min(xs)-10, 0), min(max(xs)+10, frame.shape[1])
    min_y, max_y = max(min(ys)-10, 0), min(max(ys)+10, frame.shape[0])

    return ((min_x, min_y), (max_x, max_y))

def process_frame(frame):
    global crosswalk_region
    frame = frame.copy()

    if crosswalk_region is None:
        crosswalk_region = detect_crosswalk(frame)

    results = model(frame, classes=[0], verbose=False)[0]
    occupied = False

    for box in results.boxes.xyxy.cpu().numpy():
        x1,y1,x2,y2 = box.astype(int)
        cx, cy = (x1+x2)//2, (y1+y2)//2
        if (crosswalk_region[0][0] <= cx <= crosswalk_region[1][0] and
            crosswalk_region[0][1] <= cy <= crosswalk_region[1][1]):
            occupied = True
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)

    cv2.rectangle(frame, crosswalk_region[0], crosswalk_region[1], (255,255,0), 3)

    indicator_color = (0,255,0) if not occupied else (0,0,255)
    cv2.circle(frame, (80, frame.shape[0]-80), 50, indicator_color, -1)

    return frame

@app.route('/', methods=['GET', 'POST'])
def index():
    global crosswalk_region
    crosswalk_region = None

    if request.method == 'POST':
        if 'video' not in request.files:
            return redirect(request.url)

        video = request.files['video']
        if video.filename == '':
            return redirect(request.url)

        original_path = os.path.join(UPLOAD_FOLDER, 'original.mp4')
        processed_path = os.path.join(UPLOAD_FOLDER, 'processed.mp4')
        video.save(original_path)

        clip = VideoFileClip(original_path)
        total_frames = int(clip.fps * clip.duration)

        with tqdm(total=total_frames, desc="Processing video frames") as pbar:
            def process_and_update(frame):
                result = process_frame(frame)
                pbar.update(1)
                return result

            processed_clip = clip.fl_image(process_and_update)
            processed_clip.write_videofile(processed_path, codec='libx264', audio=False)

        return render_template('index.html', processed=True)

    return render_template('index.html', processed=False)

@app.route('/download')
def download():
    return send_file(os.path.join(UPLOAD_FOLDER, 'processed.mp4'), as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

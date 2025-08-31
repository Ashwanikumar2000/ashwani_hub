import cv2 as cv
import face_recognition
from flask import Flask, render_template, Response
from PIL import Image
import numpy as np
import os
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

known_face_encodings = []
known_face_names = []

def convert_to_rgb(image_path):
    """Convert an image to RGB format."""
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert("RGB")
        return img
    except Exception as e:
        logger.error(f"Error converting image {image_path} to RGB: {str(e)}")
        return None

def load_known_faces(image_dir="known_faces"):
    """Load and process all known faces from the specified directory."""
    if not os.path.exists(image_dir):
        logger.error(f"Directory {image_dir} does not exist!")
        return False
    
    if not os.listdir(image_dir):
        logger.warning(f"No images found in directory {image_dir}!")
        return False

    loaded_count = 0
    for file_name in os.listdir(image_dir):
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        file_path = os.path.join(image_dir, file_name)
        try:
            img = convert_to_rgb(file_path)
            if img is None:
                continue
                
            image = np.array(img)
            encodings = face_recognition.face_encodings(image)
            
            if not encodings:
                logger.warning(f"No face found in {file_name}")
                continue
                
            known_face_encodings.append(encodings[0])
            name = os.path.splitext(file_name)[0]
            known_face_names.append(name)
            loaded_count += 1
            logger.info(f"Successfully loaded {file_name}")
            
        except Exception as e:
            logger.error(f"Error processing {file_name}: {str(e)}")
    
    if loaded_count == 0:
        logger.error("No valid faces were loaded!")
        return False
        
    return True

# Load known faces at startup
if not load_known_faces():
    logger.error("Failed to load known faces. The application may not work properly.")

def generate_frames():
    """Generate video frames with face recognition."""
    try:
        camera = cv.VideoCapture(0)
        
        if not camera.isOpened():
            logger.error("Cannot access camera. Check permissions or if another app is using it.")
            yield None
            return

        # Set resolution and frame rate (if supported)
        camera.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv.CAP_PROP_FPS, 30)

        process_every_n_frames = 3
        frame_count = 0
        prev_face_locations = []
        prev_face_names = []

        while True:
            success, frame = camera.read()
            if not success:
                logger.warning("Failed to capture frame from camera")
                break

            frame_count += 1
            annotated_frame = frame.copy()

            if frame_count % process_every_n_frames == 0:
                rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                small_frame = cv.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)

                face_locations = face_recognition.face_locations(small_frame)
                face_encodings = face_recognition.face_encodings(small_frame, face_locations)

                face_names = []
                scaled_locations = []

                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    top *= 2
                    right *= 2
                    bottom *= 2
                    left *= 2
                    scaled_locations.append((top, right, bottom, left))

                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"

                    if True in matches:
                        match_indices = [i for i, match in enumerate(matches) if match]
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = match_indices[np.argmin(face_distances[match_indices])]
                        name = known_face_names[best_match_index]

                    face_names.append(name)

                prev_face_locations = scaled_locations
                prev_face_names = face_names

            # Use previous detections for intermediate frames
            for (top, right, bottom, left), name in zip(prev_face_locations, prev_face_names):
                cv.rectangle(annotated_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv.rectangle(annotated_frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv.FILLED)
                cv.putText(annotated_frame, name, (left + 6, bottom - 6),
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

            ret, buffer = cv.imencode('.jpg', annotated_frame)
            if not ret:
                logger.warning("Failed to encode frame")
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n\r\n')

    except Exception as e:
        logger.error(f"Camera error: {str(e)}")
    finally:
        if 'camera' in locals():
            camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=True)
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")

# NOTE: this script is just a demo and not recommended to use in production!
# Code was partially written by LLMs (o3-mini)
# You can use your phone as IP camera by installing special app to your phone.
# For example, you can use:
# https://play.google.com/store/apps/details?id=com.pas.webcam

import time
import cv2
import torch
from flask import Flask, Response, jsonify, render_template_string, request
from torchvision.utils import draw_segmentation_masks
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

MODEL_PATH = "./tuned-segformer-epoch-30"
# Set target FPS (e.g., 10 FPS)
TARGET_FPS = 12
FRAME_DURATION = 1.0 / TARGET_FPS

app = Flask(__name__)

print(f'loading model {MODEL_PATH}')
device = torch.device('cuda')
img_processor = SegformerImageProcessor()

# load model from local trained and saved files
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()
print('model loaded!')

normal_area = 0.70

def segment_door(img):
    global model, img_processor, device

    img = torch.Tensor(img).permute(2, 0, 1)

    encoded_input = img_processor(img, return_tensors="pt")

    pixel_values = encoded_input.pixel_values.to(device)
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    pred_seg = img_processor.post_process_semantic_segmentation(
        outputs,
        target_sizes=[img.size()[-2:]]
    )
    pred_seg = pred_seg[0]

    img_w_map = draw_segmentation_masks(img, pred_seg.to(torch.bool), colors="green", alpha=0.6)

    img_w_map = img_w_map.permute(1, 2, 0).numpy().astype('uint8')
    return pred_seg, img_w_map


def generate_frames(threshold):
    # Open the IP camera stream
    cap = cv2.VideoCapture("rtsp://192.168.1.14:61112/h264.sdp",
                            cv2.CAP_FFMPEG)  # Explicitly use FFMPEG backend

    # try setting buffer size and timeouts
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    if not cap.isOpened():
        print("Error: Unable to open video stream")
        return

    while True:
        start_time = time.time()
        # frame is already numpy.ndarray
        success, frame = cap.read()
        if not success:
            break

        segmentaion_map, frame = segment_door(frame)

        cv2.putText(frame, f"Threshold: {threshold:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)


        observed_area = segmentaion_map.sum()
        # prevent zero division by adding 1e-4
        ratio = observed_area / (normal_area + 1e-4)
        if ratio < threshold:
            cv2.putText(frame, "Something blocking the way!", (10, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 250), 2)
        else:
            cv2.putText(frame, "Nothing blocking the way", (10, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)

        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        # Yield frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # FPS limiting: calculate elapsed time and sleep if needed
        elapsed = time.time() - start_time
        sleep_time = FRAME_DURATION - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


@app.route('/video_feed')
def video_feed():
    # Stream the video feed as an MJPEG stream
    threshold = float(request.args.get("threshold", 0.7))
    return Response(generate_frames(threshold),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/remember_area', methods=['POST'])
def remember_area_route():
    global normal_area

    cap = cv2.VideoCapture("rtsp://192.168.1.14:61112/h264.sdp")
    success, frame = cap.read()
    cap.release()

    # calculate area of the door on a current image
    segmentaion_map, _ = segment_door(frame)
    normal_area = segmentaion_map.sum().item()
    print(normal_area)
    return jsonify({"normal_area": normal_area})


@app.route('/')
def index():
    # HTML template with a threshold slider, "Remember Area" button, and video feed.
    return render_template_string('''
        <html>
          <head>
            <title>IP Cam Segmentation Demo with Threshold</title>
          </head>
          <body>
            <h1>IP Cam Segmentation Demo with Threshold</h1>
            <label for="thresholdSlider">Threshold: <span id="thresholdValue">0.70</span></label>
            <br/>
            <input type="range" id="thresholdSlider" min="0" max="1" step="0.01" value="0.70" onchange="updateThreshold()">
            <br/><br/>
            <button onclick="rememberArea()">Remember Area</button>
            <p>Normal Area: <span id="normalAreaDisplay">None</span></p>
            <br/>
            <img id="videoFeed" src="/video_feed?threshold=0.70" style="max-width: 100%;">
            <script>
              function updateThreshold(){
                var slider = document.getElementById("thresholdSlider");
                var threshold = slider.value;
                document.getElementById("thresholdValue").innerText = parseFloat(threshold).toFixed(2);
                // Update the video feed with the new threshold parameter.
                var videoFeed = document.getElementById("videoFeed");
                videoFeed.src = "/video_feed?threshold=" + threshold;
              }
              function rememberArea(){
                var slider = document.getElementById("thresholdSlider");
                var threshold = slider.value;
                // Send a POST request to store the threshold value as the normal_area.
                fetch('/remember_area', {
                  method: 'POST',
                  headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                  },
                  body: 'threshold=' + threshold
                })
                .then(response => response.json())
                .then(data => {
                  document.getElementById("normalAreaDisplay").innerText = data.normal_area;
                });
              }
            </script>
          </body>
        </html>
    ''')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

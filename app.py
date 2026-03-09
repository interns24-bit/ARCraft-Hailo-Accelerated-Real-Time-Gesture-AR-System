import os
import base64
import requests
import face_recognition
from flask import Flask, render_template, request, jsonify, Response
import rust_eye
import edge_tts
import asyncio
import uuid
import threading
import time
import cv2
import numpy as np

app = Flask(__name__)
camera = rust_eye.Camera()

latest_frame = None
frame_lock = threading.Lock()

# ---------------- CAMERA THREAD ----------------
def capture_loop():
    global latest_frame
    while True:
        frame = bytes(camera.get_frame())
        if frame:
            with frame_lock:
                latest_frame = frame
        time.sleep(0.03)

threading.Thread(target=capture_loop, daemon=True).start()
# ------------------------------------------------

os.makedirs("static/audio", exist_ok=True)

# ---------------- FACE RECOGNITION ----------------
suman_encoding = None
try:
    suman_image = face_recognition.load_image_file("static/faces/suman.jpg")
    suman_encoding = face_recognition.face_encodings(suman_image)[0]
    print("Suman's face loaded successfully!")
except:
    print("WARNING: Could not load static/faces/suman.jpg")
# --------------------------------------------------


# ---------------- OLLAMA CALL ----------------
def ask_ollama(prompt, system_prompt, image_bytes=None):

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    if image_bytes:
        messages[1]["images"] = [
            base64.b64encode(image_bytes).decode("utf-8")
        ]

    payload = {
        "model": "moondream",
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 80
        }
    }

    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json=payload,
            timeout=60
        )

        data = response.json()
        reply = data.get("message", {}).get("content", "")

        if reply.strip() == "":
            reply = "Sorry, I couldn't think of a response."

        return reply

    except Exception as e:
        print("OLLAMA ERROR:", e)
        return "My local AI brain is not responding."
# ------------------------------------------------


# ---------------- VIDEO STREAM ----------------
def gen_frames():
    while True:
        with frame_lock:
            frame = latest_frame

        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        time.sleep(0.03)


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
# ------------------------------------------------


@app.route('/')
def index():
    return render_template('index.html')


# ---------------- INTERACTION ----------------
@app.route('/interact', methods=['POST'])
def interact():

    user_text = request.json.get("text", "")

    with frame_lock:
        current_frame = latest_frame

    is_suman = False

    # ---------- FACE CHECK ----------
    if current_frame:

        temp_path = "temp_check.jpg"
        with open(temp_path, "wb") as f:
            f.write(current_frame)

        unknown_image = face_recognition.load_image_file(temp_path)
        encodings = face_recognition.face_encodings(unknown_image)

        if encodings and suman_encoding is not None:
            is_suman = face_recognition.compare_faces(
                [suman_encoding], encodings[0])[0]

    # ---------- SYSTEM PERSONALITY ----------
    if is_suman:

        sys_prompt = """
You are Suman's personal AR assistant.

Personality:
- Very friendly
- Warm
- Slightly playful
- Happy to see Suman

Rules:
- Maximum 2 sentences
- Speak naturally
- If greeting, you may use [WAVE]
- If agreeing use [NOD]

Example tone:
"Hey Suman! Good to see you again. [WAVE]"
"""

        mood = "happy"

    else:

        sys_prompt = """
You are a strict AR security assistant.

Personality:
- Cold
- Suspicious
- Short answers
- Protective of Suman

Rules:
- Maximum 5 words
- Ask where Suman is
- Use [SHAKE] if refusing

Example tone:
"You're not Suman. Where is he?"
"""

        mood = "sad"

    # ---------- DETECT IF VISION NEEDED ----------
    trigger_words = [
        "see",
        "look",
        "camera",
        "what is",
        "what do you see",
        "identify",
        "wearing",
        "color",
        "this"
    ]

    needs_vision = any(word in user_text.lower() for word in trigger_words)

    # ---------- PREPARE IMAGE ----------
    image_for_ai = None

    if needs_vision and current_frame:

        img_array = np.frombuffer(current_frame, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # smaller image = much faster inference
        small = cv2.resize(frame, (224, 224))

        _, compressed = cv2.imencode(
            ".jpg",
            small,
            [int(cv2.IMWRITE_JPEG_QUALITY), 60]
        )

        image_for_ai = compressed.tobytes()

    # ---------- AI RESPONSE ----------
    reply = ask_ollama(user_text, sys_prompt, image_for_ai)

    # ---------- PARSE ACTION ----------
    action = "none"

    if "[WAVE]" in reply:
        action = "wave"
        reply = reply.replace("[WAVE]", "")

    elif "[NOD]" in reply:
        action = "nod"
        reply = reply.replace("[NOD]", "")

    elif "[SHAKE]" in reply:
        action = "shake"
        reply = reply.replace("[SHAKE]", "")

    # ---------- TTS ----------
    audio_filename = f"static/audio/{uuid.uuid4().hex}.mp3"

    async def create_audio(text, voice):
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(audio_filename)

    voice_type = "en-US-AnaNeural" if is_suman else "en-GB-SoniaNeural"

    asyncio.run(create_audio(reply.strip(), voice_type))

    return jsonify({
        "reply": reply.strip(),
        "mood": mood,
        "action": action,
        "audio": audio_filename
    })


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        threaded=True,
        ssl_context='adhoc'
    )

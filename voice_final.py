import cv2
import speech_recognition as sr
import time
import threading
import io
import base64
import numpy as np
from PIL import Image
from together import Together
from dotenv import load_dotenv
import os
import pyttsx3
# Global state variables
capture_mode = False
imagecaptured=False
last_query_time = 0
queries = []
captured_image = None
load_dotenv()
client = Together(api_key=os.getenv('TOGETHER_API_KEY')) 


tts_engine = pyttsx3.init() 

def speak_text(text):
    tts_engine.say(text)
    tts_engine.runAndWait()
# Dummy ML model function
def process_image_and_query(image, question):
    # Replace this with your actual ML model logic
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()



    try:
        # Convert bytes to a NumPy array for OpenCV handling
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Invalid image format or corrupted file.")

        # Convert OpenCV image to PIL
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Encode image to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Create message payload
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", "text": question}
                ]
            }
        ]

        print("🚀 Sending image & query to Together API...")

        # API call
        response = client.chat.completions.create(
            model="meta-llama/Llama-Vision-Free",
            messages=message,
            max_tokens=200,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["<|eot_id|>", "<|eom_id|>"],
            stream=True
        )

        # Process response
        answer = ""
        for token in response:
            try:
                if hasattr(token, 'choices') and token.choices:
                    content = token.choices[0].delta.content
                    if content:
                        answer += content
            except (IndexError, AttributeError):
                continue

        return answer.strip() if answer else "No valid response received."

    except Exception as e:
        print(f"❌ Error processing image: {str(e)}")
        return f"Error: {str(e)}"




    return f"[ML Model Response] Query: '{query}' processed for the captured image."

# Voice recognition function
def recognize_speech():
    global capture_mode, last_query_time, queries, captured_image

    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)

    print("Voice recognition started... Say 'capture' to take a picture.")

    while True:
        with mic as source:
            try:
                print("Listening...")
                speak_text("I am listening")
                audio = recognizer.listen(source, timeout=5)
                command = recognizer.recognize_google(audio).lower()
                print(f"You said: {command}")

                if "capture" in command:
                    capture_mode = True
                    queries = []
                    print("[INFO] Capture command received. Waiting for query...")

                elif capture_mode:
                    queries.append(command)
                    last_query_time = time.time()

            except sr.WaitTimeoutError:
                continue
            # except sr.UnknownValueError:
                # print("Didn't catch that. Try again.")
            except Exception as e:
                print(f"Speech recognition error: {e}")

# Timer thread to detect pause in queries
def query_monitor():
    global last_query_time, queries, captured_image

    while True:
        time.sleep(0.5)
        if capture_mode and queries and (time.time() - last_query_time > 2):
            full_query = " ".join(queries)
            print(f"[INFO] Processing query: {full_query}")
            result = process_image_and_query(captured_image, full_query)
            print(result)
            speak_text(result)
            queries.clear()

# Camera stream



def camera_stream():
    global captured_image, capture_mode

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow("Live Feed - Say 'capture' to take image", frame)

        if capture_mode:
            captured_image = frame.copy()
            # print("[INFO] Image captured.")
            imagecaptured= True ; 
            time.sleep(1)  # small delay to avoid multiple captures
            capture_mode=True
            continue

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def ifiamgecaptured():
    if(imagecaptured):
        print("[INFO] Image captured.")

# Run threads
if __name__ == "__main__":
    threading.Thread(target=recognize_speech, daemon=True).start()
    threading.Thread(target=query_monitor, daemon=True).start()
    threading.Thread(target=ifiamgecaptured, daemon=True).start()
    camera_stream()
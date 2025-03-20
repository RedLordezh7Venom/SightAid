import cv2
import base64
import os
import io
import threading
from PIL import Image
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from together import Together
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
client = Together(api_key=os.getenv('TOGETHER_API_KEY'))

# Initialize video capture (0 is usually the default webcam)
cam = cv2.VideoCapture(0)

def capture_frame():
    """Capture a frame from the video stream."""
    ret, frame = cam.read()
    if not ret:
        raise RuntimeError("Failed to capture image")
    return frame

def process_image_and_ask_question(frame, question):
    """Process the image and ask a question."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    
    # Encode image to base64
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Create message with image and question
    message = [
        {
            "role": "user", 
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                {"type": "text", "text": question}
            ]
        }
    ]
    
    # Call the API
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-Vision-Free",
            messages=message,
            max_tokens=None,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["<|eot_id|>","<|eom_id|>"],
            stream=True
        )
        
        # Process streaming response
        answer = ""
        for token in response:
            if hasattr(token, 'choices') and token.choices:
                try:
                    content = token.choices[0].delta.content
                    if content:
                        answer += content
                except (IndexError, AttributeError):
                    continue
        return answer
    except Exception as e:
        return f"Error: {str(e)}"

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>Image Question</title>
        </head>
        <body>
            <h1>Ask a Question About the Image</h1>
            <form action="/ask" method="post">
                <label for="question">Question:</label>
                <input type="text" id="question" name="question" required>
                <input type="submit" value="Submit">
            </form>
        </body>
    </html>
    """

@app.post("/ask")
async def ask_question(question: str = Form(...)):
    frame = capture_frame()
    answer = process_image_and_ask_question(frame, question)
    return {"question": question, "answer": answer}

@app.on_event("shutdown")
def shutdown_event():
    cam.release()
    cv2.destroyAllWindows() 
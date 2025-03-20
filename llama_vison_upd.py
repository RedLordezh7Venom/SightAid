import cv2
import base64
import os
import io
import numpy as np

import threading
from PIL import Image
from together import Together
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Together client with API key
client = Together(api_key=os.getenv('TOGETHER_API_KEY'))  # Ensure the correct variable name in .env

def process_image(image, question):
    """Convert an image to base64 and ask a question about it."""
    try:
        # Convert BGR to RGB if it's a NumPy array (Live capture)
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
        else:
            # Load image from path (Uploaded image)
            pil_image = Image.open(image)

        # Encode image to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

        question = input("\nAsk a question about the image: ")

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
        print("\nResponse: ", end="", flush=True)
        for token in response:
            if hasattr(token, 'choices') and token.choices:
                try:
                    content = token.choices[0].delta.content
                    if content:
                        print(content, end='', flush=True)
                        answer += content
                except (IndexError, AttributeError):
                    continue
        print("\n\nFull Answer:", answer)

    except Exception as e:
        print(f"Error: {str(e)}")

def capture_image():
    """Capture an image from webcam and process it."""
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open video capture device")
        return

    print("Press 'c' to capture, 'q' to quit.")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow('Video Preview', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            threading.Thread(target=process_image, args=(frame, "any")).start()
            print("\nPress 'q' to quit, 'c' to capture again")
        elif key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

def upload_image_from_path():
    """Upload a predefined image file and process it."""
    image_path = 'captured_frame.jpg'
    if os.path.exists(image_path):
        print(f"Uploading image: {image_path}")
        process_image(image_path, "any")
    else:
        print("Error: Image file not found!")

def main():
    while True:
        print("\nChoose an option:")
        print("1. Capture Live Image")
        print("2. Upload Predefined Image")
        print("3. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            capture_image()  # Capture from webcam
        elif choice == "2":
            upload_image_from_path()  # Upload the given image
        elif choice == "3":
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    main()
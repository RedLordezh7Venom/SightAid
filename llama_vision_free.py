import cv2
import base64
import os
import io
import threading
from PIL import Image
from together import Together
from dotenv import load_dotenv

load_dotenv()

client = Together(api_key=os.getenv('TOGETHER_API_KEY'))

def capture_frame_and_ask(frame, question):
    """Save a frame, encode it, and ask a question about it."""
    # Save the captured frame as an image file
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    
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
    
    #apicall
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
        
        #process response
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

def main():
    
    cam = cv2.VideoCapture(0)
    
    if not cam.isOpened():
        print("Error: Could not open video capture device")
        return
    
    print("Press 'q' to quit, 'c' to capture and ask a question")
    
    while True:
        # Capture frame for preview
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Display the preview
        cv2.imshow('Video Preview', frame)
        
        #key press
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Capture and ask
            question  = 'any'
            # Start a new thread for the capture and ask process
            threading.Thread(target=capture_frame_and_ask, args=(frame,question)).start()
            print("\nPress 'q' to quit, 'c' to capture again")
    
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
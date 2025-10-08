import cv2
import base64
import os
import io
import numpy as np
import threading
from PIL import Image
from together import Together
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM  # Base class for custom LLM integration

# Load environment variables
load_dotenv()

# Initialize Together client with API key
client = Together(api_key=os.getenv('TOGETHER_API_KEY'))

# Custom LLM wrapper for Together API
class TogetherLLM(LLM):
    def _call(self, prompt: str, stop=None, run_manager=None) -> str:
        response = client.chat.completions.create(
            model="meta-llama/Llama-Vision-Free",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=stop,
            stream=False
        )
        return response.choices[0].message.content

    @property
    def _llm_type(self) -> str:
        return "together"

# Initialize custom LLM
together_llm = TogetherLLM()

# Define LangChain prompts for each type of image
medical_prompt = PromptTemplate(
    input_variables=["image_description"],
    template="You are a medical assistant. Analyze the image and provide the following details: "
            "1. Name of the medicine. "
            "2. Recommended dosage. "
            "3. Tips or warnings. "
            "Image description: {image_description}"
)

bill_prompt = PromptTemplate(
    input_variables=["image_description"],
    template="You are a document assistant. Analyze the image and provide the following details: "
            "1. Type of document (e.g., bill, invoice, receipt). "
            "2. Key information (e.g., total amount, date, vendor). "
            "Image description: {image_description}"
)

food_prompt = PromptTemplate(
    input_variables=["image_description"],
    template="You are a nutritionist. Analyze the image and provide the following details: "
            "1. Name of the food item. "
            "2. Approximate calories. "
            "3. Nutritional tips. "
            "Image description: {image_description}"
)

generic_prompt = PromptTemplate(
    input_variables=["image_description"],
    template="Describe the image in detail and provide any relevant information. "
            "Image description: {image_description}"
)

# Initialize LangChain chains
medical_chain = LLMChain(llm=together_llm, prompt=medical_prompt, output_key="medical_output")
bill_chain = LLMChain(llm=together_llm, prompt=bill_prompt, output_key="bill_output")
food_chain = LLMChain(llm=together_llm, prompt=food_prompt, output_key="food_output")
generic_chain = LLMChain(llm=together_llm, prompt=generic_prompt, output_key="generic_output")

def conditional_chain(image_description):
    """Determine which chain to execute based on the image description."""
    if "medical" in image_description.lower() or "medicine" in image_description.lower():
        return medical_chain.run(image_description=image_description)
    elif "bill" in image_description.lower() or "invoice" in image_description.lower() or "receipt" in image_description.lower():
        return bill_chain.run(image_description=image_description)
    elif "food" in image_description.lower() or "nutrition" in image_description.lower():
        return food_chain.run(image_description=image_description)
    else:
        return generic_chain.run(image_description=image_description)

def process_image(image):
    """Convert an image to base64, describe it, and allow user to ask questions."""
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

        # Step 1: Describe the image
        describe_message = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", "text": "Describe the image in detail."}
                ]
            }
        ]

        # API call to Together for image description
        describe_response = client.chat.completions.create(
            model="meta-llama/Llama-Vision-Free",
            messages=describe_message,
            max_tokens=200,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["<|eot_id|>", "<|eom_id|>"],
            stream=False
        )

        # Extract image description
        image_description = describe_response.choices[0].message.content
        print("\nImage Description:", image_description)

        # Step 2: Use LangChain to process the image description
        result = conditional_chain(image_description)
        print("\nAnalysis Result:", result)

        # Step 3: Allow user to ask questions
        while True:
            question = input("\nAsk a question about the image (or type 'exit' to quit): ")
            if question.lower() == "exit":
                break

            # Step 4: Answer the user's question
            question_message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        {"type": "text", "text": f"{question}"}
                    ]
                }
            ]

            # API call to Together for answering the question
            question_response = client.chat.completions.create(
                model="meta-llama/Llama-Vision-Free",
                messages=question_message,
                max_tokens=200,
                temperature=0.4,
                top_p=0.7,
                top_k=50,
                repetition_penalty=1,
                stop=["<|eot_id|>", "<|eom_id|>"],
                stream=False
            )

            # Extract and display the answer
            answer = question_response.choices[0].message.content
            print("\nAnswer:", answer)

    except Exception as e:
        print(f"Error: {str(e)}")

# Rest of the code (capture_image, upload_image_from_path, main) remains the same
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
            threading.Thread(target=process_image, args=(frame,)).start()  # Pass only the frame
            print("\nPress 'q' to quit, 'c' to capture again")
        elif key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

def upload_image_from_path():
    """Upload a predefined image file and process it."""
    image_path = r"C:\Users\Sarthak singh\Downloads\WhatsApp Image 2025-03-21 at 12.50.20.jpeg"
    if os.path.exists(image_path):
        print(f"Uploading image: {image_path}")
        process_image(image_path)
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
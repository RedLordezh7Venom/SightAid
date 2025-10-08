import cv2
import base64
import os
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from together import Together
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
import uvicorn

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Vision Language Model API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

async def process_image_base64(base64_image, question=None):
    """Process an image in base64 format and optionally answer a question."""
    try:
        # Step 1: Describe the image
        describe_message = [
            {
                "role": "system",
                "content": (
                    "You are an assistant for the elderly who have irreversible eye damage. "
                    "You will work as their eyes. If you detect a medicine label being read, "
                    "detail all about the medicine along with dosage and text. "
                    "If it is about some bills, read and summarize text in short, answer any query "
                    "related to the bills or documents you are given. "
                    "If it is a nutrition label, give all necessary details after extracting all text."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", "text": question if question else "Describe the image in detail."}
                ]
            }
        ]

        # API call to Together for image processing
        response = client.chat.completions.create(
            model="meta-llama/Llama-Vision-Free",
            messages=describe_message,
            max_tokens=500,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["<|eot_id|>", "<|eom_id|>"],
            stream=False
        )

        # Extract the response
        result = response.choices[0].message.content
        
        # If no specific question was asked, use LangChain to further analyze the result
        if not question:
            enhanced_result = conditional_chain(result)
            return {"description": result, "analysis": enhanced_result}
        
        return {"answer": result}

    except Exception as e:
        return {"error": str(e)}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/process")
async def process_image(file: UploadFile = File(...), query: str = Form(None)):
    """Process an uploaded image and optionally answer a query."""
    try:
        # Read image data
        contents = await file.read()
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(contents))
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Process image and get response
        result = await process_image_base64(base64_image, query)
        
        return result
    except Exception as e:
        return {"error": str(e)}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            # Process the received data
            # Here you would extract the base64 image and query
            # For simplicity, we'll assume the data is a JSON object
            
            # Send a response back
            await websocket.send_text(f"Received: {data}")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run("vlm_api:app", host="0.0.0.0", port=8000, reload=True)
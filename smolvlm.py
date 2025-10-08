import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load images
image = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")

# Initialize processor and model
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")

# Load model with 8-bit quantization
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    load_in_8bit=True,          # Enable 8-bit quantization
    device_map="auto",          # Automatically decide device mapping
    low_cpu_mem_usage=True,     # Reduce CPU memory usage during loading
)

print(f"Model loaded with quantization. Using device: {DEVICE}")
print(f"Model memory footprint: {model.get_memory_footprint() / 1024 / 1024:.2f} MB")

# Create input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Can you describe this image?"}
        ]
    },
]

# Prepare inputs
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt")
inputs = inputs.to(DEVICE)

# Generation config for better efficiency
generation_config = {
    "max_new_tokens": 500,
    "do_sample": False,        # Use greedy decoding to save computation
    "num_beams": 1,            # Disable beam search for faster inference
    "temperature": 1.0,
    "repetition_penalty": 1.1, # Slight penalty to avoid repetition
}

# Track inference time
import time
start_time = time.time()

# Generate outputs with efficient generation settings
generated_ids = model.generate(**inputs, **generation_config)

# Calculate and print inference time
inference_time = time.time() - start_time
print(f"Inference completed in {inference_time:.2f} seconds")

generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)

print("\nGenerated Description:")
print(generated_texts[0])

# Optional: Clean up to free memory
del model
torch.cuda.empty_cache() if torch.cuda.is_available() else None

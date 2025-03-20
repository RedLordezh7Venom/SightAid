from florence import FlorenceModel

# Load the model
model = FlorenceModel.load_pretrained("florence-2")

# Input video or image
input_data = "path_to_video_or_image"

# Object detection
detections = model.detect_objects(input_data, prompt="<OD>")
print(detections)

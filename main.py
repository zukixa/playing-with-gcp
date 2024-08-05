import os
import logging
import google.generativeai as genai
from google.cloud import vision
from google.colab import userdata
from IPython.display import Markdown

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_gemini_api():
    api_key = userdata.get('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("API key is not set. Please set the GOOGLE_API_KEY in Colab.")
    genai.configure(api_key=api_key)
    logging.info("Gemini API configured successfully")

# Function to detect labels in an image using Google Cloud Vision
def detect_labels(image_path):
    client = vision.ImageAnnotatorClient()
    labels = []
    try:
        with open(image_path, "rb") as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        response = client.label_detection(image=image)
        for label in response.label_annotations:
            labels.append(label.description)
        if response.error.message:
            logging.error(f"Error in response: {response.error.message}")
            raise Exception(f"{response.error.message}\nFor more info on error messages, check: "
                            "https://cloud.google.com/apis/design/errors")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    return labels

# Function to upload images
def upload_images(image_paths):
    uploaded_files = []
    for image_path in image_paths:
        uploaded_file = genai.upload_file(path=image_path, display_name=os.path.basename(image_path))
        uploaded_files.append(uploaded_file)
        logging.info(f"Uploaded file '{uploaded_file.display_name}' as: {uploaded_file.uri}")
    return uploaded_files

# Function to verify uploaded files
def verify_files(uploaded_files):
    for file in uploaded_files:
        retrieved_file = genai.get_file(name=file.name)
        logging.info(f"Retrieved file '{retrieved_file.display_name}' as: {retrieved_file.uri}")

# Function to prompt the Gemini API with images
def prompt_with_images(images, text_prompt, model_name="gemini-1.5-pro-latest"):
    model = genai.GenerativeModel(model_name=model_name)
    response = model.generate_content([*images, text_prompt])
    print(Markdown(">" + response.text))

# Main script
if __name__ == "__main__":
    setup_gemini_api()

    image_files = ['./image1.jpg', './image2.jpg', './image3.jpg']
    detected_labels = [detect_labels(image) for image in image_files]
    for i, labels in enumerate(detected_labels):
        print(f"Labels for {image_files[i]}: {', '.join(labels)}")

    uploaded_images = upload_images(image_files)
    verify_files(uploaded_images)

    # Prompting with images
    prompt_with_images(uploaded_images, "Describe what you can see in these images.")



import os
import base64
import json
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from mistralai.client import MistralClient
from pydantic import BaseModel
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

# Note: ImageURLChunk and TextChunk are now constructed as dicts within the message list,
# so they are no longer imported directly for chat completion.

# These classes are used for submitting documents to the OCR-specific endpoint.
from mistralai.models.ocr import DocumentURLChunk, ImageURLChunk as OCRImageURLChunk

# Load environment variables from .env file
load_dotenv()

# Initialize Flask App and CORS
app = Flask(__name__)
CORS(app)

# --- Configuration ---
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY environment variable not set!")

# Set up a temporary upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize Mistral Client
client = MistralClient(api_key=MISTRAL_API_KEY)

# --- Pydantic Model for Custom Structured Output ---
class StructuredOCR(BaseModel):
    file_name: str
    topics: list[str]
    languages: str
    ocr_contents: dict

# --- API Endpoints ---
@app.route('/')
def index():
    """Render the HTML frontend."""
    return render_template('index.html')

def _handle_file_upload():
    """Helper function to handle file uploads and check for errors."""
    if 'file' not in request.files:
        return None, jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return None, jsonify({"error": "No file selected"}), 400
    if not file:
        return None, jsonify({"error": "Invalid file"}), 400
    return file, None, None

@app.route('/ocr/pdf', methods=['POST'])
def ocr_pdf():
    """
    Accepts a PDF file, performs OCR, and returns the result in JSON format.
    """
    file, error_response, status_code = _handle_file_upload()
    if error_response:
        return error_response, status_code

    filepath = ""
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        pdf_file = Path(filepath)

        uploaded_file = client.files.upload(
            file={"file_name": pdf_file.stem, "content": pdf_file.read_bytes()},
            purpose="ocr",
        )
        signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
        pdf_response = client.ocr.process(
            document=DocumentURLChunk(document_url=signed_url.url),
            model="mistral-ocr-latest"
        )
        
        os.remove(filepath)
        response_dict = json.loads(pdf_response.model_dump_json())
        return jsonify(response_dict)

    except Exception as e:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
        app.logger.error(f"Error in /ocr/pdf: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/ocr/image', methods=['POST'])
def ocr_image():
    """
    Accepts an image file, performs OCR, and returns the result in JSON format.
    """
    file, error_response, status_code = _handle_file_upload()
    if error_response:
        return error_response, status_code

    try:
        encoded_image = base64.b64encode(file.read()).decode()
        mime_type = file.mimetype or 'image/jpeg'
        base64_data_url = f"data:{mime_type};base64,{encoded_image}"

        image_response = client.ocr.process(
            document=OCRImageURLChunk(image_url=base64_data_url),
            model="mistral-ocr-latest"
        )
        
        response_dict = json.loads(image_response.model_dump_json())
        return jsonify(response_dict)

    except Exception as e:
        app.logger.error(f"Error in /ocr/image: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/ocr/structured/image', methods=['POST'])
def structured_ocr_image():
    """
    Accepts an image, performs OCR, and uses another model (vision or text)
    to extract a structured JSON response.
    """
    file, error_response, status_code = _handle_file_upload()
    if error_response:
        return error_response, status_code
        
    model_type = request.args.get('model_type', 'vision')
    if model_type not in ['vision', 'text']:
        return jsonify({"error": "Invalid model_type. Choose 'vision' or 'text'."}), 400

    try:
        file_bytes = file.read()
        encoded_image = base64.b64encode(file_bytes).decode()
        mime_type = file.mimetype or 'image/jpeg'
        base64_data_url = f"data:{mime_type};base64,{encoded_image}"

        ocr_response = client.ocr.process(
            document=OCRImageURLChunk(image_url=base64_data_url),
            model="mistral-ocr-latest"
        )
        image_ocr_markdown = ocr_response.pages[0].markdown

        prompt = (
            f"This is an image's OCR in markdown:\n\n{image_ocr_markdown}\n\n"
            "Convert this into a sensible structured json response. "
            "The output should be strictly be json with no extra commentary."
        )

        messages = []
        if model_type == 'vision':
            model = "pixtral-12b-latest"
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": base64_data_url, "detail": "high"}},
                    {"type": "text", "text": prompt},
                ],
            })
        else:
            model = "ministral-8b-latest"
            messages.append({"role": "user", "content": prompt})

        chat_response = client.chat.complete(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
        )

        response_dict = json.loads(chat_response.choices[0].message.content)
        return jsonify(response_dict)
        
    except Exception as e:
        app.logger.error(f"Error in /ocr/structured/image: {e}")
        return jsonify({"error": str(e)}), 500
        
@app.route('/ocr/custom_structured/image', methods=['POST'])
def custom_structured_ocr_image():
    """
    Accepts an image, performs OCR, and uses chat.parse to enforce a custom
    Pydantic schema on the output.
    """
    file, error_response, status_code = _handle_file_upload()
    if error_response:
        return error_response, status_code
    
    try:
        file_bytes = file.read()
        filename = secure_filename(file.filename)
        encoded_image = base64.b64encode(file_bytes).decode()
        mime_type = file.mimetype or 'image/jpeg'
        base64_data_url = f"data:{mime_type};base64,{encoded_image}"

        ocr_response = client.ocr.process(
            document=OCRImageURLChunk(image_url=base64_data_url),
            model="mistral-ocr-latest"
        )
        image_ocr_markdown = ocr_response.pages[0].markdown

        prompt = (
            f"This is the image's OCR in markdown:\n{image_ocr_markdown}\n\n"
            f"Convert this into a structured JSON response. "
            f"For the `file_name`, use the following string: '{Path(filename).stem}'. "
            f"For `ocr_contents`, create a sensible dictionary."
        )

        chat_response = client.chat.parse(
            model="pixtral-12b-latest",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": base64_data_url, "detail": "high"}},
                    {"type": "text", "text": prompt}
                ]
            }],
            response_format=StructuredOCR,
            temperature=0
        )
        
        structured_response = chat_response.choices[0].message.parsed
        response_dict = json.loads(structured_response.model_dump_json())
        return jsonify(response_dict)

    except Exception as e:
        app.logger.error(f"Error in /ocr/custom_structured/image: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
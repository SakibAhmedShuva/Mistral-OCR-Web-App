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

# --- Corrected Import Paths ---
# These classes are used for constructing chat messages with vision models.
from mistralai.models.messages import ImageURLChunk, TextChunk
# These classes are used for submitting documents to the OCR endpoint.
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

    try:
        # Save the file temporarily to read its bytes
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        pdf_file = Path(filepath)

        # 1. Upload the file to Mistral's service
        uploaded_file = client.files.upload(
            file={"file_name": pdf_file.stem, "content": pdf_file.read_bytes()},
            purpose="ocr",
        )

        # 2. Get a temporary signed URL for the uploaded file
        signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)

        # 3. Process the PDF with OCR
        pdf_response = client.ocr.process(
            document=DocumentURLChunk(document_url=signed_url.url),
            model="mistral-ocr-latest"
        )
        
        # Clean up the uploaded file
        os.remove(filepath)
        
        # Convert Pydantic model to JSON-serializable dict
        response_dict = json.loads(pdf_response.model_dump_json())
        return jsonify(response_dict)

    except Exception as e:
        # Clean up in case of error
        if 'filepath' in locals() and os.path.exists(filepath):
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
        # Encode image as a base64 data URL
        encoded_image = base64.b64encode(file.read()).decode()
        mime_type = file.mimetype or 'image/jpeg'
        base64_data_url = f"data:{mime_type};base64,{encoded_image}"

        # Process image with OCR
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
        # 1. OCR the image
        file_bytes = file.read()
        encoded_image = base64.b64encode(file_bytes).decode()
        mime_type = file.mimetype or 'image/jpeg'
        base64_data_url = f"data:{mime_type};base64,{encoded_image}"

        ocr_response = client.ocr.process(
            document=OCRImageURLChunk(image_url=base64_data_url),
            model="mistral-ocr-latest"
        )
        image_ocr_markdown = ocr_response.pages[0].markdown

        # 2. Prepare the prompt for the structuring model
        prompt = (
            f"This is an image's OCR in markdown:\n\n{image_ocr_markdown}\n\n"
            "Convert this into a sensible structured json response. "
            "The output should be strictly be json with no extra commentary."
        )

        # 3. Call the structuring model
        messages = []
        if model_type == 'vision':
            model = "pixtral-12b-latest"
            messages.append({
                "role": "user",
                "content": [
                    ImageURLChunk(image_url=base64_data_url),
                    TextChunk(text=prompt),
                ],
            })
        else: # text-only
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
        # 1. OCR the image
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

        # 2. Prepare prompt for chat.parse
        prompt = (
            f"This is the image's OCR in markdown:\n{image_ocr_markdown}\n\n"
            f"Convert this into a structured JSON response. "
            f"For the `file_name`, use the following string: '{Path(filename).stem}'. "
            f"For `ocr_contents`, create a sensible dictionary."
        )

        # 3. Use chat.parse with the Pydantic model
        chat_response = client.chat.parse(
            model="pixtral-12b-latest",
            messages=[{
                "role": "user",
                "content": [
                    ImageURLChunk(image_url=base64_data_url),
                    TextChunk(text=prompt)
                ]
            }],
            response_format=StructuredOCR,
            temperature=0
        )
        
        # The .parsed attribute holds the Pydantic object
        structured_response = chat_response.choices[0].message.parsed
        
        # model_dump_json converts the Pydantic object to a JSON string
        response_dict = json.loads(structured_response.model_dump_json())
        return jsonify(response_dict)

    except Exception as e:
        app.logger.error(f"Error in /ocr/custom_structured/image: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
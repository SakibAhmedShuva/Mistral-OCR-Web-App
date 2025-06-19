from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pathlib import Path
from mistralai import Mistral, DocumentURLChunk, ImageURLChunk, TextChunk
from mistralai.models import OCRResponse
from pydantic import BaseModel
import json
import base64
import os
import tempfile
import uuid
from typing import Optional, List, Dict, Any
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Mistral client
API_KEY = os.getenv('MISTRAL_API_KEY', 'YOUR_API_KEY_HERE')
client = Mistral(api_key=API_KEY)

# Pydantic models for structured responses
class StructuredOCR(BaseModel):
    file_name: str
    topics: List[str]
    languages: str
    ocr_contents: Dict[str, Any]

class OCRResult(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Utility functions
def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_pdf(filename):
    """Check if file is PDF"""
    return filename.lower().endswith('.pdf')

def is_image(filename):
    """Check if file is an image"""
    image_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in image_extensions

def replace_images_in_markdown(markdown_str: str, images_dict: Dict[str, str]) -> str:
    """Replace image placeholders in markdown with base64-encoded images"""
    for img_name, base64_str in images_dict.items():
        markdown_str = markdown_str.replace(
            f"![{img_name}]({img_name})", f"![{img_name}]({base64_str})"
        )
    return markdown_str

def get_combined_markdown(ocr_response: OCRResponse) -> str:
    """Combine OCR text and images into a single markdown document"""
    markdowns = []
    for page in ocr_response.pages:
        image_data = {}
        for img in page.images:
            image_data[img.id] = img.image_base64
        markdowns.append(replace_images_in_markdown(page.markdown, image_data))
    return "\n\n".join(markdowns)

def encode_image_to_base64(file_path: str) -> str:
    """Encode image file to base64 data URL"""
    with open(file_path, 'rb') as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
        return f"data:image/jpeg;base64,{encoded}"

# API Routes

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Mistral OCR API",
        "version": "1.0.0",
        "endpoints": {
            "pdf_ocr": "/api/pdf/ocr",
            "pdf_ocr_structured": "/api/pdf/ocr/structured",
            "image_ocr": "/api/image/ocr",
            "image_ocr_structured": "/api/image/ocr/structured",
            "batch_ocr": "/api/batch/ocr"
        }
    })

@app.route('/api/pdf/ocr', methods=['POST'])
def pdf_ocr():
    """Process PDF file with OCR"""
    try:
        if 'file' not in request.files:
            return jsonify(OCRResult(
                success=False,
                message="No file provided",
                error="File is required"
            ).dict()), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify(OCRResult(
                success=False,
                message="No file selected",
                error="File selection is required"
            ).dict()), 400

        if not (file and allowed_file(file.filename) and is_pdf(file.filename)):
            return jsonify(OCRResult(
                success=False,
                message="Invalid file type",
                error="Only PDF files are allowed"
            ).dict()), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        try:
            # Upload to Mistral OCR service
            pdf_file = Path(file_path)
            uploaded_file = client.files.upload(
                file={
                    "file_name": pdf_file.stem,
                    "content": pdf_file.read_bytes(),
                },
                purpose="ocr",
            )

            # Get signed URL
            signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)

            # Process with OCR
            include_images = request.form.get('include_images', 'true').lower() == 'true'
            pdf_response = client.ocr.process(
                document=DocumentURLChunk(document_url=signed_url.url),
                model="mistral-ocr-latest",
                include_image_base64=include_images
            )

            # Convert to response format
            response_dict = json.loads(pdf_response.model_dump_json())
            
            # Get combined markdown if requested
            include_markdown = request.form.get('include_markdown', 'false').lower() == 'true'
            if include_markdown:
                response_dict['combined_markdown'] = get_combined_markdown(pdf_response)

            return jsonify(OCRResult(
                success=True,
                message="PDF processed successfully",
                data=response_dict
            ).dict())

        finally:
            # Clean up uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)

    except Exception as e:
        return jsonify(OCRResult(
            success=False,
            message="OCR processing failed",
            error=str(e)
        ).dict()), 500

@app.route('/api/pdf/ocr/structured', methods=['POST'])
def pdf_ocr_structured():
    """Process PDF file with OCR and return structured data"""
    try:
        if 'file' not in request.files:
            return jsonify(OCRResult(
                success=False,
                message="No file provided",
                error="File is required"
            ).dict()), 400

        file = request.files['file']
        if not (file and allowed_file(file.filename) and is_pdf(file.filename)):
            return jsonify(OCRResult(
                success=False,
                message="Invalid file type",
                error="Only PDF files are allowed"
            ).dict()), 400

        # Save and process file (similar to pdf_ocr)
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        try:
            # Process with OCR
            pdf_file = Path(file_path)
            uploaded_file = client.files.upload(
                file={
                    "file_name": pdf_file.stem,
                    "content": pdf_file.read_bytes(),
                },
                purpose="ocr",
            )

            signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
            pdf_response = client.ocr.process(
                document=DocumentURLChunk(document_url=signed_url.url),
                model="mistral-ocr-latest",
                include_image_base64=False
            )

            # Get OCR markdown
            combined_markdown = get_combined_markdown(pdf_response)

            # Get structured response
            structure_prompt = request.form.get('structure_prompt', 
                "Convert this OCR content into a structured JSON response with meaningful organization.")
            
            chat_response = client.chat.complete(
                model="pixtral-12b-latest",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            TextChunk(text=f"This is OCR content in markdown:\n\n{combined_markdown}\n\n{structure_prompt}")
                        ],
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )

            structured_data = json.loads(chat_response.choices[0].message.content)

            return jsonify(OCRResult(
                success=True,
                message="PDF processed and structured successfully",
                data={
                    "original_ocr": json.loads(pdf_response.model_dump_json()),
                    "structured_data": structured_data
                }
            ).dict())

        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    except Exception as e:
        return jsonify(OCRResult(
            success=False,
            message="Structured OCR processing failed",
            error=str(e)
        ).dict()), 500

@app.route('/api/image/ocr', methods=['POST'])
def image_ocr():
    """Process image file with OCR"""
    try:
        if 'file' not in request.files:
            return jsonify(OCRResult(
                success=False,
                message="No file provided",
                error="File is required"
            ).dict()), 400

        file = request.files['file']
        if not (file and allowed_file(file.filename) and is_image(file.filename)):
            return jsonify(OCRResult(
                success=False,
                message="Invalid file type",
                error="Only image files are allowed"
            ).dict()), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        try:
            # Encode image to base64
            base64_data_url = encode_image_to_base64(file_path)

            # Process with OCR
            image_response = client.ocr.process(
                document=ImageURLChunk(image_url=base64_data_url),
                model="mistral-ocr-latest"
            )

            response_dict = json.loads(image_response.model_dump_json())

            return jsonify(OCRResult(
                success=True,
                message="Image processed successfully",
                data=response_dict
            ).dict())

        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    except Exception as e:
        return jsonify(OCRResult(
            success=False,
            message="Image OCR processing failed",
            error=str(e)
        ).dict()), 500

@app.route('/api/image/ocr/structured', methods=['POST'])
def image_ocr_structured():
    """Process image file with OCR and return structured data using Pydantic model"""
    try:
        if 'file' not in request.files:
            return jsonify(OCRResult(
                success=False,
                message="No file provided",
                error="File is required"
            ).dict()), 400

        file = request.files['file']
        if not (file and allowed_file(file.filename) and is_image(file.filename)):
            return jsonify(OCRResult(
                success=False,
                message="Invalid file type",
                error="Only image files are allowed"
            ).dict()), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        try:
            # Process with structured OCR
            structured_response = structured_ocr(file_path)
            response_dict = json.loads(structured_response.model_dump_json())

            return jsonify(OCRResult(
                success=True,
                message="Image processed and structured successfully",
                data=response_dict
            ).dict())

        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    except Exception as e:
        return jsonify(OCRResult(
            success=False,
            message="Structured image OCR processing failed",
            error=str(e)
        ).dict()), 500

@app.route('/api/image/ocr/custom', methods=['POST'])
def image_ocr_custom():
    """Process image with custom structured output using text-only model"""
    try:
        if 'file' not in request.files:
            return jsonify(OCRResult(
                success=False,
                message="No file provided",
                error="File is required"
            ).dict()), 400

        file = request.files['file']
        if not (file and allowed_file(file.filename) and is_image(file.filename)):
            return jsonify(OCRResult(
                success=False,
                message="Invalid file type",
                error="Only image files are allowed"
            ).dict()), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        try:
            # First get OCR
            base64_data_url = encode_image_to_base64(file_path)
            image_response = client.ocr.process(
                document=ImageURLChunk(image_url=base64_data_url),
                model="mistral-ocr-latest"
            )

            image_ocr_markdown = image_response.pages[0].markdown

            # Get structured response using text-only model
            structure_prompt = request.form.get('structure_prompt', 
                "Convert this into a sensible structured json response. The output should be strictly JSON with no extra commentary.")
            
            chat_response = client.chat.complete(
                model="ministral-8b-latest",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            TextChunk(text=f"This is image's OCR in markdown:\n\n{image_ocr_markdown}\n\n{structure_prompt}")
                        ],
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )

            structured_data = json.loads(chat_response.choices[0].message.content)

            return jsonify(OCRResult(
                success=True,
                message="Image processed with custom structure successfully",
                data={
                    "original_ocr": json.loads(image_response.model_dump_json()),
                    "structured_data": structured_data
                }
            ).dict())

        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    except Exception as e:
        return jsonify(OCRResult(
            success=False,
            message="Custom structured OCR processing failed",
            error=str(e)
        ).dict()), 500

@app.route('/api/batch/ocr', methods=['POST'])
def batch_ocr():
    """Process multiple files with OCR"""
    try:
        if 'files' not in request.files:
            return jsonify(OCRResult(
                success=False,
                message="No files provided",
                error="Files are required"
            ).dict()), 400

        files = request.files.getlist('files')
        if not files or len(files) == 0:
            return jsonify(OCRResult(
                success=False,
                message="No files provided",
                error="At least one file is required"
            ).dict()), 400

        results = []
        for file in files:
            if file and allowed_file(file.filename):
                try:
                    filename = secure_filename(file.filename)
                    unique_filename = f"{uuid.uuid4()}_{filename}"
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                    file.save(file_path)

                    if is_pdf(file.filename):
                        # Process PDF
                        pdf_file = Path(file_path)
                        uploaded_file = client.files.upload(
                            file={
                                "file_name": pdf_file.stem,
                                "content": pdf_file.read_bytes(),
                            },
                            purpose="ocr",
                        )
                        signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
                        ocr_response = client.ocr.process(
                            document=DocumentURLChunk(document_url=signed_url.url),
                            model="mistral-ocr-latest"
                        )
                    else:
                        # Process image
                        base64_data_url = encode_image_to_base64(file_path)
                        ocr_response = client.ocr.process(
                            document=ImageURLChunk(image_url=base64_data_url),
                            model="mistral-ocr-latest"
                        )

                    results.append({
                        "filename": filename,
                        "success": True,
                        "data": json.loads(ocr_response.model_dump_json())
                    })

                    # Clean up
                    if os.path.exists(file_path):
                        os.remove(file_path)

                except Exception as e:
                    results.append({
                        "filename": filename,
                        "success": False,
                        "error": str(e)
                    })

        return jsonify(OCRResult(
            success=True,
            message=f"Processed {len(results)} files",
            data={"results": results}
        ).dict())

    except Exception as e:
        return jsonify(OCRResult(
            success=False,
            message="Batch OCR processing failed",
            error=str(e)
        ).dict()), 500

def structured_ocr(image_path: str) -> StructuredOCR:
    """Process an image using OCR and extract structured data using Pydantic model"""
    image_file = Path(image_path)
    if not image_file.is_file():
        raise FileNotFoundError("The provided image path does not exist.")

    # Read and encode the image file
    encoded_image = base64.b64encode(image_file.read_bytes()).decode()
    base64_data_url = f"data:image/jpeg;base64,{encoded_image}"

    # Process the image using OCR
    image_response = client.ocr.process(
        document=ImageURLChunk(image_url=base64_data_url),
        model="mistral-ocr-latest"
    )
    image_ocr_markdown = image_response.pages[0].markdown

    # Parse the OCR result into a structured JSON response
    chat_response = client.chat.parse(
        model="pixtral-12b-latest",
        messages=[
            {
                "role": "user",
                "content": [
                    ImageURLChunk(image_url=base64_data_url),
                    TextChunk(text=(
                        f"This is the image's OCR in markdown:\n{image_ocr_markdown}\n\n"
                        "Convert this into a structured JSON response "
                        "with the OCR contents in a sensible dictionary."
                    ))
                ]
            }
        ],
        response_format=StructuredOCR,
        temperature=0
    )

    return chat_response.choices[0].message.parsed

@app.route('/api/image/resize', methods=['POST'])
def resize_image():
    """Resize image and return OCR results"""
    try:
        if 'file' not in request.files:
            return jsonify(OCRResult(
                success=False,
                message="No file provided",
                error="File is required"
            ).dict()), 400

        file = request.files['file']
        if not (file and allowed_file(file.filename) and is_image(file.filename)):
            return jsonify(OCRResult(
                success=False,
                message="Invalid file type",
                error="Only image files are allowed"
            ).dict()), 400

        # Get resize parameters
        scale_factor = float(request.form.get('scale_factor', 0.2))  # Default to 1/5 size
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        try:
            # Resize image
            with Image.open(file_path) as img:
                new_width = int(img.width * scale_factor)
                new_height = int(img.height * scale_factor)
                resized_img = img.resize((new_width, new_height))
                
                # Save resized image
                resized_path = file_path.replace('.', '_resized.')
                resized_img.save(resized_path)

            # Process resized image with OCR
            base64_data_url = encode_image_to_base64(resized_path)
            image_response = client.ocr.process(
                document=ImageURLChunk(image_url=base64_data_url),
                model="mistral-ocr-latest"
            )

            response_dict = json.loads(image_response.model_dump_json())
            response_dict['resize_info'] = {
                'original_size': f"{int(img.width)}x{int(img.height)}",
                'resized_size': f"{new_width}x{new_height}",
                'scale_factor': scale_factor
            }

            return jsonify(OCRResult(
                success=True,
                message="Image resized and processed successfully",
                data=response_dict
            ).dict())

        finally:
            # Clean up
            for path in [file_path, resized_path]:
                if os.path.exists(path):
                    os.remove(path)

    except Exception as e:
        return jsonify(OCRResult(
            success=False,
            message="Image resize and OCR processing failed",
            error=str(e)
        ).dict()), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify(OCRResult(
        success=False,
        message="File too large",
        error="File size exceeds maximum limit of 16MB"
    ).dict()), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify(OCRResult(
        success=False,
        message="Endpoint not found",
        error="The requested endpoint does not exist"
    ).dict()), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify(OCRResult(
        success=False,
        message="Internal server error",
        error="Something went wrong on the server"
    ).dict()), 500

if __name__ == '__main__':
    # Make sure to set your Mistral API key as an environment variable
    if API_KEY == 'YOUR_API_KEY_HERE':
        print("Warning: Please set your MISTRAL_API_KEY environment variable")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
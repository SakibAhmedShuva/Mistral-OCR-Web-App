# Mistral OCR Flask API based Web App

This project provides a comprehensive Flask API for interacting with the Mistral OCR and Chat models, demonstrating functionalities for processing PDFs and images to extract both raw text and structured JSON data.

![image](https://github.com/user-attachments/assets/a64c98c7-81e5-4086-b19f-2a1aad03cf0e)


## Features

-   **PDF OCR**: Upload a PDF and get its content in Markdown.
-   **Image OCR**: Upload an image and get its content in Markdown.
-   **Structured OCR**: Upload an image to get structured data in JSON format, using either a vision model (`pixtral-12b-latest`) or a text-only model (`ministral-8b-latest`).
-   **Custom Schema OCR**: Upload an image and extract structured JSON that conforms to a predefined Pydantic model, ensuring consistent output.
-   **Simple Frontend**: An easy-to-use web interface to test all API endpoints.
-   **CORS Enabled**: Allows for easy integration with any web frontend.


## Setup and Installation

### 1. Prerequisites

-   Python 3.8+
-   `pip` for package management

### 2. Clone the Repository

```bash
git clone <your-repo-url>
cd flask-mistral-ocr

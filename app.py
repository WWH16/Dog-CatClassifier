from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import base64
from io import BytesIO

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp'}

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# Initialize model
class CatDogMobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.mobilenet_v2(pretrained=False)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, 2)

    def forward(self, x):
        return self.model(x)


# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CatDogMobileNet()
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()
model.to(device)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def predict_image(image):
    """Predict if image is cat or dog"""
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)[0]

    cat_prob = float(probs[0].item()) * 100
    dog_prob = float(probs[1].item()) * 100

    return {
        'cat_probability': round(cat_prob, 1),
        'dog_probability': round(dog_prob, 1),
        'prediction': 'cat' if cat_prob > dog_prob else 'dog'
    }


# Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload')
def upload_page():
    return render_template('upload.html')


@app.route('/camera')
def camera_page():
    return render_template('camera.html')


@app.route('/result')
def result_page():
    return render_template('result.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for image prediction"""
    try:
        # Check if image is in request
        if 'image' not in request.files and 'image_data' not in request.form:
            return jsonify({'error': 'No image provided'}), 400

        # Handle file upload
        if 'image' in request.files:
            file = request.files['image']

            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400

            if not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type. Allowed: JPG, PNG, WebP'}), 400

            # Open image
            image = Image.open(file.stream).convert('RGB')

        # Handle base64 image (from camera)
        elif 'image_data' in request.form:
            image_data = request.form['image_data']
            # Remove data URL prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]

            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes)).convert('RGB')

        # Get prediction
        result = predict_image(image)

        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'cat_probability': result['cat_probability'],
            'dog_probability': result['dog_probability']
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': True})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
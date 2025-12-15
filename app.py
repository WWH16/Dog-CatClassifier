from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import sqlite3
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp'}
app.config['DATABASE'] = 'catdog.db'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


# Database functions
def get_db():
    db = sqlite3.connect(app.config['DATABASE'])
    db.row_factory = sqlite3.Row
    return db


def init_db():
    with app.app_context():
        db = get_db()
        db.execute('''CREATE TABLE IF NOT EXISTS users
                      (
                          id
                          INTEGER
                          PRIMARY
                          KEY
                          AUTOINCREMENT,
                          username
                          TEXT
                          UNIQUE
                          NOT
                          NULL,
                          email
                          TEXT
                          UNIQUE
                          NOT
                          NULL,
                          password
                          TEXT
                          NOT
                          NULL,
                          created_at
                          TIMESTAMP
                          DEFAULT
                          CURRENT_TIMESTAMP
                      )''')

        db.execute('''CREATE TABLE IF NOT EXISTS predictions
        (
            id
            INTEGER
            PRIMARY
            KEY
            AUTOINCREMENT,
            user_id
            INTEGER
            NOT
            NULL,
            filename
            TEXT
            NOT
            NULL,
            prediction
            TEXT
            NOT
            NULL,
            cat_probability
            REAL
            NOT
            NULL,
            dog_probability
            REAL
            NOT
            NULL,
            created_at
            TIMESTAMP
            DEFAULT
            CURRENT_TIMESTAMP,
            FOREIGN
            KEY
                      (
            user_id
                      ) REFERENCES users
                      (
                          id
                      )
            )''')
        db.commit()
        db.close()


class User(UserMixin):
    def __init__(self, id, username, email):
        self.id = id
        self.username = username
        self.email = email


@login_manager.user_loader
def load_user(user_id):
    db = get_db()
    user = db.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    db.close()
    if user:
        return User(user['id'], user['username'], user['email'])
    return None


# Model setup
class CatDogMobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.mobilenet_v2(pretrained=False)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, 2)

    def forward(self, x):
        return self.model(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CatDogMobileNet()
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()
model.to(device)

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


def save_prediction(user_id, filename, result):
    db = get_db()
    db.execute('''INSERT INTO predictions
                      (user_id, filename, prediction, cat_probability, dog_probability)
                  VALUES (?, ?, ?, ?, ?)''',
               (user_id, filename, result['prediction'],
                result['cat_probability'], result['dog_probability']))
    db.commit()
    db.close()


# Routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if not username or not email or not password:
            flash('All fields are required', 'error')
            return render_template('signup.html')

        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('signup.html')

        if len(password) < 6:
            flash('Password must be at least 6 characters', 'error')
            return render_template('signup.html')

        db = get_db()
        existing = db.execute('SELECT * FROM users WHERE username = ? OR email = ?',
                              (username, email)).fetchone()

        if existing:
            flash('Username or email already exists', 'error')
            db.close()
            return render_template('signup.html')

        hashed = generate_password_hash(password)
        db.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                   (username, email, hashed))
        db.commit()
        db.close()

        flash('Account created! Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if not username or not password:
            flash('Enter username and password', 'error')
            return render_template('login.html')

        db = get_db()
        user = db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        db.close()

        if user and check_password_hash(user['password'], password):
            user_obj = User(user['id'], user['username'], user['email'])
            login_user(user_obj)
            return redirect(url_for('dashboard'))

        flash('Invalid username or password', 'error')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully', 'success')
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard():
    db = get_db()
    predictions = db.execute('''SELECT *
                                FROM predictions
                                WHERE user_id = ?
                                ORDER BY created_at DESC LIMIT 20''',
                             (current_user.id,)).fetchall()

    stats = db.execute('''SELECT COUNT(*)                                            as total,
                                 SUM(CASE WHEN prediction = 'cat' THEN 1 ELSE 0 END) as cats,
                                 SUM(CASE WHEN prediction = 'dog' THEN 1 ELSE 0 END) as dogs
                          FROM predictions
                          WHERE user_id = ?''',
                       (current_user.id,)).fetchone()
    db.close()

    return render_template('dashboard.html', predictions=predictions, stats=stats)


@app.route('/upload')
@login_required
def upload_page():
    return render_template('upload.html')


@app.route('/batch-upload')
@login_required
def batch_upload_page():
    return render_template('batch_upload.html')


@app.route('/api/predict', methods=['POST'])
@login_required
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        image = Image.open(file.stream).convert('RGB')
        result = predict_image(image)

        # Save to database
        save_prediction(current_user.id, file.filename, result)

        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'cat_probability': result['cat_probability'],
            'dog_probability': result['dog_probability']
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch-predict', methods=['POST'])
@login_required
def batch_predict():
    try:
        files = request.files.getlist('images')

        if not files or len(files) == 0:
            return jsonify({'error': 'No images provided'}), 400

        if len(files) > 10:
            return jsonify({'error': 'Maximum 10 images allowed'}), 400

        results = []

        for file in files:
            if file and allowed_file(file.filename):
                try:
                    image = Image.open(file.stream).convert('RGB')
                    result = predict_image(image)
                    save_prediction(current_user.id, file.filename, result)

                    results.append({
                        'filename': file.filename,
                        'success': True,
                        'prediction': result['prediction'],
                        'cat_probability': result['cat_probability'],
                        'dog_probability': result['dog_probability']
                    })
                except Exception as e:
                    results.append({
                        'filename': file.filename,
                        'success': False,
                        'error': str(e)
                    })

        return jsonify({
            'success': True,
            'results': results,
            'total': len(results)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': True})


if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)
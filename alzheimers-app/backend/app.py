# ============================================
#   NeuroScan — app.py
#   Flask Backend: /predict endpoint
# ============================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from model import predict_image

# ─── App Setup ───
app = Flask(__name__)
CORS(app)  # Allow requests from frontend

# ─── Config ───
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure uploads folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ─── Helper ───
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ─── Routes ───

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'NeuroScan API is running ✅'}), 200


@app.route('/predict', methods=['POST'])
def predict():
    # 1. Check file in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded. Use key "file".'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, BMP.'}), 400

    # 2. Save the uploaded file
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(save_path)
    except Exception as e:
        return jsonify({'error': f'Failed to save file: {str(e)}'}), 500

    # 3. Run prediction
    try:
        result = predict_image(save_path)
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    # 4. Return result
    return jsonify({
        'prediction': result,
        'filename': filename
    }), 200


# ─── Error Handlers ───

@app.errorhandler(413)
def file_too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 10MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found.'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error.'}), 500


# ─── Run ───
if __name__ == '__main__':
    print("🧠 NeuroScan Flask API starting on http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)

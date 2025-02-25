import easyocr
import cv2
from flask import Flask, request, jsonify

reader = easyocr.Reader(['en'])
app = Flask(__name__)

@app.route('/ocr', methods=['POST'])
def ocr_api():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image = cv2.imdecode(
        np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR
    )
    if image is None:
        return jsonify({'error': 'Invalid image format'}), 400

    results = reader.readtext(image)
    extracted_text = [result[1] for result in results]
    return jsonify({'text': extracted_text})

if __name__ == '__main__':
    import numpy as np
    app.run(debug=True)
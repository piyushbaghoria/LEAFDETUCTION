from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
# Allow uploads up to 10 MB
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# … your model‐loading code here …

@app.route('/classify', methods=['POST'])
def classify():
    # === Make sure this 'if' has an indented block beneath it ===
    if 'file' not in request.files:
        # ← this line MUST be indented under the 'if'
        return jsonify(error="No image file"), 400

    # From here on, also indent everything inside the function
    img_bytes = request.files['file'].read()

    # Your predict() call and JSON return also need to be indented
    label, conf = predict(img_bytes)
    parts = label.split()
    crop = " ".join(parts[:-1])
    status = parts[-1]

    cures = {
        'Damaged': ('Remove damaged parts; use mild pesticide.', '…'),
        # etc.
    }
    cure_eng, cure_hin = cures.get(status, ('No recommendation', '…'))

    return jsonify(
        crop=crop,
        status=status,
        confidence=conf,
        cure_eng=cure_eng,
        cure_hin=cure_hin
    )

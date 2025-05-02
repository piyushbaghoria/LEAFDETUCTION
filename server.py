from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
# Allow uploads up to 10 MB
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024   # 10 MB

# load interpreter, labels, etc. as before...


def predict(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((150, 150))
    data = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)
    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    idx = int(np.argmax(output))
    return labels[idx], float(output[idx])


@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
    return jsonify(error="No image file"), 400
img_bytes = request.files['file'].read()


    try:
        img = request.files['image'].read()
        label, conf = predict(img)
        parts = label.split()
        crop = " ".join(parts[:-1])
        status = parts[-1]

        cures = {
            'Damaged': ('Remove damaged parts; use mild pesticide.', 'क्षतिग्रस्त हिस्सों को हटाएँ; हल्का कीटनाशक लगाएँ।'),
            'Dried': ('Water adequately; improve soil.', 'पर्याप्त पानी दें; मिट्टी सुधारें।'),
            'Unripe': ('Let it mature.', 'पकने दें।'),
            'Ripe': ('Ready for harvest.', 'कटाई के लिए तैयार।'),
            'Old': ('Discard; not for sale.', 'फेंक दें।')
        }

        cure_eng, cure_hin = cures.get(status, ('No recommendation', 'कोई सुझाव नहीं'))

        return jsonify(
            crop=crop,
            status=status,
            confidence=conf,
            cure_eng=cure_eng,
            cure_hin=cure_hin
        )
    except Exception as e:
        import traceback
        return jsonify(error="Server error", detail=str(e), traceback=traceback.format_exc()), 500

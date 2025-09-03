import os
from flask import Flask, request, render_template, send_from_directory
import cv2
import numpy as np
import pickle

# Force CPU usage (avoids CUDA/CUDA_VISIBLE_DEVICES issues on Render)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "Rice_Leaf_Disease_Detection_Model.pkl"
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

CATEGORIES = ["Bacterial leaf blight", "Brown spot", "Leaf smut"]

# Preprocess image for prediction
def preprocess_image(img_path, IMG_SIZE=120):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Upload folder inside static
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    uploaded_file = None
    dark_mode = "off"  # default light mode

    if request.method == "POST":
        # Get uploaded file
        file = request.files.get("file")
        # Get dark mode state from form
        dark_mode = request.form.get("dark_mode", "off")

        if file:
            # Save uploaded file in static/uploads
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            uploaded_file = file.filename

            # Preprocess and predict
            processed_img = preprocess_image(filepath)
            preds = model.predict(processed_img)
            prediction = CATEGORIES[np.argmax(preds)]

    return render_template(
        "index.html",
        prediction=prediction,
        uploaded_file=uploaded_file,
        dark_mode=dark_mode
    )

if __name__ == "__main__":
    # Use host=0.0.0.0 and dynamic port for Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

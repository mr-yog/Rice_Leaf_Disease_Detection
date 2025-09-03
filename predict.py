import cv2
import numpy as np
import pickle

# Load trained model
with open("Rice_Leaf_Disease_Detection_Model.pkl", "rb") as file:
    model = pickle.load(file)

# Class labels
CATEGORIES = ["Bacterial leaf blight", "Brown spot", "Leaf smut"]

# Preprocess image
def preprocess_image(img_path, IMG_SIZE=120):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0  # normalize if training used normalization
    img = np.expand_dims(img, axis=0)  # (1, 120, 120, 3)
    return img

# Predict function
def predict_disease(img_path):
    img = preprocess_image(img_path)
    prediction = model.predict(img)
    predicted_class = CATEGORIES[np.argmax(prediction)]
    return predicted_class

# Example run
if __name__ == "__main__":
    test_img = "test_leaf.jpg"  # replace with unseen image
    print("Predicted:", predict_disease(test_img))

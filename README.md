
# Rice Leaf Disease Detection Web App

This is a **Flask-based web application** that allows users to upload an image of a rice leaf and get an automatic prediction of its disease type using a trained machine learning model.

The app includes features such as:  
- Image upload and preview before prediction  
- Prediction of three types of rice leaf diseases:  
  - Bacterial leaf blight  
  - Brown spot  
  - Leaf smut  
- Dark mode toggle for a better viewing experience  
- Fully responsive and user-friendly interface  



## Demo

You can try the app live here:  
[**Deployed Web App URL**](https://your-deployed-app-link.com)



## Project Structure



Rice\_Leaf\_Disease\_Detection/
│
├── app.py                          # Flask application
├── Rice\_Leaf\_Disease\_Detection\_Model.pkl   # Trained ML model
├── requirements.txt                # Required Python packages
├── Procfile                        # Deployment config for Render/Heroku
├── .gitignore                      # Files to ignore in Git
├── templates/
│   └── index.html                  # HTML UI template
├── static/
│   ├── style.css                   # CSS file
│   ├── background.png              # Light mode background
│   └── dark-background.png         # Dark mode background
├── static/uploads/                 # Temporary folder for uploaded images
├── data/                           # (Optional) dataset folder
├── notebook/                        # Jupyter notebook for model training
└── README.md





## Installation & Setup

1. **Clone the repository:**


git clone https://github.com/mr-yog/Rice_Leaf_Disease_Detection.git
cd Rice_Leaf_Disease_Detection


2. **Create a virtual environment (optional but recommended):**


python -m venv venv
source venv/bin/activate      # Linux/macOS

venv\Scripts\activate         # Windows


3. **Install dependencies:**


pip install -r requirements.txt


4. **Run the app locally:**


python app.py


5. Open your browser at `http://127.0.0.1:5000` to access the app.



## Deployment

This app is ready to deploy on platforms like **Render**, **Railway**, or **Heroku**:

* **Render:** Use `python app.py` as the start command and `pip install -r requirements.txt` as the build command.
* **Heroku:** Uses the `Procfile` included in the repository.



## Usage

1. Open the web app.
2. Click the **Upload** button and select a rice leaf image.
3. Preview the image above the predict button.
4. Click **Predict** to see the disease type.
5. Toggle **Dark Mode** for a dark-themed interface.
6. Refresh the page to reset the uploaded image and prediction.



## Technologies Used

* Python 3
* Flask
* OpenCV
* NumPy
* TensorFlow / Keras
* HTML, CSS (for frontend design)



## Authors

* **Yogesh Suryawanshi** – Developer & ML Engineer
* **Contact:** [yogeshsuryawanshi023@gmail.com](mailto:yogeshsuryawanshi023@gmail.com)



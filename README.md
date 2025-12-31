# 👤 Age and Gender Detection using CNN

This project implements an **Age and Gender Detection system** using **Convolutional Neural Networks (CNN)**.  
The model predicts a person’s **age** and **gender** from a facial image.  
It is trained using the **UTKFace dataset** and built with **Python and TensorFlow/Keras**.

---

## 📌 Project Description

Age and Gender Detection is a **computer vision & deep learning project** that analyzes facial images and predicts:
- **Age** → Regression problem
- **Gender** → Binary classification (Male / Female)

This project demonstrates:
- Image preprocessing and normalization
- CNN-based feature extraction
- Multi-output deep learning model
- Model training, evaluation, and saving

---

## 📂 Dataset Information

- **Dataset Name:** UTKFace Dataset  
- **Source:** Kaggle  
- **Link:** https://www.kaggle.com/datasets/jangedoo/utkface-new  

### Dataset Details
- 20,000+ face images
- Filename format:

- Gender labels:
- `0` → Male
- `1` → Female

---

## 🧠 Model Architecture

- Convolutional Neural Network (CNN)
- Layers used:
- Convolution
- MaxPooling
- Dropout
- Fully Connected layers
- Outputs:
- **Age Prediction** (Regression – MSE Loss)
- **Gender Prediction** (Binary Classification – Binary Crossentropy)

---

## 🛠️ Technologies Used

- Python 🐍
- TensorFlow / Keras
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Scikit-learn
- Joblib
- Jupyter / Kaggle Notebook

---

## 📁 Project Structure

Age-Gender-Detection/
│
├── dataset/
│ └── UTKFace/
│
├── notebooks/
│ └── age_gender_training.ipynb
│
├── models/
│ └── model.pkl / model.h5
│
├── requirements.txt
├── README.md
└── app.py (optional)

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/Age-Gender-Detection.git
cd Age-Gender-Detection

2️⃣ Create Virtual Environment (Optional)
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows


3️⃣ Install Required Libraries
pip install -r requirements.txt


📦 requirements.txt
tensorflow
numpy
pandas
opencv-python
matplotlib
scikit-learn
joblib

▶️ How to Run the Project
🔹 Training the Model
jupyter notebook

📊 Results

Gender classification achieved high accuracy

Age prediction evaluated using Mean Absolute Error (MAE)

Model performs well on unseen facial images


🚀 Future Enhancements

Use Transfer Learning (VGG16, ResNet, MobileNet)

Improve age regression accuracy

Add real-time webcam detection

Deploy using Flask / FastAPI

Create a web-based UI


🤝 Contribution Guidelines

Contributions are welcome!

Fork the repository

Create a new branch

Commit your changes

Submit a Pull Request


📜 License

This project is licensed under the MIT License.

🙋‍♂️ Author

Lekhraj Prajapati
Machine Learning Enthusiast | Data Scientist 
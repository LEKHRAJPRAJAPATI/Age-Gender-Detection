# 👤 Age and Gender Detection using CNN

This project focuses on **predicting age and gender from facial images** using a **Convolutional Neural Network (CNN)**.  
The model is trained on the **UTKFace dataset**, which contains facial images labeled with age, gender, and ethnicity.

---

## 📌 Project Overview

Age and Gender Detection is a **computer vision and deep learning application** that takes a face image as input and predicts:
- **Age** (Regression task)
- **Gender** (Binary Classification: Male / Female)

This project demonstrates:
- Image preprocessing
- CNN-based feature extraction
- Multi-task learning (age + gender)
- Model training, evaluation, and saving

---

## 📂 Dataset Used

- **Dataset Name:** UTKFace Dataset  
- **Source:** Kaggle  
- **Dataset Link:** https://www.kaggle.com/datasets/jangedoo/utkface-new  

### Dataset Details
- Over **20,000 facial images**
- Image filename format:


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


---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/Age-Gender-Detection.git
cd Age-Gender-Detection

python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows


pip install -r requirements.txt


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



# Face Recognition Attendance System

A prototype project that uses **Convolutional Neural Networks (CNNs)** on the **Labeled Faces in the Wild (LFW)** dataset to recognize faces and mark attendance automatically.  
This project demonstrates end-to-end machine learning workflow: data preprocessing, model training, inference, and attendance logging.

---

## 🚀 Features
- **Dataset**: LFW dataset filtered for identities with ≥70 images.
- **Preprocessing**:
  - Resize images to `(62, 47)`
  - Convert to grayscale
  - Normalize pixel values to `[0, 1]`
  - One-hot encode labels
- **Model**:
  - Lightweight CNN with Conv2D, MaxPooling, GlobalAveragePooling, and Dense layers
  - Trained with `CategoricalCrossentropy` loss and `Adam` optimizer
- **Inference**:
  - Preprocessing function for uploaded images (`PIL → NumPy → Tensor`)
  - Predicts person name from input image
- **Attendance Logging**:
  - Marks attendance in a CSV file with timestamp
  - Prevents duplicate entries for the same person on the same day
- **Model Persistence**:
  - Save/load full model (`.h5`) and weights-only (`.weights.h5`)

---

## 📂 Project Structure
├── face_model.py          # Model creation and training ├── preprocess.py          # Image preprocessing utilities ├── attendance.py          # Attendance marking logic ├── train.py               # Training script ├── inference.py           # Prediction script ├── README.md              # Project documentation




## ⚙️ Installation
```bash
git clone https://github.com/your-username/face-attendance-system.git
cd face-attendance-system
pip install -r requirements.txt
```

## Inference

from PIL import Image
from inference import predict_person_name

img = Image.open("sample_face.jpg")
print(predict_person_name(img))


## Attendance Logging
#### *Attendance is stored in:
/content/drive/MyDrive/attendace_By_face/attendance.csv

## 📊 Results
- Achieved ~X% test accuracy on held-out LFW subset.
- Inference latency: ~Y ms per image on CPU.
(Replace X and Y with your actual numbers.)

## 🔮 Future Work
- Integrate face detection & alignment (OpenCV/MTCNN).
- Switch to embedding-based recognition (FaceNet/ArcFace).
- Deploy on IoT edge devices (e.g., Raspberry Pi) for real-time attendance.
- Replace CSV with secure database storage.
- Add confidence thresholding for unknown faces.



## 🙌 Acknowledgements
- Scikit-learn LFW dataset (scikit-learn.org in Bing)
- TensorFlow/Keras for deep learning framework



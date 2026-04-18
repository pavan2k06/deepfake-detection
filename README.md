# 🧠 Deepfake Detection System

## 📌 Overview

The **Deepfake Detection System** is an AI-powered application designed to identify whether an image or video is real or manipulated (deepfake).
It uses deep learning techniques to analyze visual patterns and detect inconsistencies that are not visible to the human eye.

---

## 🚀 Features

* Upload image/video for detection
* Real-time prediction using trained model
* User-friendly interface
* High accuracy deep learning model
* Supports offline execution

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Libraries:** TensorFlow, Keras, OpenCV, NumPy
* **Frontend:** Streamlit / Flask
* **Model:** CNN-based Deep Learning Model

---

## ⚙️ System Requirements

### 🔹 Hardware

* Minimum 4 GB RAM (8 GB recommended)
* GPU (optional, for faster processing)

### 🔹 Software

* Python 3.8 or above
* Anaconda (recommended) or pip

---

## 📦 Installation (Step-by-Step)

### 🔹 1. Clone the Repository

```bash
git clone https://github.com/pavan2k06/deepfake-detection.git
cd deepfake-detection
```

---

### 🔹 2. Create Virtual Environment (Recommended)

Using Anaconda:

```bash
conda create -n deepfake python=3.9
conda activate deepfake
```

OR using venv:

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 🔹 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

If requirements.txt is not available:

```bash
pip install tensorflow opencv-python numpy streamlit
```

---

## 🧠 Model Setup

⚠️ The trained model file is not included due to size limitations.

👉 Download the model from:
[https://drive.google.com/file/d/1RfR_tFmGgufjD5zEOUum8ttgU9AtxrtF/view?usp=drive_link]

After downloading:

* Place the file in the project folder
* Update the model path in code if needed

Example:

```python
model = load_model("deepfake_model_final.h5")
```

---

## ▶️ How to Run the Project

### 🔹 For Streamlit App

```bash
streamlit run app_final.py
```

### 🔹 For Python Script

```bash
python app_final.py
```

---

## 📂 Project Structure

```
deepfake-detection/
│── app_final.py          # Main application
│── requirements.txt     # Dependencies
│── README.md            # Documentation
│── .gitignore           # Ignored files
```

---

## 📸 Output

* Displays prediction result (Real / Fake)
* Shows confidence score
* Visual output in UI

<img width="1920" height="1080" alt="Screenshot 2026-04-15 095022" src="https://github.com/user-attachments/assets/d766685f-53e7-4775-870b-31b06a816ac5" />
<img width="1920" height="1080" alt="Screenshot 2026-04-15 095040" src="https://github.com/user-attachments/assets/da492eba-5d34-49f4-9121-20e50e8960d7" />
<img width="1920" height="1080" alt="Screenshot 2026-04-15 095123" src="https://github.com/user-attachments/assets/6391cc14-097f-4a25-a554-362007a332b7" />
<img width="1920" height="1080" alt="Screenshot 2026-04-15 095300" src="https://github.com/user-attachments/assets/cc970693-43e6-42ad-86af-ec2f75a0a66b" />
<img width="1920" height="1080" alt="Screenshot 2026-04-15 095216" src="https://github.com/user-attachments/assets/ed0fd761-7deb-4224-9566-f858c95e5466" />






---

## ⚠️ Limitations

* Accuracy depends on dataset quality
* Large model size
* May not detect highly advanced deepfakes

---

## 🔮 Future Improvements

* Real-time webcam detection
* Video deepfake detection
* Deploy as web/mobile app
* Improve accuracy with larger dataset

---

## 👨‍💻 Author

**Pavan**
B.Tech Data Science Student

---

## 📜 License

This project is for educational purposes only.

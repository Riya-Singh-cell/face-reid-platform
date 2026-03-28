# AI Face Recognition Attendance System

An intelligent real-time attendance system that leverages computer vision and deep learning to automate the process of marking attendance using facial recognition. This system captures live video input through a webcam, detects and identifies individuals based on facial embeddings, and records their attendance in a structured database while displaying results through an interactive dashboard.

Unlike traditional attendance systems, this solution eliminates manual effort and reduces errors by ensuring accurate identification using deep learning models. It is designed to handle real-world scenarios such as continuous movement, multiple entries, and duplicate prevention, making it efficient and reliable for practical applications.

---

## 🚀 Features

- Real-time face detection using OpenCV  
- Face recognition using DeepFace embeddings  
- Multi-person recognition with labels  
- Attendance stored in SQLite database  
- Duplicate prevention (one entry per day per person)  
- Live dashboard using Streamlit  
- Unknown face filtering  
- Entry-based counting system  

---

## 🛠️ Tech Stack

- Python  
- OpenCV  
- DeepFace  
- Streamlit  
- SQLite  
- NumPy / Pandas  

---

---

## 📌 How It Works

1. Loads known face images and generates embeddings  
2. Captures live webcam frames  
3. Detects faces using OpenCV  
4. Compares embeddings to identify individuals  
5. Displays names with confidence scores  
6. Marks attendance in database  
7. Updates dashboard in real-time  

---

## 📊 Attendance Logic

- Each person is marked only once per day  
- Entry is counted when a person appears after absence  
- Exit delay prevents multiple counts  
- Unknown faces are ignored  

---

## 🧪 Output

- Live webcam feed with bounding boxes  
- Labels showing name and confidence  
- Attendance table updating dynamically  

---


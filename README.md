
---

# Attendance Using Face Recognition

This repository contains a Flask-based web application for managing attendance using face recognition. The application allows users to mark their attendance by recognizing their face, display attendance records, and add new users to the system.

## Features

- **Face Recognition-Based Attendance**: Users can mark their attendance by scanning their face through a webcam.
- **Attendance Logs**: The system maintains daily attendance logs saved in CSV format.
- **User Registration**: Add new users by capturing their face through a webcam.
- **Model Training**: Automatically trains a KNN-based model for face recognition using captured images.
- **Web Interface**: User-friendly web interface to take attendance, view logs, and manage users.

---

### Project Structure

- **`app.py`**: Main Flask application.
- **`templates/`**: Contains HTML files for rendering web pages.
  - `home.html`
  - `attendance.html`
- **`static/faces/`**: Stores user face images for training and recognition.
- **`static/face_recognition_model.pkl`**: Pre-trained KNN model for face recognition.
- **`Attendance/`**: Stores daily attendance logs in CSV format.

---

## Installation and Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/attendance-using-face-recognition.git
   cd attendance-using-face-recognition
   ```

2. **Install Dependencies**:
   Install the required Python libraries by running:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   Start the Flask application:
   ```bash
   python app.py
   ```

4. **Access the Web Interface**:
   Open your browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

---

## Usage

### 1. **Mark Attendance**:
- Navigate to the "Take Attendance" section.
- Scan your face using the webcam.
- Attendance will be recorded if your face is recognized.

### 2. **View Attendance**:
- Go to the "Show Attendance" section to view the logs of attendance for the current day.

### 3. **Add a New User**:
- Enter a name and ID in the "Add New User" section.
- Capture your face through the webcam to register.

---

## Prerequisites

- Python 3.x
- OpenCV
- Flask
- Joblib
- Scikit-learn
- Pandas
- A webcam for face capture

---

## Screenshots

### Web Interface Overview
![alt text](image.png)

---

## How It Works

1. **Face Registration**:
   - Users are added by capturing multiple images of their face using the webcam.
   - A KNN model is trained on the registered face data.

2. **Face Detection**:
   - The system uses OpenCV's Haar Cascade Classifier to detect faces in the webcam feed.

3. **Face Recognition**:
   - Recognized faces are matched with the registered data using the trained KNN model.

4. **Attendance Logging**:
   - If a face is identified, attendance is marked in a daily log file(.csv excel file) with the timestamp.

---

## Future Enhancements

- Integrate support for mobile devices.
- Add real-time notifications for attendance.
- Enhance accuracy using advanced face recognition models.
- use database

---

## Author

- **Ashwina Rawat**

For queries or contributions, feel free to reach out or raise an issue in the repository.

---


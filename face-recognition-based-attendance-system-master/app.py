import cv2
import os
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

app = Flask(__name__)

nimgs = 10  
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
os.makedirs('Attendance', exist_ok=True)
os.makedirs('static/faces', exist_ok=True)


attendance_file = f'Attendance/Attendance-{datetoday}.csv'
if not os.path.exists(attendance_file):
    with open(attendance_file, 'w') as f:
        f.write('Name,Roll,Time\n')


def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
    return face_points

def identify_face(facearray):
    model_path = 'static/face_recognition_model.pkl'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        return model.predict(facearray)
    else:
        raise FileNotFoundError("Model file not found. Train the model first.")

def train_model():
    faces = []
    labels = []
    for user in os.listdir('static/faces'):
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

def extract_attendance():
    df = pd.read_csv(attendance_file,on_bad_lines='skip')
    return df['Name'], df['Roll'], df['Time'], len(df)

def add_attendance(name):
    username, userid = name.split('_')
    current_time = datetime.now().strftime("%H:%M:%S")
    df = pd.read_csv(attendance_file,on_bad_lines='skip')
    
    if userid not in df['Roll'].astype(str).values:  
        with open(attendance_file, 'a') as f:
            f.write(f'{username},{userid},{current_time}\n')  



def getallusers():
    userlist = os.listdir('static/faces')
    names, rolls = [], []
    for user in userlist:
        name, roll = user.split('_')
        names.append(name)
        rolls.append(roll)
    return userlist, names, rolls, len(userlist)

def deletefolder(duser):
    for img in os.listdir(duser):
        os.remove(f"{duser}/{img}")
    os.rmdir(duser)


@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/listusers')
def listusers():
    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)


@app.route('/attendance')
def attendance():
    names, rolls, times, l = extract_attendance()  # Your logic to get attendance data
    return render_template('attendance.html', names=names, rolls=rolls, times=times, l=l)


@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()
    model_path = 'static/face_recognition_model.pkl'
    
    if not os.path.exists(model_path):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, 
                               totalreg=totalreg(), datetoday2=datetoday2, 
                               mess='Train the model before proceeding.')

    cap = cv2.VideoCapture(0)
    start_time = datetime.now()
    identified_person = None
    duration = 10  

    while (datetime.now() - start_time).seconds < duration:
        ret, frame = cap.read()
        if not ret:
            break
        faces = extract_faces(frame)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            try:
                identified_person = identify_face(face.reshape(1, -1))[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
                cv2.putText(frame, identified_person, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                break  
            except Exception as e:
                print(f"Identification error: {e}")
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) & 0xFF == 27:  
            break

    cap.release()
    cv2.destroyAllWindows()

    if identified_person:
        add_attendance(identified_person)
        message = f"Attendance marked for {identified_person}"
    else:
        message = "No match found within the given time."

    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, 
                           totalreg=totalreg(), datetoday2=datetoday2, mess=message)

@app.route('/add', methods=['POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = f'static/faces/{newusername}_{newuserid}'
    os.makedirs(userimagefolder, exist_ok=True)
    cap = cv2.VideoCapture(0)
    i = 0
    while i < nimgs:
        ret, frame = cap.read()
        if not ret:
            break
        faces = extract_faces(frame)
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            if cv2.imwrite(f"{userimagefolder}/{newusername}_{i}.jpg", frame[y:y+h, x:x+w]):
                i += 1
        cv2.imshow('Capturing Images', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    train_model()
    return home()

if __name__ == '__main__':
    app.run(debug=True)








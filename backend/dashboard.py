import streamlit as st
import cv2
import time
import pandas as pd
from datetime import datetime
import sqlite3
import os

from utils.detect import detect_faces, detect_face
from utils.embed import get_embedding
from utils.match import is_match
from database import mark_attendance


st.title("AI Face Recognition Attendance System")

run = st.checkbox("Start Camera")

frame_window = st.image([])
df_placeholder = st.empty()

conn = sqlite3.connect("attendance.db", check_same_thread=False)


@st.cache_resource
def load_known_faces():
    embeddings = []
    names = []

    folder = "test_images/students"

    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        name = os.path.splitext(file)[0]

        face = detect_face(path)
        if face is None:
            continue

        emb = get_embedding(face)

        embeddings.append(emb)
        names.append(name)

    return embeddings, names


known_embeddings, known_names = load_known_faces()

if len(known_embeddings) == 0:
    st.error("No faces found in dataset")
    st.stop()


cap = cv2.VideoCapture(0)

count = 0
person_present = {}
last_seen_time = {}

exit_delay = 3

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Camera error")
        break

    faces, coords = detect_faces(frame)

    current_time = time.time()

    for face, (x, y, w, h) in zip(faces, coords):
        if w < 80 or h < 80:
            continue

        embedding = get_embedding(face)

        best_score = 0
        best_name = "Unknown"

        for known_emb, name in zip(known_embeddings, known_names):
            match, score = is_match(known_emb, embedding)

            if score > best_score:
                best_score = score
                best_name = name

        if best_score > 0.8:
            if best_name not in person_present:
                person_present[best_name] = False
                last_seen_time[best_name] = 0

            if not person_present[best_name]:
                count += 1

                now = datetime.now()
                date = now.strftime("%Y-%m-%d")
                time_now = now.strftime("%H:%M:%S")

                mark_attendance(best_name, date, time_now)

            person_present[best_name] = True
            last_seen_time[best_name] = current_time

            label = f"{best_name} {best_score:.2f}"
        else:
            label = "Unknown"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    for name in list(person_present.keys()):
        if person_present[name] and (current_time - last_seen_time[name] > exit_delay):
            person_present[name] = False

    cv2.putText(
        frame,
        f"Count: {count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
    )

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_window.image(frame)

    df = pd.read_sql_query("SELECT * FROM attendance", conn)
    df_placeholder.dataframe(df)

cap.release()
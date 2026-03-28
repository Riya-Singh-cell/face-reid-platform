import cv2
import time
from datetime import datetime

from utils.detect import detect_faces, detect_face
from utils.embed import get_embedding
from utils.match import is_match
from database import mark_attendance


target_img = "test_images/target.png"

target_face = detect_face(target_img)

if target_face is None:
    print("Target face not detected")
    exit()

target_embedding = get_embedding(target_face)

print("Target loaded successfully")


cap = cv2.VideoCapture(0)

count = 0
person_present = False
last_seen_time = 0

exit_delay = 3
frame_skip = 3
frame_count = 0

print("Starting live detection. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % frame_skip != 0:
        continue

    faces, coords = detect_faces(frame)

    found = False

    for face, (x, y, w, h) in zip(faces, coords):
        if w < 80 or h < 80:
            continue

        embedding = get_embedding(face)
        match, score = is_match(target_embedding, embedding)

        if match:
            found = True

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Target {score:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

    current_time = time.time()

    if found:
        if not person_present:
            count += 1

            now = datetime.now()
            date = now.strftime("%Y-%m-%d")
            time_now = now.strftime("%H:%M:%S")

            saved = mark_attendance("Person1", date, time_now)

            if saved:
                print(f"Attendance marked at {time_now}")
            else:
                print("Already marked today")

        person_present = True
        last_seen_time = current_time
    else:
        if person_present and (current_time - last_seen_time > exit_delay):
            person_present = False

    cv2.putText(
        frame,
        f"Count: {count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
    )

    cv2.imshow("Live Face ReID", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
from deepface import DeepFace
import cv2
import numpy as np
import uuid
import os

def get_embedding(face_img):
    temp_file = f"temp_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(temp_file, face_img)

    embedding = DeepFace.represent(
        img_path=temp_file,
        model_name="Facenet",
        enforce_detection=False
    )

    os.remove(temp_file)

    return np.array(embedding[0]["embedding"])
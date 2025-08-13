# src/firestore_admin.py
import os
import firebase_admin
from firebase_admin import credentials, firestore

def get_db(service_key_path: str | None = None):
    if not firebase_admin._apps:
        key_path = (
            service_key_path
            or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            or "config/serviceAccountKey.json"
        )
        cred = credentials.Certificate(key_path)
        firebase_admin.initialize_app(cred)
    return firestore.client()

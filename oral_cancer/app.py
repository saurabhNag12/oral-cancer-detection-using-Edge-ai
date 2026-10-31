from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
import hashlib
import io
import json
import os
import re
import secrets
import sqlite3
import tempfile
import threading
import time

import numpy as np
import tensorflow as tf
from flask import Flask, g, jsonify, request, send_file
from PIL import Image
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.security import check_password_hash, generate_password_hash

ROOT = Path(__file__).resolve().parent
STAGES = {"Not recorded", "Stage 0", "Stage I", "Stage II", "Stage III", "Stage IV"}
SESSION_DAYS = 7


class BestModelPredictor:
    RESULT_FILES = {
        "DenseNet121": "densenet121_results.json",
        "MobileNet": "mobilenet_results.json",
        "EfficientNetB0": "efficientnetb0_results.json",
    }

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir).resolve()
        self.selection_file = self.model_dir / "best_model_selection.json"
        self.model_file = self.model_dir / "best_oral_cancer_model.keras"
        self.lock = threading.RLock()

        if not self.selection_file.is_file():
            raise FileNotFoundError(
                f"Missing {self.selection_file}. Run the comparison notebook first."
            )
        if not self.model_file.is_file():
            raise FileNotFoundError(
                f"Missing {self.model_file}. Run the comparison notebook first."
            )

        with self.selection_file.open("r", encoding="utf-8") as f:
            selection = json.load(f)

        self.model_name = self._model_name(selection.get("best_model", ""))
        result_file = self.model_dir / self.RESULT_FILES[self.model_name]

        if not result_file.is_file():
            raise FileNotFoundError(f"Missing {result_file}")

        with result_file.open("r", encoding="utf-8") as f:
            info = json.load(f)

        self.class_indices = {
            str(k): int(v) for k, v in info["class_indices"].items()
        }
        self.index_to_class = {v: k for k, v in self.class_indices.items()}

        size = info.get("image_size", [100, 100])
        self.target_size = (int(size[0]), int(size[1]))

        self.preprocess_input = self._preprocess(self.model_name)
        self._validate_classes()

        self.model = tf.keras.models.load_model(self.model_file, compile=False)

    @staticmethod
    def _model_name(name):
        value = str(name).lower()
        if "dense" in value:
            return "DenseNet121"
        if "efficient" in value:
            return "EfficientNetB0"
        if "mobile" in value:
            return "MobileNet"
        raise ValueError(f"Unsupported best model: {name}")

    @staticmethod
    def _preprocess(model_name):
        if model_name == "DenseNet121":
            from tensorflow.keras.applications.densenet import preprocess_input
        elif model_name == "MobileNet":
            from tensorflow.keras.applications.mobilenet import preprocess_input
        elif model_name == "EfficientNetB0":
            from tensorflow.keras.applications.efficientnet import preprocess_input
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        return preprocess_input

    @staticmethod
    def _norm(label):
        return re.sub(r"[^A-Z0-9]+", " ", str(label).upper()).strip()

    def is_cancer_label(self, label):
        value = self._norm(label)
        if value in {"NORMAL", "NON CANCER", "NONCANCER", "NO CANCER"}:
            return False
        if value in {"CANCER", "ORAL CANCER"}:
            return True
        if "CANCER" in value and "NON" not in value and not value.startswith("NO "):
            return True
        if "NORMAL" in value:
            return False
        raise ValueError(f"Cannot map class label {label!r} to Cancer/Non-Cancer")

    def _validate_classes(self):
        labels = list(self.class_indices)
        if len(labels) != 2:
            raise ValueError(
                f"Mobile API expects 2 classes, but model contains {labels}"
            )
        positive_count = sum(self.is_cancer_label(x) for x in labels)
        if positive_count != 1:
            raise ValueError(
                f"Expected one Cancer class and one Non-Cancer class: {labels}"
            )

    def cancer_probability(self, probabilities):
        for name, value in probabilities.items():
            if self.is_cancer_label(name):
                return float(value)
        return None

    def predict(self, image_path, results_dir, user_id):
        try:
            with Image.open(image_path) as img:
                original = img.convert("RGB")
        except Exception as exc:
            raise ValueError("Uploaded file is not a readable image") from exc

        resized = original.resize(self.target_size)
        x = np.asarray(resized, dtype=np.float32)
        x = np.expand_dims(x, axis=0)
        x = self.preprocess_input(x)

        with self.lock:
            probs = self.model.predict(x, verbose=0)[0]

        pred_index = int(np.argmax(probs))
        pred_class = self.index_to_class[pred_index]
        confidence = float(probs[pred_index])
        is_cancer = self.is_cancer_label(pred_class)

        probabilities = {
            self.index_to_class[i]: float(probs[i])
            for i in sorted(self.index_to_class)
        }

        output_dir = Path(results_dir).resolve() / f"user_{int(user_id)}"
        output_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        unique = secrets.token_hex(6)
        original_path = output_dir / f"{stamp}_{unique}_original.png"
        original.save(original_path, "PNG")

        if is_cancer:
            message = (
                "AI screening result: suspicious/cancer class predicted. "
                "Please consult a qualified dentist, oral surgeon or oncologist "
                "for clinical examination and confirmatory diagnosis."
            )
        else:
            message = (
                "AI screening result: non-cancer/normal class predicted. "
                "This screening result does not rule out disease. Seek clinical "
                "evaluation if symptoms or a lesion persist."
            )

        return {
            "class": pred_class,
            "confidence": confidence,
            "is_cancer": is_cancer,
            "probabilities": probabilities,
            "original_path": str(original_path),
            "highlight_path": None,
            "message": message,
            "model_name": self.model_name,
        }


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.update(
        MAX_CONTENT_LENGTH=12 * 1024 * 1024,
        DATABASE=str(ROOT / "data" / "cancertrack.sqlite3"),
        MODEL_DIR=str(ROOT / "model_outputs"),
        RESULTS_DIR=str(ROOT / "data" / "results"),
    )
    if test_config:
        app.config.update(test_config)

    Path(app.config["DATABASE"]).parent.mkdir(parents=True, exist_ok=True)
    Path(app.config["MODEL_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["RESULTS_DIR"]).mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(app.config["DATABASE"]) as database:
        database.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions (
            token_hash TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id),
            expires_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id),
            created_at TEXT NOT NULL,
            class_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            is_cancer INTEGER NOT NULL,
            probabilities_json TEXT NOT NULL,
            original_path TEXT NOT NULL,
            highlight_path TEXT,
            message TEXT NOT NULL,
            reported_stage TEXT NOT NULL DEFAULT 'Not recorded',
            stage_note TEXT NOT NULL DEFAULT '',
            stage_updated_at TEXT
        );
        CREATE INDEX IF NOT EXISTS prediction_user
        ON predictions(user_id, id DESC);
        """)

    predictor_cache = {"value": None}
    predictor_lock = threading.RLock()

    def db():
        if "db" not in g:
            g.db = sqlite3.connect(app.config["DATABASE"], timeout=30)
            g.db.row_factory = sqlite3.Row
            g.db.execute("PRAGMA foreign_keys=ON")
        return g.db

    @app.teardown_appcontext
    def close_db(_):
        connection = g.pop("db", None)
        if connection:
            connection.close()

    def now():
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def digest(token):
        return hashlib.sha256(token.encode()).hexdigest()

    def user_json(user):
        return {k: user[k] for k in ("id", "name", "email")}

    def new_session(user):
        token = secrets.token_urlsafe(32)
        expires = time.time() + SESSION_DAYS * 86400
        db().execute(
            "INSERT INTO sessions VALUES (?,?,?)",
            (digest(token), user["id"], expires),
        )
        db().commit()
        return {"token": token, "expires_at": expires, "user": user_json(user)}

    def authenticated(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            header = request.headers.get("Authorization", "")
            token = header[7:] if header.startswith("Bearer ") else ""
            if not token:
                return jsonify(error="Please log in again"), 401

            user = db().execute("""
                SELECT u.*
                FROM sessions s
                JOIN users u ON u.id=s.user_id
                WHERE s.token_hash=? AND s.expires_at>?
            """, (digest(token), time.time())).fetchone()

            if not user:
                return jsonify(error="Please log in again"), 401

            g.user = user
            g.token_hash = digest(token)
            return fn(*args, **kwargs)

        return wrapped

    def get_predictor():
        if predictor_cache["value"] is None:
            with predictor_lock:
                if predictor_cache["value"] is None:
                    predictor_cache["value"] = BestModelPredictor(
                        app.config["MODEL_DIR"]
                    )
        return predictor_cache["value"]

    def record(row):
        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "class": row["class_name"],
            "confidence": row["confidence"],
            "is_cancer": bool(row["is_cancer"]),
            "probabilities": json.loads(row["probabilities_json"]),
            "original_url": f'/api/predictions/{row["id"]}/image/original',
            "highlight_url": (
                f'/api/predictions/{row["id"]}/image/highlight'
                if row["highlight_path"] else None
            ),
            "message": row["message"],
            "reported_stage": row["reported_stage"],
            "stage_note": row["stage_note"],
            "stage_updated_at": row["stage_updated_at"],
            "stage_source": (
                "User-recorded from clinical report; not verified by this app"
            ),
            "model_stage": (
                "Not determined: classifier predicts Cancer/Non-Cancer only"
            ),
        }

    def owned(prediction_id):
        return db().execute(
            "SELECT * FROM predictions WHERE id=? AND user_id=?",
            (prediction_id, g.user["id"]),
        ).fetchone()

    def cancer_score(probabilities):
        try:
            return get_predictor().cancer_probability(probabilities)
        except Exception:
            for label, value in probabilities.items():
                text = re.sub(r"[^A-Z0-9]+", " ", label.upper()).strip()
                if (
                    "CANCER" in text
                    and "NON" not in text
                    and not text.startswith("NO ")
                ):
                    return float(value)
            return None

    @app.get("/api/health")
    def health():
        result = {"status": "ok", "service": "CancerTrack API"}
        try:
            p = get_predictor()
            result.update(
                model_status="loaded",
                best_model=p.model_name,
                classes=list(p.class_indices.keys()),
                image_size=list(p.target_size),
            )
        except Exception as exc:
            result.update(model_status="not_loaded", model_error=str(exc))
        return jsonify(result)

    @app.post("/api/auth/signup")
    def signup():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(error="JSON name, email and password are required"), 400

        name, email, password = (data.get(k) for k in ("name", "email", "password"))
        if not all(isinstance(v, str) for v in (name, email, password)):
            return jsonify(error="Name, email and password must be text"), 400

        name, email = name.strip(), email.strip().lower()
        if (
            not 2 <= len(name) <= 100
            or len(email) > 254
            or not re.fullmatch(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", email)
        ):
            return jsonify(error="Enter a name and valid email address"), 400

        if not 8 <= len(password) <= 128:
            return jsonify(error="Password must be 8 to 128 characters"), 400

        try:
            cursor = db().execute(
                "INSERT INTO users(name,email,password_hash,created_at) VALUES(?,?,?,?)",
                (name, email, generate_password_hash(password, method="scrypt"), now()),
            )
            db().commit()
        except sqlite3.IntegrityError:
            db().rollback()
            return jsonify(error="Email is already registered"), 409

        user = db().execute(
            "SELECT * FROM users WHERE id=?",
            (cursor.lastrowid,),
        ).fetchone()
        return jsonify(new_session(user)), 201

    @app.post("/api/auth/login")
    def login():
        data = request.get_json(silent=True) or {}
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("email"), str)
            or not isinstance(data.get("password"), str)
        ):
            return jsonify(error="Email and password are required"), 400

        if len(data["password"]) > 128:
            return jsonify(error="Invalid email or password"), 401

        user = db().execute(
            "SELECT * FROM users WHERE email=?",
            (data["email"].strip().lower(),),
        ).fetchone()

        if not user or not check_password_hash(
            user["password_hash"], data["password"]
        ):
            return jsonify(error="Invalid email or password"), 401

        return jsonify(new_session(user))

    @app.post("/api/auth/logout")
    @authenticated
    def logout():
        db().execute("DELETE FROM sessions WHERE token_hash=?", (g.token_hash,))
        db().commit()
        return jsonify(message="Logged out")

    @app.get("/api/me")
    @authenticated
    def me():
        return jsonify(user=user_json(g.user))

    @app.post("/api/predictions")
    @authenticated
    def predict_upload():
        upload = request.files.get("image")
        if not upload or not upload.filename:
            return jsonify(error="Choose an input image"), 400

        try:
            predictor = get_predictor()

            with tempfile.TemporaryDirectory(prefix="cancertrack_") as folder:
                suffix = Path(upload.filename).suffix or ".img"
                temp_path = Path(folder) / f"upload{suffix}"
                upload.save(temp_path)

                result = predictor.predict(
                    temp_path,
                    app.config["RESULTS_DIR"],
                    g.user["id"],
                )

            cursor = db().execute("""
                INSERT INTO predictions(
                    user_id, created_at, class_name, confidence, is_cancer,
                    probabilities_json, original_path, highlight_path, message
                )
                VALUES(?,?,?,?,?,?,?,?,?)
            """, (
                g.user["id"],
                now(),
                result["class"],
                result["confidence"],
                int(result["is_cancer"]),
                json.dumps(result["probabilities"]),
                result["original_path"],
                result["highlight_path"],
                result["message"],
            ))
            db().commit()

            return jsonify(
                prediction=record(owned(cursor.lastrowid)),
                model=result["model_name"],
            ), 201

        except (ValueError, FileNotFoundError, OSError) as exc:
            return jsonify(error=str(exc)), 400
        except Exception:
            app.logger.exception("Prediction failed")
            return jsonify(
                error="Prediction failed. Check model_outputs and server terminal."
            ), 500

    @app.get("/api/predictions")
    @authenticated
    def history():
        try:
            before_id = int(
                request.args.get("before_id", "9223372036854775807")
            )
        except ValueError:
            return jsonify(error="Invalid history cursor"), 400

        rows = db().execute("""
            SELECT * FROM predictions
            WHERE user_id=? AND id<?
            ORDER BY id DESC
            LIMIT 51
        """, (g.user["id"], before_id)).fetchall()

        shown = rows[:50]
        return jsonify(
            predictions=[record(r) for r in shown],
            next_before_id=shown[-1]["id"] if len(rows) > 50 else None,
        )

    @app.get("/api/summary")
    @authenticated
    def summary():
        rows = db().execute(
            "SELECT * FROM predictions WHERE user_id=? ORDER BY id DESC LIMIT 2",
            (g.user["id"],),
        ).fetchall()

        current = record(rows[0]) if rows else None
        previous = record(rows[1]) if len(rows) > 1 else None
        comparison = None

        if current and previous:
            current_score = cancer_score(current["probabilities"])
            previous_score = cancer_score(previous["probabilities"])
            delta = (
                (current_score - previous_score) * 100
                if current_score is not None and previous_score is not None
                else None
            )
            comparison = {
                "class_changed": current["class"] != previous["class"],
                "cancer_score_delta_percentage_points": delta,
                "reported_stage_changed": (
                    current["reported_stage"] != previous["reported_stage"]
                ),
                "note": (
                    "Compare recorded observations only. Model score changes do "
                    "not measure disease progression, recovery or cancer stage."
                ),
            }

        count = db().execute(
            "SELECT COUNT(*) FROM predictions WHERE user_id=?",
            (g.user["id"],),
        ).fetchone()[0]

        return jsonify(
            current=current,
            previous=previous,
            comparison=comparison,
            count=count,
        )

    @app.get("/api/predictions/<int:prediction_id>")
    @authenticated
    def details(prediction_id):
        row = owned(prediction_id)
        if not row:
            return jsonify(error="Record not found"), 404
        return jsonify(prediction=record(row))

    @app.put("/api/predictions/<int:prediction_id>/stage")
    @authenticated
    def stage(prediction_id):
        if not owned(prediction_id):
            return jsonify(error="Record not found"), 404

        data = request.get_json(silent=True)
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("stage"), str)
            or data["stage"] not in STAGES
            or not isinstance(data.get("note", ""), str)
            or len(data.get("note", "")) > 1000
        ):
            return jsonify(
                error="Select a supported stage and keep the note under 1000 characters"
            ), 400

        db().execute("""
            UPDATE predictions
            SET reported_stage=?, stage_note=?, stage_updated_at=?
            WHERE id=? AND user_id=?
        """, (
            data["stage"],
            data.get("note", "").strip(),
            now(),
            prediction_id,
            g.user["id"],
        ))
        db().commit()

        return jsonify(prediction=record(owned(prediction_id)))

    @app.get("/api/predictions/<int:prediction_id>/image/<kind>")
    @authenticated
    def prediction_image(prediction_id, kind):
        row = owned(prediction_id)
        if not row or kind not in ("original", "highlight"):
            return jsonify(error="Image not found"), 404

        value = row[kind + "_path"]
        if not value:
            return jsonify(error="Image not found"), 404

        path = Path(value).resolve()
        results_root = Path(app.config["RESULTS_DIR"]).resolve()

        try:
            inside = path.is_relative_to(results_root)
        except AttributeError:
            inside = os.path.commonpath(
                [str(path), str(results_root)]
            ) == str(results_root)

        if not inside or not path.is_file():
            return jsonify(error="Image not found"), 404

        if request.args.get("thumbnail") == "1":
            with Image.open(path) as img:
                img = img.convert("RGB")
                img.thumbnail((1200, 1200))
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
            buffer.seek(0)
            response = send_file(buffer, mimetype="image/png")
        else:
            response = send_file(path, mimetype="image/png")

        response.headers["Cache-Control"] = "no-store"
        return response

    @app.after_request
    def no_cache(response):
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_):
        return jsonify(error="Maximum upload size is 12 MB"), 413

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=5000, debug=False, threaded=True)

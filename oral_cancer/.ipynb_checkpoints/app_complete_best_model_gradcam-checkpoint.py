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

STAGES = {
    "Not recorded",
    "Stage 0",
    "Stage I",
    "Stage II",
    "Stage III",
    "Stage IV",
}

SESSION_DAYS = 7


# ============================================================
# BEST MODEL PREDICTOR
# ============================================================

class BestModelPredictor:

    RESULT_FILES = {
        "DenseNet121": "densenet121_results.json",
        "MobileNet": "mobilenet_results.json",
        "EfficientNetB0": "efficientnetb0_results.json",
    }

    def __init__(self, model_dir):

        print("\n========== MODEL INITIALIZATION ==========")

        self.model_dir = Path(model_dir).resolve()

        self.selection_file = (
            self.model_dir / "best_model_selection.json"
        )

        self.model_file = (
            self.model_dir / "best_oral_cancer_model.keras"
        )

        self.lock = threading.RLock()

        print("Model directory :", self.model_dir)
        print("Selection file  :", self.selection_file)
        print("Model file      :", self.model_file)

        if not self.selection_file.is_file():
            raise FileNotFoundError(
                f"Missing file: {self.selection_file}"
            )

        if not self.model_file.is_file():
            raise FileNotFoundError(
                f"Missing file: {self.model_file}"
            )

        # ----------------------------------------------------
        # Read best-model selection
        # ----------------------------------------------------

        with self.selection_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            selection = json.load(file)

        print("Selection JSON:", selection)

        self.model_name = self._canonical_model_name(
            selection.get("best_model", "")
        )

        print("Selected best model:", self.model_name)

        # ----------------------------------------------------
        # Read winning model metadata
        # ----------------------------------------------------

        result_filename = self.RESULT_FILES[
            self.model_name
        ]

        self.result_file = (
            self.model_dir / result_filename
        )

        print("Result file:", self.result_file)

        if not self.result_file.is_file():
            raise FileNotFoundError(
                f"Missing result file: {self.result_file}"
            )

        with self.result_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            model_info = json.load(file)

        print("Model metadata:", model_info)

        raw_class_indices = model_info.get(
            "class_indices"
        )

        if (
            not isinstance(raw_class_indices, dict)
            or not raw_class_indices
        ):
            raise ValueError(
                "class_indices missing in result JSON"
            )

        self.class_indices = {
            str(class_name): int(index)
            for class_name, index
            in raw_class_indices.items()
        }

        self.index_to_class = {
            index: class_name
            for class_name, index
            in self.class_indices.items()
        }

        image_size = model_info.get(
            "image_size",
            [100, 100],
        )

        if (
            not isinstance(image_size, (list, tuple))
            or len(image_size) != 2
        ):
            image_size = [100, 100]

        self.target_size = (
            int(image_size[0]),
            int(image_size[1]),
        )

        print(
            "Class indices:",
            self.class_indices,
        )

        print(
            "Image size:",
            self.target_size,
        )

        # ----------------------------------------------------
        # Correct preprocessing for winning architecture
        # ----------------------------------------------------

        self.preprocess_input = (
            self._get_preprocess_function(
                self.model_name
            )
        )

        self._validate_binary_classes()

        # ----------------------------------------------------
        # Load Keras model
        # ----------------------------------------------------

        print("Loading Keras model...")

        self.model = tf.keras.models.load_model(
            self.model_file,
            compile=False,
        )

        print(
            "MODEL LOADED SUCCESSFULLY:",
            self.model_name,
        )

        print(
            "=========================================\n"
        )

    # --------------------------------------------------------
    # Model name normalization
    # --------------------------------------------------------

    @staticmethod
    def _canonical_model_name(name):

        value = str(name).strip().lower()

        if "dense" in value:
            return "DenseNet121"

        if "efficient" in value:
            return "EfficientNetB0"

        if "mobile" in value:
            return "MobileNet"

        raise ValueError(
            f"Unsupported model name: {name}"
        )

    # --------------------------------------------------------
    # Preprocessing selector
    # --------------------------------------------------------

    @staticmethod
    def _get_preprocess_function(
        model_name
    ):

        if model_name == "DenseNet121":

            from tensorflow.keras.applications.densenet import (
                preprocess_input,
            )

            return preprocess_input

        if model_name == "MobileNet":

            from tensorflow.keras.applications.mobilenet import (
                preprocess_input,
            )

            return preprocess_input

        if model_name == "EfficientNetB0":

            from tensorflow.keras.applications.efficientnet import (
                preprocess_input,
            )

            return preprocess_input

        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    # --------------------------------------------------------
    # Label helper
    # --------------------------------------------------------

    @staticmethod
    def _normalize_label(label):

        return re.sub(
            r"[^A-Z0-9]+",
            " ",
            str(label).upper(),
        ).strip()

    def is_cancer_label(
        self,
        label,
    ):

        value = self._normalize_label(
            label
        )

        negative_labels = {
            "NORMAL",
            "NON CANCER",
            "NONCANCER",
            "NO CANCER",
            "NO CANCER DETECTED",
        }

        positive_labels = {
            "CANCER",
            "ORAL CANCER",
        }

        if value in negative_labels:
            return False

        if value in positive_labels:
            return True

        if (
            "CANCER" in value
            and "NON" not in value
            and not value.startswith("NO ")
        ):
            return True

        if "NORMAL" in value:
            return False

        raise ValueError(
            f"Unable to map class label: {label}"
        )

    def _validate_binary_classes(self):

        labels = list(
            self.class_indices.keys()
        )

        if len(labels) != 2:

            raise ValueError(
                "The app currently expects exactly "
                f"two classes. Found: {labels}"
            )

        cancer_count = sum(
            self.is_cancer_label(label)
            for label in labels
        )

        if cancer_count != 1:

            raise ValueError(
                "Expected one Cancer class and one "
                f"Non-Cancer class. Found: {labels}"
            )

    # --------------------------------------------------------
    # Cancer probability
    # --------------------------------------------------------

    def cancer_probability(
        self,
        probabilities,
    ):

        for label, probability in (
            probabilities.items()
        ):

            if self.is_cancer_label(
                label
            ):

                return float(
                    probability
                )

        return None

    # --------------------------------------------------------
    # Grad-CAM-style attention heatmap
    # --------------------------------------------------------

    def _gradcam(
        self,
        batch,
        predicted_index,
    ):

        try:

            backbone = self.model.layers[0]

            if len(
                backbone.output.shape
            ) != 4:

                print(
                    "Grad-CAM skipped: "
                    "backbone output is not 4D"
                )

                return None

            grad_model = tf.keras.Model(
                inputs=self.model.inputs,
                outputs=[
                    backbone.output,
                    self.model.output,
                ],
            )

            with tf.GradientTape() as tape:

                feature_map, predictions = (
                    grad_model(
                        batch,
                        training=False,
                    )
                )

                score = predictions[
                    :,
                    predicted_index,
                ]

            gradients = tape.gradient(
                score,
                feature_map,
            )

            if gradients is None:

                print(
                    "Grad-CAM skipped: "
                    "gradient is None"
                )

                return None

            weights = tf.reduce_mean(
                gradients,
                axis=(0, 1, 2),
            )

            feature_map = (
                feature_map[0]
            )

            heatmap = tf.reduce_sum(
                feature_map * weights,
                axis=-1,
            )

            heatmap = tf.nn.relu(
                heatmap
            )

            max_value = tf.reduce_max(
                heatmap
            )

            if (
                float(
                    max_value.numpy()
                )
                <= 0
            ):

                return None

            heatmap = (
                heatmap
                / max_value
            )

            return heatmap.numpy()

        except Exception as exc:

            print(
                "Grad-CAM error:",
                repr(exc),
            )

            return None

    # --------------------------------------------------------
    # Save red attention overlay
    # --------------------------------------------------------

    @staticmethod
    def _save_overlay(
        original,
        heatmap,
        output_path,
    ):

        heat_image = Image.fromarray(
            np.uint8(
                np.clip(
                    heatmap,
                    0,
                    1,
                )
                * 255
            ),
            mode="L",
        )

        heat_image = heat_image.resize(
            original.size
        )

        heat_array = np.asarray(
            heat_image,
            dtype=np.uint8,
        )

        base = np.asarray(
            original.convert("RGBA"),
            dtype=np.uint8,
        ).copy()

        overlay = np.zeros_like(
            base
        )

        overlay[..., 0] = 255

        overlay[..., 3] = (
            heat_array.astype(
                np.float32
            )
            * 0.60
        ).astype(
            np.uint8
        )

        result = Image.alpha_composite(
            Image.fromarray(
                base,
                mode="RGBA",
            ),
            Image.fromarray(
                overlay,
                mode="RGBA",
            ),
        ).convert(
            "RGB"
        )

        result.save(
            output_path,
            format="PNG",
        )

    # --------------------------------------------------------
    # Predict image
    # --------------------------------------------------------

    def predict(
        self,
        image_path,
        results_dir,
        user_id,
    ):

        print("\n---------- PREDICTION ----------")

        print(
            "Input image:",
            image_path,
        )

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        try:

            with Image.open(
                image_path
            ) as image:

                original = (
                    image
                    .convert("RGB")
                )

        except Exception as exc:

            raise ValueError(
                "Uploaded file is not "
                "a readable image"
            ) from exc

        # ----------------------------------------------------
        # Resize and preprocess
        # ----------------------------------------------------

        resized = original.resize(
            self.target_size
        )

        image_array = np.asarray(
            resized,
            dtype=np.float32,
        )

        batch = np.expand_dims(
            image_array,
            axis=0,
        )

        batch = self.preprocess_input(
            batch
        )

        # ----------------------------------------------------
        # Model inference
        # ----------------------------------------------------

        with self.lock:

            probabilities_array = (
                self.model.predict(
                    batch,
                    verbose=0,
                )[0]
            )

        print(
            "Raw probabilities:",
            probabilities_array,
        )

        predicted_index = int(
            np.argmax(
                probabilities_array
            )
        )

        if (
            predicted_index
            not in self.index_to_class
        ):

            raise ValueError(
                "Model returned unexpected "
                f"class index {predicted_index}"
            )

        predicted_class = (
            self.index_to_class[
                predicted_index
            ]
        )

        confidence = float(
            probabilities_array[
                predicted_index
            ]
        )

        is_cancer = (
            self.is_cancer_label(
                predicted_class
            )
        )

        probabilities = {
            self.index_to_class[index]:
                float(
                    probabilities_array[
                        index
                    ]
                )
            for index
            in sorted(
                self.index_to_class
            )
        }

        print(
            "Predicted class:",
            predicted_class,
        )

        print(
            "Confidence:",
            confidence,
        )

        print(
            "Cancer:",
            is_cancer,
        )

        # ----------------------------------------------------
        # Result directory
        # ----------------------------------------------------

        result_root = Path(
            results_dir
        ).resolve()

        user_folder = (
            result_root
            / f"user_{int(user_id)}"
        )

        user_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        timestamp = (
            datetime.now(
                timezone.utc
            )
            .strftime(
                "%Y%m%dT%H%M%S"
            )
        )

        unique_id = secrets.token_hex(
            6
        )

        # ----------------------------------------------------
        # Save original image
        # ----------------------------------------------------

        original_path = (
            user_folder
            / (
                f"{timestamp}_"
                f"{unique_id}_"
                "original.png"
            )
        )

        original.save(
            original_path,
            format="PNG",
        )

        print(
            "Original saved:",
            original_path,
        )

        # ----------------------------------------------------
        # Highlight only for cancer prediction
        # ----------------------------------------------------

        highlight_path = None

        if is_cancer:

            print(
                "Generating attention highlight..."
            )

            heatmap = self._gradcam(
                batch,
                predicted_index,
            )

            if heatmap is not None:

                highlight_file = (
                    user_folder
                    / (
                        f"{timestamp}_"
                        f"{unique_id}_"
                        "highlight.png"
                    )
                )

                self._save_overlay(
                    original,
                    heatmap,
                    highlight_file,
                )

                highlight_path = str(
                    highlight_file.resolve()
                )

                print(
                    "Highlight saved:",
                    highlight_path,
                )

            else:

                print(
                    "Highlight could not be generated"
                )

        # ----------------------------------------------------
        # User message
        # ----------------------------------------------------

        if is_cancer:

            message = (
                "AI screening result: suspicious/"
                "cancer class predicted. "
                "The highlighted region represents "
                "model attention and is not a "
                "confirmed tumour boundary. "
                "Please consult a qualified dentist, "
                "oral surgeon or oncologist for "
                "clinical examination and confirmation."
            )

        else:

            message = (
                "AI screening result: "
                "non-cancer/normal class predicted. "
                "This does not completely rule out "
                "disease. Seek clinical evaluation "
                "if symptoms or a lesion persist."
            )

        print(
            "-------------------------------\n"
        )

        return {
            "class":
                predicted_class,

            "confidence":
                confidence,

            "is_cancer":
                is_cancer,

            "probabilities":
                probabilities,

            "original_path":
                str(
                    original_path.resolve()
                ),

            "highlight_path":
                highlight_path,

            "message":
                message,

            "model_name":
                self.model_name,
        }


# ============================================================
# FLASK APPLICATION
# ============================================================

def create_app(
    test_config=None
):

    app = Flask(
        __name__
    )

    app.config.update(

        MAX_CONTENT_LENGTH=(
            12
            * 1024
            * 1024
        ),

        DATABASE=str(
            ROOT
            / "data"
            / "cancertrack.sqlite3"
        ),

        MODEL_DIR=str(
            ROOT
            / "model_outputs"
        ),

        RESULTS_DIR=str(
            ROOT
            / "data"
            / "results"
        ),
    )

    if test_config:

        app.config.update(
            test_config
        )

    # --------------------------------------------------------
    # Ensure folders exist
    # --------------------------------------------------------

    Path(
        app.config[
            "DATABASE"
        ]
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(
        app.config[
            "MODEL_DIR"
        ]
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(
        app.config[
            "RESULTS_DIR"
        ]
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Database
    # --------------------------------------------------------

    with sqlite3.connect(
        app.config[
            "DATABASE"
        ]
    ) as database:

        database.executescript(
            """
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
                user_id INTEGER NOT NULL
                    REFERENCES users(id),
                expires_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL
                    REFERENCES users(id),

                created_at TEXT NOT NULL,

                class_name TEXT NOT NULL,

                confidence REAL NOT NULL,

                is_cancer INTEGER NOT NULL,

                probabilities_json TEXT NOT NULL,

                original_path TEXT NOT NULL,

                highlight_path TEXT,

                message TEXT NOT NULL,

                reported_stage TEXT
                    NOT NULL
                    DEFAULT 'Not recorded',

                stage_note TEXT
                    NOT NULL
                    DEFAULT '',

                stage_updated_at TEXT
            );

            CREATE INDEX IF NOT EXISTS prediction_user
            ON predictions(
                user_id,
                id DESC
            );
            """
        )

    # --------------------------------------------------------
    # Predictor cache
    # --------------------------------------------------------

    predictor_cache = {
        "value": None
    }

    predictor_lock = (
        threading.RLock()
    )

    # --------------------------------------------------------
    # Database helper
    # --------------------------------------------------------

    def db():

        if "db" not in g:

            g.db = sqlite3.connect(
                app.config[
                    "DATABASE"
                ],
                timeout=30,
            )

            g.db.row_factory = (
                sqlite3.Row
            )

            g.db.execute(
                "PRAGMA foreign_keys=ON"
            )

        return g.db

    @app.teardown_appcontext
    def close_db(_):

        connection = (
            g.pop(
                "db",
                None,
            )
        )

        if connection:

            connection.close()

    # --------------------------------------------------------
    # Common helpers
    # --------------------------------------------------------

    def now():

        return datetime.now(
            timezone.utc
        ).isoformat(
            timespec="seconds"
        )

    def digest(token):

        return hashlib.sha256(
            token.encode()
        ).hexdigest()

    def user_json(user):

        return {
            key: user[key]
            for key
            in (
                "id",
                "name",
                "email",
            )
        }

    def new_session(user):

        token = (
            secrets.token_urlsafe(
                32
            )
        )

        expires = (
            time.time()
            + SESSION_DAYS
            * 86400
        )

        db().execute(
            """
            INSERT INTO sessions
            VALUES (?,?,?)
            """,
            (
                digest(token),
                user["id"],
                expires,
            ),
        )

        db().commit()

        return {
            "token":
                token,

            "expires_at":
                expires,

            "user":
                user_json(
                    user
                ),
        }

    # --------------------------------------------------------
    # Authentication decorator
    # --------------------------------------------------------

    def authenticated(fn):

        @wraps(fn)
        def wrapped(
            *args,
            **kwargs,
        ):

            header = (
                request.headers.get(
                    "Authorization",
                    "",
                )
            )

            token = (
                header[7:]
                if header.startswith(
                    "Bearer "
                )
                else ""
            )

            if not token:

                return jsonify(
                    error=(
                        "Please log in again"
                    )
                ), 401

            user = db().execute(
                """
                SELECT u.*
                FROM sessions s

                JOIN users u
                ON u.id=s.user_id

                WHERE
                    s.token_hash=?
                    AND
                    s.expires_at>?
                """,
                (
                    digest(token),
                    time.time(),
                ),
            ).fetchone()

            if not user:

                return jsonify(
                    error=(
                        "Please log in again"
                    )
                ), 401

            g.user = user

            g.token_hash = (
                digest(token)
            )

            return fn(
                *args,
                **kwargs,
            )

        return wrapped

    # --------------------------------------------------------
    # Load predictor exactly once
    # --------------------------------------------------------

    def get_predictor():

        if (
            predictor_cache[
                "value"
            ]
            is None
        ):

            print(
                "Predictor is not loaded yet"
            )

            with predictor_lock:

                if (
                    predictor_cache[
                        "value"
                    ]
                    is None
                ):

                    print(
                        "Loading model from:"
                    )

                    print(
                        app.config[
                            "MODEL_DIR"
                        ]
                    )

                    predictor_cache[
                        "value"
                    ] = (
                        BestModelPredictor(
                            app.config[
                                "MODEL_DIR"
                            ]
                        )
                    )

        return predictor_cache[
            "value"
        ]

    # --------------------------------------------------------
    # Record serializer
    # --------------------------------------------------------

    def record(row):

        return {
            "id":
                row["id"],

            "created_at":
                row[
                    "created_at"
                ],

            "class":
                row[
                    "class_name"
                ],

            "confidence":
                row[
                    "confidence"
                ],

            "is_cancer":
                bool(
                    row[
                        "is_cancer"
                    ]
                ),

            "probabilities":
                json.loads(
                    row[
                        "probabilities_json"
                    ]
                ),

            "original_url":
                (
                    f'/api/predictions/'
                    f'{row["id"]}/'
                    f'image/original'
                ),

            "highlight_url":
                (
                    f'/api/predictions/'
                    f'{row["id"]}/'
                    f'image/highlight'
                    if row[
                        "highlight_path"
                    ]
                    else None
                ),

            "message":
                row[
                    "message"
                ],

            "reported_stage":
                row[
                    "reported_stage"
                ],

            "stage_note":
                row[
                    "stage_note"
                ],

            "stage_updated_at":
                row[
                    "stage_updated_at"
                ],

            "stage_source":
                (
                    "User-recorded from "
                    "clinical report; "
                    "not verified by this app"
                ),

            "model_stage":
                (
                    "Not determined: "
                    "classifier predicts "
                    "Cancer/Non-Cancer only"
                ),
        }

    # --------------------------------------------------------
    # Ownership
    # --------------------------------------------------------

    def owned(
        prediction_id
    ):

        return db().execute(
            """
            SELECT *
            FROM predictions

            WHERE
                id=?
                AND
                user_id=?
            """,
            (
                prediction_id,
                g.user["id"],
            ),
        ).fetchone()

    # ========================================================
    # HEALTH
    # ========================================================

    @app.get(
        "/api/health"
    )
    def health():

        result = {
            "status":
                "ok",

            "service":
                "CancerTrack API",
        }

        try:

            model_predictor = (
                get_predictor()
            )

            result.update(

                model_status=
                    "loaded",

                best_model=
                    model_predictor
                    .model_name,

                classes=list(
                    model_predictor
                    .class_indices
                    .keys()
                ),

                image_size=list(
                    model_predictor
                    .target_size
                ),
            )

        except Exception as exc:

            print(
                "HEALTH MODEL ERROR:",
                repr(exc),
            )

            result.update(

                model_status=
                    "not_loaded",

                model_error=
                    str(exc),
            )

        return jsonify(
            result
        )

    # ========================================================
    # SIGNUP
    # ========================================================

    @app.post(
        "/api/auth/signup"
    )
    def signup():

        data = request.get_json(
            silent=True
        )

        if not isinstance(
            data,
            dict,
        ):

            return jsonify(
                error=(
                    "JSON name, email and "
                    "password are required"
                )
            ), 400

        name = data.get(
            "name"
        )

        email = data.get(
            "email"
        )

        password = data.get(
            "password"
        )

        if not all(
            isinstance(
                value,
                str,
            )
            for value
            in (
                name,
                email,
                password,
            )
        ):

            return jsonify(
                error=(
                    "Name, email and "
                    "password must be text"
                )
            ), 400

        name = (
            name.strip()
        )

        email = (
            email
            .strip()
            .lower()
        )

        if (
            not 2
            <= len(name)
            <= 100
        ):

            return jsonify(
                error=(
                    "Enter a valid name"
                )
            ), 400

        if (
            len(email) > 254
            or not re.fullmatch(
                r"^[^\s@]+@[^\s@]+\.[^\s@]+$",
                email,
            )
        ):

            return jsonify(
                error=(
                    "Enter a valid email address"
                )
            ), 400

        if (
            not 8
            <= len(password)
            <= 128
        ):

            return jsonify(
                error=(
                    "Password must be "
                    "8 to 128 characters"
                )
            ), 400

        try:

            cursor = db().execute(
                """
                INSERT INTO users(
                    name,
                    email,
                    password_hash,
                    created_at
                )
                VALUES(?,?,?,?)
                """,
                (
                    name,
                    email,
                    generate_password_hash(
                        password,
                        method="scrypt",
                    ),
                    now(),
                ),
            )

            db().commit()

        except sqlite3.IntegrityError:

            db().rollback()

            return jsonify(
                error=(
                    "Email is already registered"
                )
            ), 409

        user = db().execute(
            """
            SELECT *
            FROM users
            WHERE id=?
            """,
            (
                cursor.lastrowid,
            ),
        ).fetchone()

        return jsonify(
            new_session(
                user
            )
        ), 201

    # ========================================================
    # LOGIN
    # ========================================================

    @app.post(
        "/api/auth/login"
    )
    def login():

        data = (
            request.get_json(
                silent=True
            )
            or {}
        )

        if (
            not isinstance(
                data,
                dict,
            )
            or
            not isinstance(
                data.get(
                    "email"
                ),
                str,
            )
            or
            not isinstance(
                data.get(
                    "password"
                ),
                str,
            )
        ):

            return jsonify(
                error=(
                    "Email and password "
                    "are required"
                )
            ), 400

        email = (
            data[
                "email"
            ]
            .strip()
            .lower()
        )

        password = data[
            "password"
        ]

        if len(
            password
        ) > 128:

            return jsonify(
                error=(
                    "Invalid email or password"
                )
            ), 401

        user = db().execute(
            """
            SELECT *
            FROM users
            WHERE email=?
            """,
            (
                email,
            ),
        ).fetchone()

        if (
            not user
            or
            not check_password_hash(
                user[
                    "password_hash"
                ],
                password,
            )
        ):

            return jsonify(
                error=(
                    "Invalid email or password"
                )
            ), 401

        return jsonify(
            new_session(
                user
            )
        )

    # ========================================================
    # LOGOUT
    # ========================================================

    @app.post(
        "/api/auth/logout"
    )
    @authenticated
    def logout():

        db().execute(
            """
            DELETE FROM sessions
            WHERE token_hash=?
            """,
            (
                g.token_hash,
            ),
        )

        db().commit()

        return jsonify(
            message="Logged out"
        )

    # ========================================================
    # CURRENT USER
    # ========================================================

    @app.get(
        "/api/me"
    )
    @authenticated
    def me():

        return jsonify(
            user=user_json(
                g.user
            )
        )

    # ========================================================
    # PREDICT IMAGE
    # ========================================================

    @app.post(
        "/api/predictions"
    )
    @authenticated
    def predict_upload():

        print(
            "\n========== IMAGE REQUEST =========="
        )

        print(
            "receiving image"
        )

        upload = request.files.get(
            "image"
        )

        if upload is None:

            print(
                "No multipart field named image"
            )

            return jsonify(
                error=(
                    "No image field received"
                )
            ), 400

        print(
            "filename:",
            upload.filename,
        )

        if not upload.filename:

            return jsonify(
                error=(
                    "Choose an input image"
                )
            ), 400

        try:

            print(
                "before get_predictor"
            )

            model_predictor = (
                get_predictor()
            )

            print(
                "predictor loaded:",
                model_predictor
                .model_name,
            )

            with tempfile.TemporaryDirectory(
                prefix="cancertrack_"
            ) as folder:

                suffix = (
                    Path(
                        upload.filename
                    )
                    .suffix
                )

                if not suffix:

                    suffix = ".jpg"

                temporary_path = (
                    Path(folder)
                    / (
                        "upload"
                        + suffix
                    )
                )

                upload.save(
                    temporary_path
                )

                print(
                    "Uploaded image saved to:",
                    temporary_path,
                )

                print(
                    "File exists:",
                    temporary_path
                    .exists(),
                )

                print(
                    "File size:",
                    temporary_path
                    .stat()
                    .st_size,
                )

                result = (
                    model_predictor
                    .predict(
                        temporary_path,
                        app.config[
                            "RESULTS_DIR"
                        ],
                        g.user[
                            "id"
                        ],
                    )
                )

            print(
                "Prediction result:",
                result,
            )

            cursor = db().execute(
                """
                INSERT INTO predictions(
                    user_id,
                    created_at,
                    class_name,
                    confidence,
                    is_cancer,
                    probabilities_json,
                    original_path,
                    highlight_path,
                    message
                )
                VALUES(
                    ?,?,?,?,?,?,?,?,?
                )
                """,
                (
                    g.user[
                        "id"
                    ],

                    now(),

                    result[
                        "class"
                    ],

                    result[
                        "confidence"
                    ],

                    int(
                        result[
                            "is_cancer"
                        ]
                    ),

                    json.dumps(
                        result[
                            "probabilities"
                        ]
                    ),

                    result[
                        "original_path"
                    ],

                    result[
                        "highlight_path"
                    ],

                    result[
                        "message"
                    ],
                ),
            )

            db().commit()

            saved_row = owned(
                cursor.lastrowid
            )

            print(
                "Database record saved:",
                cursor.lastrowid,
            )

            print(
                "===================================\n"
            )

            return jsonify(

                prediction=
                    record(
                        saved_row
                    ),

                model=
                    result[
                        "model_name"
                    ],

            ), 201

        except (
            ValueError,
            FileNotFoundError,
            OSError,
        ) as exc:

            print(
                "PREDICTION ERROR:",
                repr(exc),
            )

            return jsonify(
                error=str(exc)
            ), 400

        except Exception as exc:

            print(
                "UNEXPECTED ERROR:",
                repr(exc),
            )

            app.logger.exception(
                "Prediction failed"
            )

            return jsonify(
                error=str(exc)
            ), 500

    # ========================================================
    # PREDICTION HISTORY
    # ========================================================

    @app.get(
        "/api/predictions"
    )
    @authenticated
    def history():

        try:

            before_id = int(
                request.args.get(
                    "before_id",
                    "9223372036854775807",
                )
            )

        except ValueError:

            return jsonify(
                error=(
                    "Invalid history cursor"
                )
            ), 400

        rows = db().execute(
            """
            SELECT *
            FROM predictions

            WHERE
                user_id=?
                AND id<?

            ORDER BY id DESC

            LIMIT 51
            """,
            (
                g.user[
                    "id"
                ],
                before_id,
            ),
        ).fetchall()

        visible_rows = rows[
            :50
        ]

        return jsonify(

            predictions=[
                record(row)
                for row
                in visible_rows
            ],

            next_before_id=(
                visible_rows[
                    -1
                ][
                    "id"
                ]
                if len(rows) > 50
                else None
            ),
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    @app.get(
        "/api/summary"
    )
    @authenticated
    def summary():

        rows = db().execute(
            """
            SELECT *
            FROM predictions

            WHERE user_id=?

            ORDER BY id DESC

            LIMIT 2
            """,
            (
                g.user[
                    "id"
                ],
            ),
        ).fetchall()

        current = (
            record(
                rows[0]
            )
            if rows
            else None
        )

        previous = (
            record(
                rows[1]
            )
            if len(rows) > 1
            else None
        )

        comparison = None

        if (
            current
            and previous
        ):

            current_score = (
                None
            )

            previous_score = (
                None
            )

            try:

                model_predictor = (
                    get_predictor()
                )

                current_score = (
                    model_predictor
                    .cancer_probability(
                        current[
                            "probabilities"
                        ]
                    )
                )

                previous_score = (
                    model_predictor
                    .cancer_probability(
                        previous[
                            "probabilities"
                        ]
                    )
                )

            except Exception:

                pass

            delta = (
                (
                    current_score
                    - previous_score
                )
                * 100
                if (
                    current_score
                    is not None
                    and
                    previous_score
                    is not None
                )
                else None
            )

            comparison = {

                "class_changed":
                    (
                        current[
                            "class"
                        ]
                        !=
                        previous[
                            "class"
                        ]
                    ),

                "cancer_score_delta_percentage_points":
                    delta,

                "reported_stage_changed":
                    (
                        current[
                            "reported_stage"
                        ]
                        !=
                        previous[
                            "reported_stage"
                        ]
                    ),

                "note":
                    (
                        "Model score changes do not "
                        "measure disease progression, "
                        "recovery or cancer stage."
                    ),
            }

        count = db().execute(
            """
            SELECT COUNT(*)
            FROM predictions
            WHERE user_id=?
            """,
            (
                g.user[
                    "id"
                ],
            ),
        ).fetchone()[0]

        return jsonify(

            current=current,

            previous=previous,

            comparison=comparison,

            count=count,
        )

    # ========================================================
    # PREDICTION DETAILS
    # ========================================================

    @app.get(
        "/api/predictions/<int:prediction_id>"
    )
    @authenticated
    def details(
        prediction_id
    ):

        row = owned(
            prediction_id
        )

        if not row:

            return jsonify(
                error=(
                    "Record not found"
                )
            ), 404

        return jsonify(
            prediction=record(
                row
            )
        )

    # ========================================================
    # USER-RECORDED STAGE
    # ========================================================

    @app.put(
        "/api/predictions/<int:prediction_id>/stage"
    )
    @authenticated
    def stage(
        prediction_id
    ):

        if not owned(
            prediction_id
        ):

            return jsonify(
                error=(
                    "Record not found"
                )
            ), 404

        data = request.get_json(
            silent=True
        )

        if (
            not isinstance(
                data,
                dict,
            )
            or
            not isinstance(
                data.get(
                    "stage"
                ),
                str,
            )
            or
            data[
                "stage"
            ]
            not in STAGES
            or
            not isinstance(
                data.get(
                    "note",
                    "",
                ),
                str,
            )
            or
            len(
                data.get(
                    "note",
                    "",
                )
            )
            > 1000
        ):

            return jsonify(
                error=(
                    "Select a supported stage "
                    "and keep note below "
                    "1000 characters"
                )
            ), 400

        db().execute(
            """
            UPDATE predictions

            SET
                reported_stage=?,
                stage_note=?,
                stage_updated_at=?

            WHERE
                id=?
                AND
                user_id=?
            """,
            (
                data[
                    "stage"
                ],

                data.get(
                    "note",
                    "",
                ).strip(),

                now(),

                prediction_id,

                g.user[
                    "id"
                ],
            ),
        )

        db().commit()

        return jsonify(
            prediction=record(
                owned(
                    prediction_id
                )
            )
        )

    # ========================================================
    # PROTECTED RESULT IMAGE
    # ========================================================

    @app.get(
        "/api/predictions/"
        "<int:prediction_id>/"
        "image/<kind>"
    )
    @authenticated
    def prediction_image(
        prediction_id,
        kind,
    ):

        row = owned(
            prediction_id
        )

        if (
            not row
            or kind
            not in (
                "original",
                "highlight",
            )
        ):

            return jsonify(
                error=(
                    "Image not found"
                )
            ), 404

        path_value = row[
            kind
            + "_path"
        ]

        if not path_value:

            return jsonify(
                error=(
                    "Image not found"
                )
            ), 404

        image_path = Path(
            path_value
        ).resolve()

        results_root = Path(
            app.config[
                "RESULTS_DIR"
            ]
        ).resolve()

        try:

            safe = (
                image_path
                .is_relative_to(
                    results_root
                )
            )

        except AttributeError:

            safe = (
                os.path.commonpath(
                    [
                        str(
                            image_path
                        ),
                        str(
                            results_root
                        ),
                    ]
                )
                ==
                str(
                    results_root
                )
            )

        if (
            not safe
            or
            not image_path.is_file()
        ):

            return jsonify(
                error=(
                    "Image not found"
                )
            ), 404

        if (
            request.args.get(
                "thumbnail"
            )
            == "1"
        ):

            with Image.open(
                image_path
            ) as image:

                image = (
                    image.convert(
                        "RGB"
                    )
                )

                image.thumbnail(
                    (
                        1200,
                        1200,
                    )
                )

                buffer = (
                    io.BytesIO()
                )

                image.save(
                    buffer,
                    format="PNG",
                )

            buffer.seek(
                0
            )

            response = send_file(
                buffer,
                mimetype=(
                    "image/png"
                ),
            )

        else:

            response = send_file(
                image_path,
                mimetype=(
                    "image/png"
                ),
            )

        response.headers[
            "Cache-Control"
        ] = "no-store"

        return response

    # ========================================================
    # NO CACHE
    # ========================================================

    @app.after_request
    def no_cache(
        response
    ):

        response.headers[
            "Cache-Control"
        ] = "no-store"

        return response

    # ========================================================
    # LARGE FILE ERROR
    # ========================================================

    @app.errorhandler(
        RequestEntityTooLarge
    )
    def too_large(_):

        return jsonify(
            error=(
                "Maximum upload size "
                "is 12 MB"
            )
        ), 413

    return app


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    create_app().run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
    )

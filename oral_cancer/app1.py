
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
import io
import hashlib
import json
import os
import re
import secrets
import sqlite3
import tempfile
import time
from flask import Flask, g, jsonify, request, send_file
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.exceptions import RequestEntityTooLarge
from predict import get_predictor, read_metadata, validate_cancer_classes

ROOT = Path(__file__).resolve().parent
STAGES = {'Not recorded', 'Stage 0', 'Stage I', 'Stage II', 'Stage III', 'Stage IV'}
SESSION_DAYS = 7

def create_app(test_config=None):
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=12 * 1024 * 1024,
                      DATABASE=str(ROOT / 'data' / 'cancertrack.sqlite3'),
                      MODEL_DIR=str(ROOT / 'model_outputs'),
                      RESULTS_DIR=str(ROOT / 'data' / 'results'))
    if test_config:
        app.config.update(test_config)
    Path(app.config['DATABASE']).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(app.config['DATABASE']) as db:
        db.executescript('''
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS users (
          id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL UNIQUE,
          password_hash TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS sessions (
          token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id), expires_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS predictions (
          id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id), created_at TEXT NOT NULL,
          class_name TEXT NOT NULL, confidence REAL NOT NULL, is_cancer INTEGER NOT NULL,
          probabilities_json TEXT NOT NULL, original_path TEXT NOT NULL, highlight_path TEXT,
          message TEXT NOT NULL, reported_stage TEXT NOT NULL DEFAULT 'Not recorded',
          stage_note TEXT NOT NULL DEFAULT '', stage_updated_at TEXT);
        CREATE INDEX IF NOT EXISTS prediction_user ON predictions(user_id, id DESC);
        ''')

    def db():
        if 'db' not in g:
            g.db = sqlite3.connect(app.config['DATABASE'], timeout=30)
            g.db.row_factory = sqlite3.Row
            g.db.execute('PRAGMA foreign_keys=ON')
        return g.db

    @app.teardown_appcontext
    def close_db(_):
        connection = g.pop('db', None)
        if connection:
            connection.close()

    def now():
        return datetime.now(timezone.utc).isoformat(timespec='seconds')

    def digest(token):
        return hashlib.sha256(token.encode()).hexdigest()

    def user_json(user):
        return {k: user[k] for k in ('id', 'name', 'email')}

    def session(user):
        token = secrets.token_urlsafe(32)
        expires = time.time() + SESSION_DAYS * 86400
        db().execute('INSERT INTO sessions VALUES (?,?,?)', (digest(token), user['id'], expires))
        db().commit()
        return {'token': token, 'expires_at': expires, 'user': user_json(user)}

    def authenticated(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            header = request.headers.get('Authorization', '')
            token = header[7:] if header.startswith('Bearer ') else ''
            user = db().execute('''SELECT u.* FROM sessions s JOIN users u ON u.id=s.user_id
                               WHERE s.token_hash=? AND s.expires_at>?''', (digest(token), time.time())).fetchone()
            if not user:
                return jsonify(error='Please log in again'), 401
            g.user = user
            g.token_hash = digest(token)
            return fn(*args, **kwargs)
        return wrapped

    def record(row):
        return {'id': row['id'], 'created_at': row['created_at'], 'class': row['class_name'],
                'confidence': row['confidence'], 'is_cancer': bool(row['is_cancer']),
                'probabilities': json.loads(row['probabilities_json']),
                'original_url': f'/api/predictions/{row["id"]}/image/original',
                'highlight_url': f'/api/predictions/{row["id"]}/image/highlight' if row['highlight_path'] else None,
                'message': row['message'], 'reported_stage': row['reported_stage'],
                'stage_note': row['stage_note'], 'stage_updated_at': row['stage_updated_at'],
                'stage_source': 'User-recorded from clinical report; not verified by this app',
                'model_stage': 'Not determined: classifier predicts Cancer/NORMAL only'}

    def owned(prediction_id):
        return db().execute('SELECT * FROM predictions WHERE id=? AND user_id=?',
                            (prediction_id, g.user['id'])).fetchone()

    @app.get('/api/health')
    def health():
        return jsonify(status='ok', service='CancerTrack API')

    @app.post('/api/auth/signup')
    def signup():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(error='JSON name, email and password are required'), 400
        name, email, password = (data.get(k) for k in ('name', 'email', 'password'))
        if not all(isinstance(v, str) for v in (name, email, password)):
            return jsonify(error='Name, email and password must be text'), 400
        name, email = name.strip(), email.strip().lower()
        if not 2 <= len(name) <= 100 or len(email) > 254 or not re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+', email):
            return jsonify(error='Enter a name and valid email address'), 400
        if not 8 <= len(password) <= 128:
            return jsonify(error='Password must be 8 to 128 characters'), 400
        try:
            cursor = db().execute('INSERT INTO users(name,email,password_hash,created_at) VALUES(?,?,?,?)',
                                  (name, email, generate_password_hash(password, method='scrypt'), now()))
            db().commit()
        except sqlite3.IntegrityError:
            db().rollback()
            return jsonify(error='Email is already registered'), 409
        user = db().execute('SELECT * FROM users WHERE id=?', (cursor.lastrowid,)).fetchone()
        return jsonify(session(user)), 201

    @app.post('/api/auth/login')
    def login():
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict) or not isinstance(data.get('email'), str) or not isinstance(data.get('password'), str):
            return jsonify(error='Email and password are required'), 400
        if len(data['password']) > 128:
            return jsonify(error='Invalid email or password'), 401
        user = db().execute('SELECT * FROM users WHERE email=?', (data['email'].strip().lower(),)).fetchone()
        if not user or not check_password_hash(user['password_hash'], data['password']):
            return jsonify(error='Invalid email or password'), 401
        return jsonify(session(user))

    @app.post('/api/auth/logout')
    @authenticated
    def logout():
        db().execute('DELETE FROM sessions WHERE token_hash=?', (g.token_hash,))
        db().commit()
        return jsonify(message='Logged out')

    @app.get('/api/me')
    @authenticated
    def me():
        return jsonify(user=user_json(g.user))

    @app.post('/api/predictions')
    @authenticated
    def predict_upload():
        upload = request.files.get('image')
        if not upload or not upload.filename:
            return jsonify(error='Choose an input image'), 400
        try:
            meta = read_metadata(app.config['MODEL_DIR'])
            # The uploaded notebook reported exact Cancer and NORMAL labels.
            # Refuse another label set instead of silently mapping unfamiliar labels.
            if set(meta['class_names']) != {'Cancer', 'NORMAL'}:
                raise ValueError('This mobile API expects Cancer/NORMAL labels from your notebook. Configure mapping for another model explicitly.')
            validate_cancer_classes(meta, ['Cancer'], app.config['MODEL_DIR'])
            predictor = get_predictor(str(Path(app.config['MODEL_DIR']).resolve()))
            with tempfile.TemporaryDirectory(prefix='cancertrack_') as folder:
                path = Path(folder) / 'upload'
                upload.save(path)
                result = predictor.predict(path, ['Cancer'], app.config['RESULTS_DIR'])
            # Prediction directory is outside the web root; image access checks ownership.
            cursor = db().execute('''INSERT INTO predictions(user_id,created_at,class_name,confidence,is_cancer,
                                   probabilities_json,original_path,highlight_path,message) VALUES(?,?,?,?,?,?,?,?,?)''',
                (g.user['id'], now(), result['class'], result['confidence'], int(result['is_cancer']),
                 json.dumps(result['probabilities']), result['original_path'], result['highlight_path'], result['message']))
            db().commit()
            return jsonify(prediction=record(owned(cursor.lastrowid))), 201
        except (ValueError, FileNotFoundError) as exc:
            return jsonify(error=str(exc)), 400
        except Exception:
            app.logger.exception('Prediction failed')
            return jsonify(error='Prediction failed. Check model files and the server terminal.'), 500

    @app.get('/api/predictions')
    @authenticated
    def history():
        try:
            before_id = int(request.args.get('before_id', '9223372036854775807'))
        except ValueError:
            return jsonify(error='Invalid history cursor'), 400
        rows = db().execute('SELECT * FROM predictions WHERE user_id=? AND id<? ORDER BY id DESC LIMIT 51',
                            (g.user['id'], before_id)).fetchall()
        return jsonify(predictions=[record(r) for r in rows[:50]],
                       next_before_id=rows[49]['id'] if len(rows) > 50 else None)

    @app.get('/api/summary')
    @authenticated
    def summary():
        rows = db().execute('SELECT * FROM predictions WHERE user_id=? ORDER BY id DESC LIMIT 2',
                            (g.user['id'],)).fetchall()
        current = record(rows[0]) if rows else None
        previous = record(rows[1]) if len(rows) > 1 else None
        comparison = None
        if previous:
            current_score = current['probabilities'].get('Cancer')
            previous_score = previous['probabilities'].get('Cancer')
            delta = (current_score-previous_score)*100 if current_score is not None and previous_score is not None else None
            comparison = {'class_changed': current['class'] != previous['class'],
                          'cancer_score_delta_percentage_points': delta,
                          'reported_stage_changed': current['reported_stage'] != previous['reported_stage'],
                          'note': 'Compare recorded observations only. Model score changes do not measure disease progression, recovery or cancer stage.'}
        return jsonify(current=current, previous=previous, comparison=comparison,
                       count=db().execute('SELECT COUNT(*) FROM predictions WHERE user_id=?', (g.user['id'],)).fetchone()[0])

    @app.get('/api/predictions/<int:prediction_id>')
    @authenticated
    def details(prediction_id):
        row = owned(prediction_id)
        return (jsonify(prediction=record(row)) if row else (jsonify(error='Record not found'), 404))

    @app.put('/api/predictions/<int:prediction_id>/stage')
    @authenticated
    def stage(prediction_id):
        if not owned(prediction_id):
            return jsonify(error='Record not found'), 404
        data = request.get_json(silent=True)
        if (not isinstance(data, dict) or not isinstance(data.get('stage'), str) or
            data['stage'] not in STAGES or not isinstance(data.get('note', ''), str) or len(data.get('note', '')) > 1000):
            return jsonify(error='Select a supported stage and keep the note under 1000 characters'), 400
        db().execute('UPDATE predictions SET reported_stage=?,stage_note=?,stage_updated_at=? WHERE id=? AND user_id=?',
                     (data['stage'], data.get('note', '').strip(), now(), prediction_id, g.user['id']))
        db().commit()
        return jsonify(prediction=record(owned(prediction_id)))

    @app.get('/api/predictions/<int:prediction_id>/image/<kind>')
    @authenticated
    def image(prediction_id, kind):
        row = owned(prediction_id)
        if not row or kind not in ('original', 'highlight') or not row[kind + '_path']:
            return jsonify(error='Image not found'), 404
        path = Path(row[kind + '_path']).resolve()
        if not path.is_relative_to(Path(app.config['RESULTS_DIR']).resolve()) or not path.is_file():
            return jsonify(error='Image not found'), 404
        if request.args.get('thumbnail') == '1':
            from PIL import Image
            with Image.open(path) as img:
                img.thumbnail((1200, 1200))
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
            buffer.seek(0)
            response = send_file(buffer, mimetype='image/png')
        else:
            response = send_file(path, mimetype='image/png')
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.after_request
    def no_cache(response):
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_):
        return jsonify(error='Maximum upload size is 12 MB'), 413

    return app

if __name__ == '__main__':
    create_app().run(host='0.0.0.0', port=5000, debug=False, threaded=True)

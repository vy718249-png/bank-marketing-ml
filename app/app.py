import os
import uuid
import json
import re

import mysql.connector
import pandas as pd

from io import BytesIO
from pathlib import Path
from datetime import datetime
from collections import Counter

from dotenv import load_dotenv

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    send_file
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from functools import wraps

from src.predict import (
    predict_customer,
    predict_batch,
    FEATURES
)


# ==================================================
# CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

app = Flask(__name__)

app.secret_key = os.getenv(
    "SECRET_KEY",
    "change-this-secret-key"
)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


UPLOAD_FOLDER = BASE_DIR / "uploads" / "batch_results"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


DB_CONFIG = {

    "host": os.getenv("DB_HOST", "127.0.0.1"),

    "port": int(os.getenv("DB_PORT", 3306)),

    "user": os.getenv("DB_USER"),

    "password": os.getenv("DB_PASSWORD"),

    "database": os.getenv("DB_NAME"),

    "autocommit": False

}


NUMERIC_FEATURES = [
    "age",
    "balance",
    "day",
    "campaign",
    "pdays",
    "previous"
]


# ==================================================
# DATABASE CONNECTION
# ==================================================

def get_db():

    return mysql.connector.connect(**DB_CONFIG)


# ==================================================
# INITIALIZE DATABASE
# ==================================================

def init_db():

    conn = get_db()

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (

            id INT PRIMARY KEY AUTO_INCREMENT,

            username VARCHAR(100) NOT NULL UNIQUE,

            email VARCHAR(150) NOT NULL UNIQUE,

            password VARCHAR(255) NOT NULL,

            role ENUM('admin', 'user')
            NOT NULL DEFAULT 'user',

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (

            id INT PRIMARY KEY AUTO_INCREMENT,

            user_id INT NOT NULL,

            prediction INT NOT NULL,

            probability FLOAT NOT NULL,

            status VARCHAR(100),

            input_data JSON NULL,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (user_id)
            REFERENCES users(id)
            ON DELETE CASCADE

        )
    """)

    conn.commit()

    cursor.close()

    conn.close()


# ==================================================
# LOGIN REQUIRED
# ==================================================

def login_required(func):

    @wraps(func)

    def wrapper(*args, **kwargs):

        if "user_id" not in session:

            return redirect(url_for("login"))

        return func(*args, **kwargs)

    return wrapper


# ==================================================
# ADMIN REQUIRED
# ==================================================

def admin_required(func):

    @wraps(func)

    def wrapper(*args, **kwargs):

        if "user_id" not in session:

            return redirect(url_for("login"))

        if session.get("role") != "admin":

            flash(
                "You are not authorized to access this page.",
                "error"
            )

            return redirect(url_for("dashboard"))

        return func(*args, **kwargs)

    return wrapper


# ==================================================
# SAVE PREDICTION
# ==================================================

def save_prediction(user_id, result, input_data):

    conn = get_db()

    cursor = conn.cursor()

    try:

        cursor.execute("""

            INSERT INTO predictions

            (
                user_id,
                prediction,
                probability,
                status,
                input_data
            )

            VALUES (%s, %s, %s, %s, %s)

        """, (

            user_id,

            result["prediction"],

            result["probability"],

            result["status"],

            json.dumps(input_data)

        ))

        conn.commit()

    except Exception:

        conn.rollback()

        raise

    finally:

        cursor.close()

        conn.close()


# ==================================================
# HOME
# ==================================================

@app.route("/")

def home():

    if "user_id" in session:

        return redirect(url_for("dashboard"))

    return redirect(url_for("login"))


# ==================================================
# REGISTER
# ==================================================

@app.route("/register", methods=["GET", "POST"])

def register():

    if request.method == "POST":

        username = request.form.get(
            "username", ""
        ).strip()

        email = request.form.get(
            "email", ""
        ).strip()

        password = request.form.get(
            "password", ""
        )

        if not username or not email or not password:

            flash(
                "All fields are required.",
                "error"
            )

            return redirect(url_for("register"))

        if len(password) < 6:

            flash(
                "Password must contain at least 6 characters.",
                "error"
            )

            return redirect(url_for("register"))

        hashed_password = generate_password_hash(password)

        conn = get_db()

        cursor = conn.cursor()

        try:

            cursor.execute("""

                INSERT INTO users

                (username, email, password)

                VALUES (%s, %s, %s)

            """, (

                username,

                email,

                hashed_password

            ))

            conn.commit()

            flash(
                "Registration successful. Please login.",
                "success"
            )

            return redirect(url_for("login"))

        except mysql.connector.IntegrityError:

            conn.rollback()

            flash(
                "Username or email already exists.",
                "error"
            )

        finally:

            cursor.close()

            conn.close()

    return render_template("register.html")


# ==================================================
# LOGIN
# ==================================================

@app.route("/login", methods=["GET", "POST"])

def login():

    if request.method == "POST":

        email = request.form.get(
            "email", ""
        ).strip()

        password = request.form.get(
            "password", ""
        )

        conn = get_db()

        cursor = conn.cursor(dictionary=True)

        try:

            cursor.execute("""

                SELECT *

                FROM users

                WHERE email = %s

            """, (email,))

            user = cursor.fetchone()

        finally:

            cursor.close()

            conn.close()

        if user and check_password_hash(
            user["password"],
            password
        ):

            session.clear()

            session["user_id"] = user["id"]

            session["username"] = user["username"]

            session["role"] = user["role"]

            return redirect(url_for("dashboard"))

        flash(
            "Invalid email or password.",
            "error"
        )

    return render_template("login.html")


# ==================================================
# USER DASHBOARD
# ==================================================

@app.route("/dashboard")

@login_required

def dashboard():

    conn = get_db()

    cursor = conn.cursor(dictionary=True)

    try:

        cursor.execute("""

            SELECT

                COUNT(*) AS total,

                COALESCE(
                    SUM(prediction = 1),
                    0
                ) AS subscribed,

                COALESCE(
                    SUM(prediction = 0),
                    0
                ) AS not_subscribed

            FROM predictions

            WHERE user_id = %s

        """, (session["user_id"],))

        stats = cursor.fetchone()

        cursor.execute("""

            SELECT *

            FROM predictions

            WHERE user_id = %s

            ORDER BY created_at DESC

            LIMIT 10

        """, (session["user_id"],))

        history = cursor.fetchall()

    finally:

        cursor.close()

        conn.close()

    return render_template(

        "dashboard.html",

        username=session["username"],

        total=stats["total"],

        subscribed=stats["subscribed"],

        not_subscribed=stats["not_subscribed"],

        history=history,

        is_admin=session.get("role") == "admin"

    )


# ==================================================
# SINGLE CUSTOMER PREDICTION
# ==================================================

@app.route("/predict", methods=["GET", "POST"])

@login_required

def predict():

    result = None

    features = [

        {"name": "age", "type": "number"},

        {
            "name": "job",
            "type": "text",
            "options": [
                "admin.",
                "blue-collar",
                "entrepreneur",
                "housemaid",
                "management",
                "retired",
                "self-employed",
                "services",
                "student",
                "technician",
                "unemployed",
                "unknown"
            ]
        },

        {
            "name": "marital",
            "type": "text",
            "options": [
                "married",
                "single",
                "divorced"
            ]
        },

        {
            "name": "education",
            "type": "text",
            "options": [
                "primary",
                "secondary",
                "tertiary",
                "unknown"
            ]
        },

        {
            "name": "default",
            "type": "text",
            "options": ["yes", "no"]
        },

        {"name": "balance", "type": "number"},

        {
            "name": "housing",
            "type": "text",
            "options": ["yes", "no"]
        },

        {
            "name": "loan",
            "type": "text",
            "options": ["yes", "no"]
        },

        {
            "name": "contact",
            "type": "text",
            "options": [
                "cellular",
                "telephone",
                "unknown"
            ]
        },

        {"name": "day", "type": "number"},

        {
            "name": "month",
            "type": "text",
            "options": [
                "jan", "feb", "mar", "apr",
                "may", "jun", "jul", "aug",
                "sep", "oct", "nov", "dec"
            ]
        },

        {"name": "campaign", "type": "number"},

        {"name": "pdays", "type": "number"},

        {"name": "previous", "type": "number"},

        {
            "name": "poutcome",
            "type": "text",
            "options": [
                "unknown",
                "other",
                "failure",
                "success"
            ]
        }

    ]

    if request.method == "POST":

        try:

            input_data = {}

            for feature in FEATURES:

                value = request.form.get(
                    feature, ""
                ).strip()

                if not value:

                    raise ValueError(
                        f"{feature} is required."
                    )

                input_data[feature] = value

            for feature in NUMERIC_FEATURES:

                input_data[feature] = float(
                    input_data[feature]
                )

            result = predict_customer(input_data)

            save_prediction(
                session["user_id"],
                result,
                input_data
            )

        except Exception as e:

            flash(str(e), "error")

    return render_template(

        "index.html",

        features=features,

        result=result

    )


# ==================================================
# BATCH CUSTOMER PREDICTION
# ==================================================

@app.route("/batch-predict", methods=["GET", "POST"])

@login_required

def batch_predict_route():

    if request.method == "POST":

        file = request.files.get("file")

        if not file or file.filename == "":

            flash(
                "Please select a CSV file.",
                "error"
            )

            return redirect(
                url_for("batch_predict_route")
            )

        if not file.filename.lower().endswith(".csv"):

            flash(
                "Only CSV files are allowed.",
                "error"
            )

            return redirect(
                url_for("batch_predict_route")
            )

        try:

            df = pd.read_csv(
                file,
                sep=None,
                engine="python"
            )

            if df.empty:

                raise ValueError(
                    "CSV file is empty."
                )

            if len(df) > 10000:

                raise ValueError(
                    "Maximum 10,000 customer records allowed."
                )

            df = df.drop(
                columns=["y", "duration"],
                errors="ignore"
            )

            missing_columns = [
                col for col in FEATURES
                if col not in df.columns
            ]

            if missing_columns:

                raise ValueError(
                    "Missing columns: "
                    + ", ".join(missing_columns)
                )

            df = df[FEATURES].copy()

            results = predict_batch(df)

            total = len(results)

            subscribed = int(
                (results["prediction"] == 1).sum()
            )

            not_subscribed = total - subscribed

            file_id = str(uuid.uuid4())

            result_path = (
                UPLOAD_FOLDER / f"{file_id}.csv"
            )

            results.to_csv(
                result_path,
                index=False
            )

            conn = get_db()

            cursor = conn.cursor()

            try:

                records = []

                for _, row in results.iterrows():

                    input_data = {
                        feature: row[feature]
                        for feature in FEATURES
                    }

                    records.append((

                        session["user_id"],

                        int(row["prediction"]),

                        float(row["probability"]),

                        row["status"],

                        json.dumps(
                            input_data,
                            default=str
                        )

                    ))

                cursor.executemany("""

                    INSERT INTO predictions

                    (
                        user_id,
                        prediction,
                        probability,
                        status,
                        input_data
                    )

                    VALUES (%s, %s, %s, %s, %s)

                """, records)

                conn.commit()

            except Exception:

                conn.rollback()

                raise

            finally:

                cursor.close()

                conn.close()

            session["batch_result_file"] = file_id

            return render_template(

                "batch_prediction.html",

                total=total,

                subscribed=subscribed,

                not_subscribed=not_subscribed,

                results=results.head(100).to_dict(
                    orient="records"
                ),

                file_id=file_id

            )

        except Exception as e:

            flash(
                f"Batch prediction failed: {str(e)}",
                "error"
            )

            return redirect(
                url_for("batch_predict_route")
            )

    return render_template("batch_prediction.html")


# ==================================================
# DOWNLOAD BATCH RESULTS
# ==================================================

@app.route("/download-batch/<file_id>")

@login_required

def download_batch(file_id):

    if session.get("batch_result_file") != file_id:

        flash(
            "Invalid download request.",
            "error"
        )

        return redirect(
            url_for("dashboard")
        )

    file_path = UPLOAD_FOLDER / f"{file_id}.csv"

    if not file_path.exists():

        flash(
            "Result file not found.",
            "error"
        )

        return redirect(
            url_for("dashboard")
        )

    return send_file(

        file_path,

        as_attachment=True,

        download_name="bank_marketing_predictions.csv"

    )


# ==================================================
# ANALYTICS FILTER HELPER
# ==================================================

def get_analytics_filters():

    start_date = request.args.get(
        "start_date", ""
    ).strip()

    end_date = request.args.get(
        "end_date", ""
    ).strip()

    user_id = request.args.get(
        "user_id", ""
    ).strip()

    if start_date:

        datetime.strptime(
            start_date,
            "%Y-%m-%d"
        )

    if end_date:

        datetime.strptime(
            end_date,
            "%Y-%m-%d"
        )

    if user_id and not user_id.isdigit():

        raise ValueError("Invalid user filter.")

    conditions = []

    params = []

    if start_date:

        conditions.append(
            "DATE(p.created_at) >= %s"
        )

        params.append(start_date)

    if end_date:

        conditions.append(
            "DATE(p.created_at) <= %s"
        )

        params.append(end_date)

    if user_id:

        conditions.append(
            "p.user_id = %s"
        )

        params.append(int(user_id))

    where_clause = ""

    if conditions:

        where_clause = (
            " WHERE " + " AND ".join(conditions)
        )

    return (
        where_clause,
        params,
        start_date,
        end_date,
        user_id
    )


# ==================================================
# CUSTOMER DEMOGRAPHIC ANALYTICS
# ==================================================

def build_demographic_analytics(rows):

    age_counter = Counter()

    job_counter = Counter()

    education_counter = Counter()

    for row in rows:

        data = row["input_data"]

        if not data:

            continue

        if isinstance(data, str):

            try:

                data = json.loads(data)

            except (ValueError, TypeError):

                continue

        if not isinstance(data, dict):

            continue

        age = data.get("age")

        try:

            age = float(age)

            if age < 30:

                age_counter["Below 30"] += 1

            elif age < 40:

                age_counter["30-39"] += 1

            elif age < 50:

                age_counter["40-49"] += 1

            elif age < 60:

                age_counter["50-59"] += 1

            else:

                age_counter["60+"] += 1

        except (ValueError, TypeError):

            pass

        job = data.get(
            "job",
            "unknown"
        )

        education = data.get(
            "education",
            "unknown"
        )

        job_counter[str(job)] += 1

        education_counter[str(education)] += 1

    age_order = [
        "Below 30",
        "30-39",
        "40-49",
        "50-59",
        "60+"
    ]

    age_labels = [
        label for label in age_order
        if age_counter[label] > 0
    ]

    return {

        "age_labels": age_labels,

        "age_values": [
            age_counter[label]
            for label in age_labels
        ],

        "job_labels": list(job_counter.keys()),

        "job_values": list(job_counter.values()),

        "education_labels": list(
            education_counter.keys()
        ),

        "education_values": list(
            education_counter.values()
        )

    }


# ==================================================
# ADMIN ANALYTICS DASHBOARD
# ==================================================

@app.route("/admin/analytics")

@admin_required

def admin_analytics():

    conn = get_db()

    cursor = conn.cursor(dictionary=True)

    try:

        (
            where_clause,
            params,
            start_date,
            end_date,
            selected_user
        ) = get_analytics_filters()

        # ------------------------------------------
        # TOTAL USERS
        # ------------------------------------------

        cursor.execute("""

            SELECT COUNT(*) AS total_users

            FROM users

        """)

        total_users = cursor.fetchone()["total_users"]

        # ------------------------------------------
        # PREDICTION KPIs
        # ------------------------------------------

        cursor.execute(f"""

            SELECT

                COUNT(*) AS total_predictions,

                COALESCE(
                    SUM(p.prediction = 1),
                    0
                ) AS subscribed,

                COALESCE(
                    SUM(p.prediction = 0),
                    0
                ) AS not_subscribed,

                COALESCE(
                    AVG(p.probability),
                    0
                ) AS avg_probability

            FROM predictions p

            {where_clause}

        """, params)

        stats = cursor.fetchone()

        total = stats["total_predictions"]

        subscribed = stats["subscribed"]

        not_subscribed = stats["not_subscribed"]

        avg_probability = float(
            stats["avg_probability"] or 0
        )

        conversion_rate = (
            subscribed / total * 100
            if total > 0
            else 0
        )

        # ------------------------------------------
        # PREDICTION DISTRIBUTION
        # ------------------------------------------

        prediction_labels = [
            "Subscribed",
            "Not Subscribed"
        ]

        prediction_values = [
            subscribed,
            not_subscribed
        ]

        # ------------------------------------------
        # DAILY TREND
        # ------------------------------------------

        cursor.execute(f"""

            SELECT

                DATE(p.created_at) AS prediction_date,

                COUNT(*) AS total

            FROM predictions p

            {where_clause}

            GROUP BY DATE(p.created_at)

            ORDER BY prediction_date

        """, params)

        daily_rows = cursor.fetchall()

        daily_labels = [
            str(row["prediction_date"])
            for row in daily_rows
        ]

        daily_values = [
            row["total"]
            for row in daily_rows
        ]

        # ------------------------------------------
        # MONTHLY TREND
        # ------------------------------------------

        cursor.execute(f"""

            SELECT

                DATE_FORMAT(
                    p.created_at,
                    '%Y-%m'
                ) AS prediction_month,

                COUNT(*) AS total

            FROM predictions p

            {where_clause}

            GROUP BY prediction_month

            ORDER BY prediction_month

        """, params)

        monthly_rows = cursor.fetchall()

        monthly_labels = [
            row["prediction_month"]
            for row in monthly_rows
        ]

        monthly_values = [
            row["total"]
            for row in monthly_rows
        ]

        # ------------------------------------------
        # TOP 5 ACTIVE USERS
        # ------------------------------------------

        cursor.execute(f"""

            SELECT

                u.id,

                u.username,

                u.email,

                COUNT(p.id) AS total_predictions

            FROM users u

            INNER JOIN predictions p

            ON u.id = p.user_id

            {where_clause}

            GROUP BY
                u.id,
                u.username,
                u.email

            ORDER BY total_predictions DESC

            LIMIT 5

        """, params)

        top_users = cursor.fetchall()

        # ------------------------------------------
        # RECENT ACTIVITY
        # ------------------------------------------

        cursor.execute(f"""

            SELECT

                p.id,

                u.username,

                p.prediction,

                p.probability,

                p.status,

                p.created_at

            FROM predictions p

            INNER JOIN users u

            ON p.user_id = u.id

            {where_clause}

            ORDER BY p.created_at DESC

            LIMIT 10

        """, params)

        recent_activity = cursor.fetchall()

        # ------------------------------------------
        # ALL USERS FOR FILTER
        # ------------------------------------------

        cursor.execute("""

            SELECT

                id,

                username,

                email

            FROM users

            ORDER BY username

        """)

        users = cursor.fetchall()

        # ------------------------------------------
        # CUSTOMER DEMOGRAPHIC DATA
        # ------------------------------------------

        cursor.execute(f"""

            SELECT

                p.input_data,

                p.prediction,

                p.probability

            FROM predictions p

            {where_clause}

            AND p.input_data IS NOT NULL

        """ if where_clause else """

            SELECT

                input_data,

                prediction,

                probability

            FROM predictions

            WHERE input_data IS NOT NULL

        """, params)

        demographic_rows = cursor.fetchall()

        demographics = build_demographic_analytics(
            demographic_rows
        )

    except ValueError as e:

        flash(
            str(e),
            "error"
        )

        return redirect(
            url_for("admin_analytics")
        )

    finally:

        cursor.close()

        conn.close()

    return render_template(

        "admin_analytics.html",

        username=session["username"],

        total_users=total_users,

        total=total,

        subscribed=subscribed,

        not_subscribed=not_subscribed,

        avg_probability=avg_probability * 100,

        conversion_rate=conversion_rate,

        prediction_labels=prediction_labels,

        prediction_values=prediction_values,

        daily_labels=daily_labels,

        daily_values=daily_values,

        monthly_labels=monthly_labels,

        monthly_values=monthly_values,

        top_users=top_users,

        recent_activity=recent_activity,

        users=users,

        demographics=demographics,

        start_date=start_date,

        end_date=end_date,

        selected_user=selected_user

    )


# ==================================================
# EXPORT ANALYTICS CSV
# ==================================================

@app.route("/admin/export")

@admin_required

def export_analytics():

    conn = get_db()

    cursor = conn.cursor(dictionary=True)

    try:

        (
            where_clause,
            params,
            start_date,
            end_date,
            selected_user
        ) = get_analytics_filters()

        cursor.execute(f"""

            SELECT

                p.id AS prediction_id,

                u.username,

                u.email,

                p.prediction,

                p.probability,

                p.status,

                p.input_data,

                p.created_at

            FROM predictions p

            INNER JOIN users u

            ON p.user_id = u.id

            {where_clause}

            ORDER BY p.created_at DESC

        """, params)

        rows = cursor.fetchall()

        for row in rows:

            if isinstance(row["input_data"], dict):

                row["input_data"] = json.dumps(
                    row["input_data"]
                )

        df = pd.DataFrame(rows)

        if not df.empty:

            df["prediction"] = df["prediction"].map({

                1: "Subscribed",

                0: "Not Subscribed"

            })

            df["probability"] = (
                df["probability"] * 100
            ).round(2)

        output = BytesIO()

        df.to_csv(
            output,
            index=False,
            encoding="utf-8-sig"
        )

        output.seek(0)

        filename = (
            "bank_marketing_analytics_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + ".csv"
        )

        return send_file(

            output,

            mimetype="text/csv",

            as_attachment=True,

            download_name=filename

        )

    except ValueError as e:

        flash(str(e), "error")

        return redirect(
            url_for("admin_analytics")
        )

    finally:

        cursor.close()

        conn.close()


# ==================================================
# LOGOUT
# ==================================================

@app.route("/logout")

def logout():

    session.clear()

    return redirect(
        url_for("login")
    )


# ==================================================
# ERROR HANDLING
# ==================================================

@app.errorhandler(413)

def file_too_large(error):

    flash(
        "File size cannot exceed 10 MB.",
        "error"
    )

    return redirect(
        url_for("batch_predict_route")
    )


# ==================================================
# RUN APPLICATION
# ==================================================

if __name__ == "__main__":

    init_db()

    app.run(
        debug=True
    )
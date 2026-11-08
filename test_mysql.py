import mysql.connector

from dotenv import load_dotenv

import os

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")


try:

    conn = mysql.connector.connect(

        host=os.getenv("DB_HOST"),

        port=int(os.getenv("DB_PORT", 3306)),

        user=os.getenv("DB_USER"),

        password=os.getenv("DB_PASSWORD"),

        database=os.getenv("DB_NAME")

    )

    print("MySQL Connected Successfully!")

    cursor = conn.cursor()

    cursor.execute("SHOW TABLES")

    tables = cursor.fetchall()

    print("\nAvailable Tables:")

    for table in tables:

        print(table[0])

    cursor.close()

    conn.close()

except mysql.connector.Error as e:

    print("MySQL Connection Failed!")

    print(e)
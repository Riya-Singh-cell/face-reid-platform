import sqlite3

conn = sqlite3.connect("attendance.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    date TEXT,
    time TEXT
)
""")

conn.commit()


def mark_attendance(name, date, time):
    cursor.execute("""
    SELECT * FROM attendance 
    WHERE name=? AND date=?
    """, (name, date))

    result = cursor.fetchone()

    if result is None:
        cursor.execute("""
        INSERT INTO attendance (name, date, time)
        VALUES (?, ?, ?)
        """, (name, date, time))
        conn.commit()
        return True

    return False
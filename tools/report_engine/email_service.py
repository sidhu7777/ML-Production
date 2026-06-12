import os
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT") or 587)
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
BASE_URL = os.getenv("BASE_URL")  # VERY IMPORTANT


def _ensure_smtp_config():
    missing = [
        name
        for name, value in {
            "SMTP_HOST": SMTP_HOST,
            "SMTP_USER": SMTP_USER,
            "SMTP_PASS": SMTP_PASS,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing SMTP configuration: {', '.join(missing)}")


def send_report_ready_email(
    to_email: str,
    user_name: str,
    project_name: str,
    report_id: str,
    download_url: str | None = None,
):
    _ensure_smtp_config()

    if not download_url:
        base = (BASE_URL or "").rstrip("/")
        download_url = f"{base}/api/report/download/{report_id}"

    msg = EmailMessage()
    msg["From"] = SMTP_USER
    msg["To"] = to_email
    msg["Subject"] = f"Drive Test Report Ready – {project_name}"

    msg.set_content(
        f"""Hello {user_name},

Your drive test report for {project_name} has been generated successfully.

You can download the report using the link below:
{download_url}

Regards,
Network Analytics Team
"""
    )

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)

import smtplib
from email.mime.text import MIMEText

def send_email_alert(subject, body, to_email):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = "alerts@example.com"
    msg['To'] = to_email

    try:
        with smtplib.SMTP('localhost') as server:
            server.sendmail("alerts@example.com", [to_email], msg.as_string())
        print(f"Email alert sent to {to_email}")
    except Exception as e:
        print(f"Email alert failed: {e}")
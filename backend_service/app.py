import base64
import logging
import smtplib
import ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from flask import Flask, jsonify, g, request
from twilio.rest import Client
import yaml

app = Flask(__name__)


def init_config():
    with open("alert_service.yaml", "r") as yaml_config:
        config = yaml.load(yaml_config)
    return config


def get_app_config():
    return app.config['alert_service_config']


def setup_sms_client():
    config = get_app_config()

    # SMS an alert using twilio
    sms_account_sid = config['sms_service']['account_sid']
    sms_auth_token = config['sms_service']['auth_token']
    sms_client = Client(sms_account_sid, sms_auth_token)

    return sms_client


@app.route('/api_list', methods=['GET'])
def api_list():
    api_list_items = [
        {
            "id": "sms",
            "description" : "SMS service"
        },
        {
            "id": "email",
            "description": "Email service"
        }
    ]

    app.logger.info("show api_list:" + str(api_list_items))

    return jsonify({"api": api_list_items})


@app.route('/sms', methods=['POST'])
def sms_api():
    config = get_app_config()
    sms_client = setup_sms_client()
    sms_message = request.json['sms_message']
    target_phone = request.json['target_phone']

    from_phone = config['sms_service']['from_phone']
    to_phone = config['sms_service']['to_phone'] if target_phone == 'empty' else target_phone

    try:
        sms_results = sms_client.api.account.messages.create(
            body=sms_message, from_=from_phone, to=to_phone
        )

        resp_message = str(sms_results)
    except Exception:
        resp_message = 'Failed to send sms'
        logging.error("Failed to send sms", exc_info=True)

    return jsonify({"response_message": resp_message})

@app.route('/email', methods=['POST'])
def email_api():
    config = get_app_config()

    smtp_server = config['email_service']['smtp_server']
    smtp_port = int(config['email_service']['smtp_port'])
    smtp_login = config['email_service']['smtp_login']
    smtp_password = config['email_service']['smtp_password']
    from_account = config['email_service']['from_account']
    to_accounts = config['email_service']['to_accounts']
    subject_template = config['email_service']['email_template']['email_subject']
    body_template = config['email_service']['email_template']['email_body']

    alert_type = request.json['alert_type']
    subject_message = request.json['subject_message']
    body_message = request.json['body_message']
    # Total number of attachment files
    total_att_files = int(request.json['total_att_files'])

    # Create a multipart message and set headers
    email_message = MIMEMultipart()
    email_message["From"] = from_account
    email_message["To"] = to_accounts
    email_message["Subject"] = subject_template.format(alert_type=alert_type, subject_message=subject_message)
    email_body = body_template.format(body_message=body_message)

    # Add body to email
    email_message.attach(MIMEText(email_body, "plain"))

    if total_att_files > 0:
        att_file_list = request.json['att_files']
        for att_file in att_file_list:
            att_file_data = base64.decodebytes(att_file['file_data'].encode('ascii'))
            att_file_name = att_file['file_name']

            part = MIMEBase('application', 'octet-stream')
            part.set_payload(att_file_data)
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', 'attachment; filename="%s"' % att_file_name)
            email_message.attach(part)

    try:
        context = ssl.create_default_context()

        with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as server:
            server.login(smtp_login, smtp_password)
            server.sendmail(from_account, to_accounts, email_message.as_string())

        resp_message = 'Sent OK!'
    except Exception:
        resp_message = 'Failed to send email'
        logging.error("Failed to send email", exc_info=True)

    return jsonify({"response_message": resp_message})


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
                        handlers=[
                            logging.FileHandler("logs/alert_service.log"),
                            logging.StreamHandler()
                        ])
    app.logger.info("Starting backend_service")
    app.config['alert_service_config'] = init_config()
    app.run(host='127.0.0.1', port=4980)

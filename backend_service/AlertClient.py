import base64
import json
import cv2
import numpy as np
from os import path

import requests
import logging
import time

logging.basicConfig(level=logging.DEBUG,
                    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
                    handlers=[
                        logging.FileHandler("logs/alert_client.log"),
                        logging.StreamHandler()
                    ])


class Email_Att(dict):
    def __init__(self, file_name, file_data):
        self.file_name = file_name
        self.file_data = file_data
        dict.__init__(self, file_name=file_name, file_data=file_data)


class Email_Msg(dict):
    def __init__(self, alert_type, subject_message, body_message, att_files):
        if att_files is None:
            total_att_files = 0
            current_att_files = []
        else:
            total_att_files = len(att_files)
            current_att_files = att_files

        dict.__init__(self, alert_type=alert_type, subject_message=subject_message,
                      body_message=body_message, total_att_files=total_att_files,
                      att_files=current_att_files)


class Intruder_Status(dict):
    def __init__(self, detected_status):
        dict.__init(self, instruder_detected=detected_status)


def _url(path):
    return 'http://127.0.0.1:4980' + path


def show_api_list():
    return requests.get(_url('/api_list'))


# API: Check whether alert service was running
def check_service(api_function_name, api_function):
    response = api_function()
    resp_dict = response.json()
    logging.info(api_function_name + ":" + str(resp_dict))

# API: Inform Intruder status message
def inform_intruder_status(intruder_detected=True):
    response = requests.post(_url('/report_intruder'), json={
        'intruder_status': str(intruder_detected),
    })

    resp_message = str(response.json())
    inform_results = not ('Failed to put status' in resp_message)
    logging.info("inform_intruder_status (detected: {}), Result: {}, details: {}'".format(
        intruder_detected, inform_results, resp_message))

    return inform_results

# API: Query Intruder status message
def query_intruder_status():
    response = requests.post(_url('/query_intruder'), json={
        'intruder_status': 'check',
    })

    resp_message = str(response.json())
    query_results = not ('Failed to query status' in resp_message)
    logging.info("query_intruder_status Result: {}, details: {}'".format(query_results, resp_message))

    if 'success' in resp_message:
        return True
    else:
        return False

# API: Send SMS message
def send_sms(sms_message, target_phone='empty'):
    response = requests.post(_url('/sms'), json={
        'sms_message': sms_message,
        'target_phone': target_phone
    })

    resp_message = str(response.json())
    sms_results = not ('Failed to send sms' in resp_message)
    logging.info("send_sms Result: {}, details: {}'".format(sms_results, resp_message))

    return sms_results


def read_file_binary(filepath):
    with open(filepath, 'rb') as infile:
        file_data = infile.read()

    return file_data


# API: Send email with attachment files
def send_email_with_files(email_content, subject_message, alert_type='Notice',
                          attachment_filepath_list=[]):
    if len(attachment_filepath_list) == 0:
        return send_email(email_content, subject_message, alert_type)

    att_file_names = []
    att_file_data_list = []

    for att_file_path in attachment_filepath_list:
        _, att_filename = path.split(att_file_path)
        att_file_data = read_file_binary(att_file_path)
        att_file_names.append(att_filename)
        att_file_data_list.append(att_file_data)

    return send_email(email_content, subject_message, alert_type,
                      att_file_names, att_file_data_list)


def test_send_email_with_images(email_content, subject_message, alert_type='Notice',
                                attachment_filepath_list=[]):
    if len(attachment_filepath_list) == 0:
        return send_email(email_content, subject_message, alert_type)

    image_data_list = []

    for att_file_path in attachment_filepath_list:
        _, att_filename = path.split(att_file_path)
        image_bytes = read_file_binary(att_file_path)
        image_np_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        image_data = cv2.imdecode(image_np_array, cv2.IMREAD_COLOR)
        image_data_list.append(image_data)

    return send_email_with_images(email_content, subject_message, alert_type,
                                  image_name_prefix="unknown_", image_list=image_data_list)


# API: Send email with images
def send_email_with_images(email_content, subject_message, alert_type='Notice',
                           image_name_prefix='image_', image_list=[]):
    if len(image_list) == 0:
        return send_email(email_content, subject_message, alert_type)

    att_file_names = []
    att_file_data_list = []
    att_file_index = 0

    for image in image_list:
        att_file_index = att_file_index + 1
        is_success, im_buf_arr = cv2.imencode(".jpg", image)
        att_file_data = im_buf_arr.tobytes()
        att_file_names.append(f"{image_name_prefix}{att_file_index}.jpg")
        att_file_data_list.append(att_file_data)

    return send_email(email_content, subject_message, alert_type,
                      att_file_names, att_file_data_list)


# API: Send email with / without attachment files
def send_email(email_content, subject_message, alert_type='Notice',
               attachment_filenames=[], attachment_files_data=[]):
    att_files = []
    if 0 < len(attachment_filenames) == len(attachment_files_data) > 0:
        logging.info('Add {} attachment files'.format(len(attachment_filenames)))
        for i in range(len(attachment_filenames)):
            att_file_name = attachment_filenames[i]
            att_file_data = base64.b64encode(attachment_files_data[i]).decode('ascii')
            email_att_file = Email_Att(file_name=att_file_name, file_data=att_file_data)
            att_files.append(email_att_file)
    else:
        if len(attachment_filenames) == 0:
            logging.info('No attachment file')
        else:
            logging.info('Error in attachment files parameter (names={}, data={})'.format(
                len(attachment_filenames), len(attachment_files_data)
            ))

    email_msg_obj = Email_Msg(alert_type=alert_type, subject_message=subject_message,
                              body_message=email_content, att_files=att_files)

    if logging.getLogger().isEnabledFor(logging.DEBUG):
        email_json = json.dumps(email_msg_obj, default=lambda o: o.__dict__, indent=4)
        logging.debug('email_json: {}'.format(email_json))

    response = requests.post(_url('/email'), json=email_msg_obj)
    resp_message = str(response.json())
    email_results = not ('Failed to send email' in resp_message)
    logging.info("send_email Result: {}, details: {}'".format(email_results, resp_message))

    return email_results


if __name__ == '__main__':
    check_service("show_api_list", show_api_list)

    # Check sms API
    # send_sms("test sms")

    # Check email API (without attachments)
    # send_email('Test sending mail.', 'Test email', 'Alert')

    # send email API (with attachements)
    # send_email_with_files('Test sending mail.', 'Test email', 'Alert',
    #                       ['img/test1.png', 'img/test2.png', 'img/test2.jpg'])
    # test_send_email_with_images('Test sending mail with images.', 'Test email with images', 'Alert',
    #                             ['img/test1.png', 'img/test2.png', 'img/test2.jpg'])

    # Report intruder status
    inform_intruder_status(intruder_detected=True)

    time.sleep(1)

    intruder_found = query_intruder_status()

    time.sleep(1)

    intruder_found = query_intruder_status()

    time.sleep(1)

    inform_intruder_status(intruder_detected=False)

    time.sleep(1)

    intruder_found = query_intruder_status()

    time.sleep(1)

    intruder_found = query_intruder_status()
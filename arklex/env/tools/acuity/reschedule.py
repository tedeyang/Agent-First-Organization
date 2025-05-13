import json
import requests
from requests.auth import HTTPBasicAuth
from arklex.env.tools.tools import register_tool, logger
from arklex.env.tools.acuity.utils import EXCEPTIONS

description = "Help the user to reschedule the appointment"
slots = [
    {
        "name": "apt_id",
        "type": "string",
        "description": "The appointment id of the info session and it should be consisted of numbers. e.g. 76474933",
        "prompt": "Which info session would you like to attend?",
        "required": True,
    },
    {
        "name": "time",
        "type": "string",
        "description": "The time of the info session the user wants to reschedule. It could be like Apr 19th 13:00. If you are not sure, ask them to confirm. The final format is like: 2025-04-19T13:00:00-0400",
        "prompt": "What is the original date of your appointment?",
        "required": True,
    }
]
outputs = [
    {
        "name": "res_apt",
        "type": "string",
        "description": "The rescheduled appointment information after the user reschedules.",
    }
]

CREDENTIAL_NOT_FOUND = 'error: missing credential information'
errors = [
    EXCEPTIONS,
    CREDENTIAL_NOT_FOUND
]

@register_tool(description, slots, outputs)
def get_available_date(apt_id, time, **kwargs):
    user_id = kwargs.get('ACUITY_USER_ID')
    api_key = kwargs.get('ACUITY_API_KEY')
    if not api_key or not user_id:
        return CREDENTIAL_NOT_FOUND
    base_url = 'https://acuityscheduling.com/api/v1/appointments'.format(apt_id)
    body = {
        "datetime": time,
    }

    response = requests.put(base_url, json=body, auth=HTTPBasicAuth(user_id, api_key))

    if response.status_code == 200:
        data = response.json()
        return json.dumps(data)
    else:
        return EXCEPTIONS

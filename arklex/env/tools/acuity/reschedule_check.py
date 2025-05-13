import json
import requests
from requests.auth import HTTPBasicAuth
from arklex.env.tools.tools import register_tool, logger
from arklex.env.tools.acuity.utils import EXCEPTIONS

description = "Check the appointment whether exists in the appointment list"
slots = [
    {
        "name": "apt_name",
        "type": "string",
        "description": "The appointment name of the info session. It allow user to input some parts of the name, but if you are unsure, ask the user to confirm.",
        "prompt": "Which info session would you like to reschedule?",
        "required": True,
    },
    {
        "name": "rdate",
        "type": "string",
        "description": "The original date of the info session the user currently appointment. This will help to find the correct one when there are multiple appointmemts. It should consist of year, month, day. e.g. 2025-04-12. If you are not sure about the user's input, ask them to confirm. In addition, transform the pattern into this format: \'April 7, 2025\'",
        "prompt": "What is the original date of your appointment?",
        "required": True,
    },
    {
        "name": "apt_ls",
        "type": "string",
        "description": "The all appointment information of the specific users. It should be like JSON format.",
        "prompt": "",
        "required": True,
    }
]
outputs = [
    {
        "name": "apt_info",
        "type": "string",
        "description": "The appointment information the user wants to reschedule.",
    }
]

CREDENTIAL_NOT_FOUND = 'error: missing credential information'
APT_NOT_FOUND = 'error: no appointment found'
errors= [
    EXCEPTIONS,
    CREDENTIAL_NOT_FOUND,
    APT_NOT_FOUND
]
@register_tool(description, slots, outputs, lambda x: x not in errors)
def reschedule_check(apt_name, rdate, apt_ls, **kwargs):
    apt_ls = json.loads(apt_ls)
    apt = [item for item in apt_ls if item.get("type") == apt_name and item.get("date") == rdate]
    if len(apt) == 0:
        return APT_NOT_FOUND

    return apt[0]



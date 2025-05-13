import inspect
import json
import requests
from requests.auth import HTTPBasicAuth

from arklex.env.tools.acuity._exception_prompt import AcuityExceptionPrompt
from arklex.env.tools.tools import register_tool, logger
from arklex.env.tools.acuity.utils import authenticate_acuity
from arklex.exceptions import ToolExecutionError

description = "Make an appointment for the info session"
slots = [
    {
        "name": "fname",
        "type": "string",
        "description": "The first name of the user.",
        "prompt": "Could you please give me your first name?",
        "required": True,
    },
    {
        "name": "lname",
        "type": "string",
        "description": "The last name of the user.",
        "prompt": "Could you please give me your last name?",
        "required": True,
    },
    {
        "name": "email",
        "type": "string",
        "description": "The email of the student. e.g. sarah@gmail.com",
        "prompt": "Could you please give me your email to check",
        "required": True,
    },
    {
        "name": "book_time",
        "type": "string",
        "description": "The time of the info session the user wants to book. It could be like Apr 19th 13:00. If you are not sure, ask them to confirm. The final format is like: 2025-04-19T13:00:00-0400.",
        "prompt": "What time would you like to book?",
        "required": True,
    },
    {
        "name": "apt_tid",
        "type": "string",
        "description": "The appointment type id of the info session and it should be consisted of numbers. e.g. 76474933",
        "prompt": "Which info session would you like to attend?",
        "required": True,
    }
]
outputs = [
    {
        "name": "apt_binfo",
        "type": "string",
        "description": "The booking appointment information.",
    }
]

@register_tool(description, slots, outputs)
def book_info_session(fname, lname, email, book_time, apt_tid, **kwargs):
    func_name = inspect.currentframe().f_code.co_name
    user_id, api_key = authenticate_acuity(kwargs)

    body = {
      "firstName": fname,
      "lastName": lname,
      "email": email,
      "datetime": book_time,
      "appointmentTypeID": apt_tid
    }
    base_url = 'https://acuityscheduling.com/api/v1/appointments'
    response = requests.post(base_url, json=body, auth=HTTPBasicAuth(user_id, api_key))

    if response.status_code == 200:
        data = response.json()
        return json.dumps(data)
    else:
        raise ToolExecutionError(func_name, AcuityExceptionPrompt.BOOK_SESSION_EXCEPTION_PROMPT)



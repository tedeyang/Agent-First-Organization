import json
from pprint import pprint

import requests
import inspect

from arklex.env.tools.acuity._exception_prompt import AcuityExceptionPrompt
from arklex.exceptions import ToolExecutionError
from requests.auth import HTTPBasicAuth

from arklex.env.tools.tools import register_tool, logger
from arklex.env.tools.acuity.utils import authenticate_acuity

description = "Help the user to reschedule the appointment"
slots = [
    {
        "name": "apt_id",
        "type": "str",
        "description": "The appointment id of the info session and it should be consisted of numbers. e.g. 1470211171. NOTE THAT IT IS NOT TYPE ID.",
        "prompt": "",
        "required": True,
        "verified": True,
    },
    {
        "name": "time",
        "type": "str",
        "description": "The time of the info session the user wants to reschedule. It could be like Apr 19th 13:00. If you are not sure, ask them to confirm. The final format is like: 2025-04-19T13:00:00-0400",
        "prompt": "What is the original date of your appointment?",
        "required": True,
    }
]
outputs = [
    {
        "name": "res_apt",
        "type": "list[dict]",
        "description": "The rescheduled appointment information after the user reschedules.",
    }
]


@register_tool(description, slots, outputs)
def reschedule(apt_id, time, **kwargs):
    func_name = inspect.currentframe().f_code.co_name
    user_id, api_key = authenticate_acuity(kwargs)


    base_url = 'https://acuityscheduling.com/api/v1/appointments/{}/reschedule'.format(apt_id)
    body = {
        "datetime": time,
    }

    response = requests.put(base_url, json=body, auth=HTTPBasicAuth(user_id, api_key))
    pprint(response.json())
    if response.status_code == 200:
        data = response.json()
        return json.dumps(data)
    else:
        return ToolExecutionError(func_name, AcuityExceptionPrompt.RESCHEDULE_PROMPT)

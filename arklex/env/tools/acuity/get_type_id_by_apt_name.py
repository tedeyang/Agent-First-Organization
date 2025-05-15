import inspect
import json
from pprint import pprint

import requests
from requests.auth import HTTPBasicAuth

from arklex.env.tools.acuity._exception_prompt import AcuityExceptionPrompt
from arklex.env.tools.tools import register_tool, logger
from arklex.env.tools.acuity.utils import authenticate_acuity
from arklex.exceptions import ToolExecutionError

description = "Retrieve the list of all info sessions for users."

slots = [
    {
        "name": "apt_name",
        "type": "str",
        "description": "The appointment name of the info session. USER NEEDS TO INPUT IT. It allow user to input some parts of the name, but if you are unsure, ask the user to confirm.",
        "prompt": "Which info session would you like to cancel?",
        "required": True,
    },
]
outputs = [
    {
        "name": "apt_type_id",
        "type": "str",
        "description": "The appointment type id of the info session.",
    }
]

@register_tool(description, slots, outputs)
def get_type_id_by_apt_name(apt_name, **kwargs):
    func_name = inspect.currentframe().f_code.co_name
    user_id, api_key = authenticate_acuity(kwargs)

    base_url = 'https://acuityscheduling.com/api/v1/appointment-types'

    response = requests.get(base_url, auth=HTTPBasicAuth(user_id, api_key))

    if response.status_code == 200:
        data = response.json()
        apt_type_id = next((item['id'] for item in data if item['name'] == apt_name), None)
        response_str = f"The appointment type id is {apt_type_id}\n"
        return response_str
    else:
        raise ToolExecutionError(func_name, AcuityExceptionPrompt.GET_TYPE_ID_PROMPT)





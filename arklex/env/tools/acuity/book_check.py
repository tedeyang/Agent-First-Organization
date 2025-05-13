import inspect
import json
import requests
from requests.auth import HTTPBasicAuth

from arklex.env.tools.acuity._exception_prompt import AcuityExceptionPrompt
from arklex.env.tools.tools import register_tool, logger
from arklex.exceptions import ToolExecutionError

description = "Check whether the info session exists in the available list"
slots = [
    {
        "name": "apt_name",
        "type": "string",
        "description": "The name of the info session (appointment type). It allows user to input some parts of the name, but if you are unsure, ask the user to confirm.",
        "prompt": "Which info session would you like to reschedule?",
        "required": True,
    },
    {
        "name": "sessions",
        "type": "dict",
        "description": "All available information sessions. In sessions it should like \"{'sessions': [{'id': 76850002, 'name': 'Test N1', 'schedulingUrl': 'https://app.acuityscheduling.com/schedule.php?owner=35334298&appointmentType=76474933'}]}\".",
        "prompt": "",
        "required": True,
    }
]
outputs = [
    {
        "name": "apt_tid",
        "type": "string",
        "description": "The appointment type id of the info session",
    }
]

@register_tool(description, slots, outputs)
def book_check(apt_name, sessions, **kwargs):
    func_name = inspect.currentframe().f_code.co_name
    apt = [item.get('id') for item in sessions if item.get("name") == apt_name]
    if len(apt) == 0:
        raise ToolExecutionError(func_name, AcuityExceptionPrompt.AVAILABLE_TYPES_EXCEPTION_PROMPT)

    return apt[0]



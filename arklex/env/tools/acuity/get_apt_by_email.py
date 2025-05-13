import json
import requests
import inspect
from datetime import datetime

from requests.auth import HTTPBasicAuth
from arklex.env.tools.tools import register_tool, logger
from arklex.env.tools.acuity.utils import authenticate_acuity
from arklex.exceptions import ToolExecutionError
from arklex.env.tools.acuity._exception_prompt import AcuityExceptionPrompt


description = "Get the list of all information sessions"

slots = [
    {
        "name": "email",
        "type": "str",
        "description": "The email of the student. e.g. sarah@gmail.com",
        "prompt": "Could you please give me your email to check your appointment?",
        "required": True,
    }
]
outputs = [
    {
        "name": "apt_ls",
        "type": "list[dict]",
        "description": "All appointment information about the specific user.",
    }
]

@register_tool(description, slots, outputs)
def get_apt_by_email(email, **kwargs):
    func_name = inspect.currentframe().f_code.co_name
    user_id, api_key = authenticate_acuity(kwargs)

    base_url = 'https://acuityscheduling.com/api/v1/appointments'

    response = requests.get(base_url, auth=HTTPBasicAuth(user_id, api_key))
    apt_ls = []
    if response.status_code == 200:
        data = response.json()
        today = datetime.now().date()
        for item in data:
            if item.get("email") == email:
                apt_date = datetime.strptime(item["date"], "%B %d, %Y").date()
                if apt_date > today:
                    apt_ls.append({
                        "id": item["id"],
                        "date": item["date"],
                        "time": item["time"],
                        "endtime": item["endTime"],
                        "type": item["type"],
                    })
        if not apt_ls:
            raise ToolExecutionError(func_name, AcuityExceptionPrompt.GET_APT_BY_EMAIL_EXCEPTION_PROMPT_1)
        else:
            return json.dumps(apt_ls)
    else:
        raise ToolExecutionError(func_name, AcuityExceptionPrompt.GET_APT_BY_EMAIL_EXCEPTION_PROMPT_2)



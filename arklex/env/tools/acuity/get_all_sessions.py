import json
import requests
import inspect

from requests.auth import HTTPBasicAuth
from arklex.env.tools.tools import register_tool, logger
from arklex.env.tools.acuity.utils import authenticate_acuity




description = "Get the list of all information sessions"

slots = [
]
outputs = [
    {
        "name": "session_ls",
        "type": "list[dict]",
        "description": "All available information sessions",
    }
]


@register_tool(description, slots, outputs)
def get_all_sessions(**kwargs):
    func_name = inspect.currentframe().f_code.co_name
    user_id, api_key = authenticate_acuity(kwargs)

    base_url = 'https://acuityscheduling.com/api/v1/appointments'

    response = requests.get(base_url, auth=HTTPBasicAuth(user_id, api_key))

    if response.status_code == 200:
        data = response.json()
        return json.dumps(data)
    else:
        return EXCEPTIONS



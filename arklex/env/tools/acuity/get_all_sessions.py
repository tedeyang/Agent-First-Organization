import json
import requests
from requests.auth import HTTPBasicAuth
from arklex.env.tools.tools import register_tool, logger
from arklex.env.tools.acuity.utils import EXCEPTIONS


description = "Get the list of all information sessions"

slots = [
]
outputs = [
    {
        "name": "session_ls",
        "type": "string",
        "description": "All available information sessions",
    }
]
CREDENTIAL_NOT_FOUND = 'error: missing credential information'
errors= [
    EXCEPTIONS
]

@register_tool(description, slots, outputs, lambda x: x not in errors)
def get_all_sessions(**kwargs):
    user_id = kwargs.get('ACUITY_USER_ID')
    api_key = kwargs.get('ACUITY_API_KEY')
    if not api_key or not user_id:
        return CREDENTIAL_NOT_FOUND

    base_url = 'https://acuityscheduling.com/api/v1/appointments'

    response = requests.get(base_url, auth=HTTPBasicAuth(user_id, api_key))

    if response.status_code == 200:
        data = response.json()
        return json.dumps(data)
    else:
        return EXCEPTIONS



import inspect
import requests
from requests.auth import HTTPBasicAuth

from arklex.env.tools.acuity._exception_prompt import AcuityExceptionPrompt
from arklex.env.tools.tools import register_tool, logger
from arklex.env.tools.acuity.utils import authenticate_acuity
from arklex.exceptions import ToolExecutionError

description = "Retrieve the list of all info sessions for users."

slots = []
outputs = [
    {
        "name": "sessions",
        "type": "list[dict]",
        "description": "All available information sessions. In sessions it should like \"[{'apt_type_id': 76850002, 'name': 'Test N1'}]\".",
    }
]


@register_tool(description, slots, outputs)
def get_session_types(**kwargs):
    func_name = inspect.currentframe().f_code.co_name
    user_id, api_key = authenticate_acuity(kwargs)

    base_url = "https://acuityscheduling.com/api/v1/appointment-types"

    response = requests.get(base_url, auth=HTTPBasicAuth(user_id, api_key))

    if response.status_code == 200:
        data = response.json()
        response_text = "The information below is the details of all info session types. You must include the list of all session names in the following response at least.\n"

        for session in data:
            response_text += (
                f"Info session ID (Appointment Type ID):{session.get('id')}\n"
            )
            response_text += f"Info session Name: {session.get('name')}\n"
        return response_text
    else:
        raise ToolExecutionError(
            func_name, AcuityExceptionPrompt.AVAILABLE_TYPES_EXCEPTION_PROMPT
        )

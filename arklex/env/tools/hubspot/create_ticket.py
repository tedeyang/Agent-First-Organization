"""
Tool for creating support tickets via HubSpot in the Arklex framework.

This module implements a tool for creating support tickets for customers using the HubSpot API. It handles ticket creation, association with contacts, and is designed for integration with the Arklex tool system.
"""

import inspect
from datetime import datetime
from typing import Any

import hubspot
from hubspot.crm.associations.v4 import AssociationSpec
from hubspot.crm.objects.emails import ApiException
from hubspot.crm.tickets import SimplePublicObjectInput
from hubspot.crm.tickets.models import SimplePublicObjectInputForCreate

from arklex.env.tools.hubspot._exception_prompt import HubspotExceptionPrompt
from arklex.env.tools.hubspot.base.entities import HubspotAuth
from arklex.env.tools.hubspot.utils import authenticate_hubspot
from arklex.env.tools.tools import register_tool
from arklex.utils.exceptions import ToolExecutionError
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)

# Tool description for creating support tickets
description: str = "Create a ticket for the existing customer when the customer has some problem about the specific product."

# List of required parameters for the tool
slots: list[dict[str, Any]] = [
    {
        "name": "cus_cid",
        "type": "str",
        "description": "The id of the customer contact. It consists of all numbers. e.g. 97530152525",
        "prompt": "",
        "required": True,
        "verified": True,
    },
    {
        "name": "issue",
        "type": "str",
        "description": "The question that the customer has for the specific product",
        "prompt": "",
        "required": True,
        "verified": True,
    },
    {
        "name": "cus_fname",
        "type": "str",
        "description": "The first name of the customer contact.",
        "prompt": "",
        "required": True,
        "verified": True,
    },
    {
        "name": "cus_lname",
        "type": "str",
        "description": "The last name of the customer contact.",
        "prompt": "",
        "required": True,
        "verified": True,
    },
]


@register_tool(description, slots)
def create_ticket(
    cus_cid: str,
    issue: str,
    cus_fname: str,
    cus_lname: str,
    auth: HubspotAuth,
    **kwargs: dict[str, Any],
) -> str:
    """
    Create a support ticket for a customer and associate it with their contact record.

    Args:
        cus_cid (str): Customer contact ID
        issue (str): Description of the customer's issue
        **kwargs (Dict[str, Any]): Additional keyword arguments

    Returns:
        str: ID of the created ticket

    Raises:
        ToolExecutionError: If ticket creation or association fails
    """
    func_name: str = inspect.currentframe().f_code.co_name
    access_token: str = authenticate_hubspot(auth)

    api_client: hubspot.Client = hubspot.Client.create(access_token=access_token)

    timestamp: str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-3] + "Z"
    subject_name: str = "Issue of " + cus_cid + " at " + timestamp
    ticket_properties: dict[str, Any] = {
        "hs_pipeline_stage": 1,
        "content": issue,
        "subject": subject_name,
    }
    ticket_for_create: SimplePublicObjectInputForCreate = (
        SimplePublicObjectInputForCreate(properties=ticket_properties)
    )
    try:
        ticket_creation_response: Any = api_client.crm.tickets.basic_api.create(
            simple_public_object_input_for_create=ticket_for_create
        )
        ticket_creation_response: dict[str, Any] = ticket_creation_response.to_dict()
        ticket_id: str = ticket_creation_response["id"]
        ticket_name: dict[str, Any] = {
            "subject": ticket_id + "_" + cus_fname + " " + cus_lname,
        }
        simple_public_object_input: SimplePublicObjectInput = SimplePublicObjectInput(
            properties=ticket_name
        )
        try:
            api_client.crm.tickets.basic_api.update(
                ticket_id=ticket_id,
                simple_public_object_input=simple_public_object_input,
            )
        except ApiException as e:
            log_context.info(f"Exception when calling Crm.tickets.update: {e}\n")
            raise ToolExecutionError(
                func_name, HubspotExceptionPrompt.TICKET_UPDATE_ERROR_PROMPT
            ) from e

        association_spec: list[AssociationSpec] = [
            AssociationSpec(
                association_category="HUBSPOT_DEFINED", association_type_id=15
            )
        ]
        try:
            api_client.crm.associations.v4.basic_api.create(
                object_type="contact",
                object_id=cus_cid,
                to_object_type="ticket",
                to_object_id=ticket_id,
                association_spec=association_spec,
            )
            return ticket_id
        except ApiException as e:
            log_context.info(f"Exception when calling AssociationV4: {e}\n")
            raise ToolExecutionError(
                func_name, HubspotExceptionPrompt.TICKET_CREATION_ERROR_PROMPT
            ) from e
    except ApiException as e:
        log_context.info(f"Exception when calling Crm.tickets.create: {e}\n")
        raise ToolExecutionError(
            func_name, HubspotExceptionPrompt.TICKET_CREATION_ERROR_PROMPT
        ) from e

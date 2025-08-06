"""
This module provides functionality to retrieve detailed information about a specific product
from the Shopify store using the Admin API. It supports retrieving comprehensive product details
including inventory, descriptions, variants, and category information.

The module:
1. Uses the Shopify Admin API to fetch product data
2. Supports pagination and cursor-based navigation
3. Handles authentication and error cases
4. Formats product information for easy consumption

Module Name: get_web_product

This file contains the code for retrieving product information using the Shopify Admin API.
"""

import inspect
import json

import shopify
from pydantic import BaseModel

from arklex.env.tools.shopify._exception_prompt import ShopifyExceptionPrompt
from arklex.env.tools.shopify.base.entities import ShopifyAdminAuth
from arklex.env.tools.shopify.legacy.utils_nav import cursorify
from arklex.env.tools.shopify.utils.utils import authorify_admin
from arklex.env.tools.shopify.utils.utils_slots import (
    ShopifyGetWebProductSlots,
)
from arklex.env.tools.tools import register_tool
from arklex.utils.exceptions import ToolExecutionError
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


class GetWebProductOutput(BaseModel):
    message_flow: str


@register_tool(
    description="Get the inventory information and description details of a product.",
    slots=ShopifyGetWebProductSlots.get_all_slots(),
)
def get_web_product(
    web_product_id: str, auth: ShopifyAdminAuth, **kwargs: object
) -> GetWebProductOutput:
    """
    Retrieve detailed information about a specific product using the Shopify Admin API.

    Args:
        web_product_id (str): The ID of the product to retrieve information for.
            Can be a full product ID or just the numeric portion.
        auth (ShopifyAdminAuth): Authentication credentials for the Shopify store.
        **kwargs: Additional keyword arguments for llm configuration.

    Returns:
        GetWebProductOutput: A formatted string containing detailed product information, including:
            - Product ID and title
            - Description and total inventory
            - Online store URL
            - Product options and values
            - Category name
            - Variant details (name, ID, price, inventory quantity)

    Raises:
        ToolExecutionError: If:
            - The product is not found
            - There's an error retrieving the product information
            - The API request fails

    Note:
        The function automatically extracts the numeric portion of the product ID
        if a full ID is provided (e.g., "gid://shopify/Product/123456" -> "123456").
    """
    func_name = inspect.currentframe().f_code.co_name
    auth = authorify_admin(auth)
    nav = cursorify(kwargs)

    try:
        with shopify.Session.temp(**auth):
            response = shopify.GraphQL().execute(f"""
                {{
                    products ({nav[0]}, query:"id:{web_product_id.split("/")[-1]}") {{
                        nodes {{
                            id
                            title
                            description
                            totalInventory
                            onlineStoreUrl
                            options {{
                                name
                                values
                            }}
                            category {{
                                name
                            }}
                            variants (first: 2) {{
                                nodes {{
                                    displayName
                                    id
                                    price
                                    inventoryQuantity
                                }}
                            }}
                        }}
                        pageInfo {{
                            endCursor
                            hasNextPage
                            hasPreviousPage
                            startCursor
                        }}
                    }}
                }}
            """)
            result: dict[str, list[dict[str, str]]] = json.loads(response)["data"][
                "products"
            ]
            response: list[dict[str, str]] = result["nodes"]
            if len(response) == 0:
                raise ToolExecutionError(
                    func_name,
                    extra_message=ShopifyExceptionPrompt.PRODUCT_NOT_FOUND_PROMPT,
                )
            product: dict[str, str] = response[0]
            response_text = ""
            response_text += f"Product ID: {product.get('id', 'None')}\n"
            response_text += f"Title: {product.get('title', 'None')}\n"
            response_text += f"Description: {product.get('description', 'None')}\n"
            response_text += (
                f"Total Inventory: {product.get('totalInventory', 'None')}\n"
            )
            response_text += f"Options: {product.get('options', 'None')}\n"
            response_text += f"Category: {product.get('category', {})}\n"
            response_text += "The following are several variants of the product:\n"
            for variant in product.get("variants", {}).get("nodes", []):
                response_text += f"Variant name: {variant.get('displayName', 'None')}, Variant ID: {variant.get('id', 'None')}, Price: {variant.get('price', 'None')}, Inventory Quantity: {variant.get('inventoryQuantity', 'None')}\n"
            response_text += "\n"

            return GetWebProductOutput(
                message_flow=response_text,
            )
    except Exception as e:
        raise ToolExecutionError(
            func_name,
            extra_message=ShopifyExceptionPrompt.PRODUCT_NOT_FOUND_PROMPT,
        ) from e

from arklex.exceptions import AuthenticationError

ACUITY_AUTH_ERROR = "Missing some or all required hubspot authentication parameters: ACUITY_USER_ID, ACUITY_API_KEY. Please set up 'fixed_args' in the config file. For example, {'name': <unique name of the tool>, 'fixed_args': {'token': <shopify_access_token>, 'shop_url': <shopify_shop_url>, 'api_version': <Shopify API version>}}"


def authenticate_acuity(kwargs):
    user_id = kwargs.get("ACUITY_USER_ID")
    api_key = kwargs.get("ACUITY_API_KEY")
    if not user_id and not api_key:
        raise AuthenticationError(ACUITY_AUTH_ERROR)

    return user_id, api_key

RESPOND_ACTION_NAME = "respond"
RESPOND_ACTION_FIELD_NAME = "content"

REACT_INSTRUCTION = """
# Instruction
You need to act as an agent that use a set of tools to help the user according to the policy.

# Conversation record
{conversation_record}

# Available tools
{available_tools}

Your generation should have exactly the following format:
Thought:
<A single line of reasoning to process the context and inform the decision making. Do not include extra lines.>
Action:
{{"name": <The name of the action>, "arguments": <The arguments to the action in json format>}}

You current task is:
{task}

Make the decision based on the current task, conversation record, and available tools. If the task has not been finished and available tools are helpful for the task, you should use the appropriate tool to finish it instead of directly give a response.

Thought:
"""

### REACT PLANNER PROMPTS

PLANNER_REACT_INSTRUCTION_ZERO_SHOT = """
# Instruction
Please act as an agent that selects the next appropriate action in a sequence of actions in order to satisfy the user's request.

# User message
{user_message}

# Available actions
{available_actions}

Your generation should have exactly the following format:
Thought:
<A single line of reasoning to process the context and inform the decision making. Do not include extra lines.>

Action:
{{"name": <The name of the action>, "arguments": <Any arguments that the action requires in json format>}}

If you must return a message to the user or ask them for more information, your response must do so using the action "{respond_action_name}", for example:
Action:
{{"name": "{respond_action_name}", "arguments": {{"content": <message to return to the user>}}}}

Never provide a response that is not in this format.

Your current task is:
{task}

Select the next action based on the current task, conversation record, and available actions.

Thought:
"""

PLANNER_REACT_INSTRUCTION_FEW_SHOT = """
# Instruction
Please act as an agent that selects the next appropriate action in a sequence of actions in order to satisfy the user's request.

Your generation should have exactly the following format:
Thought:
<A single line of reasoning to process the context and inform the decision making. Do not include extra lines.>

Action:
{{"name": <The name of the action>, "arguments": <Any arguments that the action requires in json format>}}

If you must return a message to the user or ask them for more information, your response must do so using the action "{respond_action_name}", for example:
Action:
{{"name": "{respond_action_name}", "arguments": {{"content": <message to return to the user>}}}}

Never provide a response that is not in this format.

For example: 
---

Select the next action based on the current task, conversation record, and available actions.

# User message
Can you please retrieve my user details? My email is sample-email@arklex.ai.

# Available actions
{{'FaissRAGWorker': {{'name': 'FaissRAGWorker', 'type': 'worker', 'description': "Answer the user's questions based on the company's internal documentations (unstructured text data), such as the policies, FAQs, and product information", 'parameters': [], 'required': []}}, 'MessageWorker': {{'name': 'MessageWorker', 'type': 'worker', 'description': 'The worker that used to deliver the message to the user, either a question or provide some information.', 'parameters': [], 'required': []}}, 'DefaultWorker': {{'name': 'DefaultWorker', 'type': 'worker', 'description': "Default worker decided by chat records if there is no specific worker for the user's query", 'parameters': [], 'required': []}}, 'shopify-find_user_id_by_email-find_user_id_by_email': {{'name': 'shopify-find_user_id_by_email-find_user_id_by_email', 'type': 'tool', 'description': 'Find user id by email. If the user is not found, the function will return an error message.', 'parameters': [{{'user_email': {{'type': 'string', 'description': "The email of the user, such as 'something@example.com'."}}}}], 'required': ['user_email']}}, 'shopify-get_user_details_admin-get_user_details_admin': {{'name': 'shopify-get_user_details_admin-get_user_details_admin', 'type': 'tool', 'description': 'Get the details of a user with Admin API.', 'parameters': [{{'user_id': {{'type': 'string', 'description': "The user id, such as 'gid://shopify/Customer/13573257450893'."}}}}, {{'limit': {{'type': 'string', 'description': 'Maximum number of entries to show.'}}}}, {{'navigate': {{'type': 'string', 'description': "navigate relative to previous view. 'next' to search after previous view, 'prev' to search before the previous view. 'stay' or None to remain.'"}}}}, {{'pageInfo': {{'type': 'string', 'description': 'The previous pageInfo object, such as "{{'endCursor': 'eyJsYXN0X2lkIjo3Mjk2NTgxODk0MjU3LCJsYXN0X3ZhbHVlIjoiNzI5NjU4MTg5NDI1NyJ9', 'hasNextPage': True, 'hasPreviousPage': False, 'startCursor': 'eyJsYXN0X2lkIjo3Mjk2NTgwODQ1NjgxLCJsYXN0X3ZhbHVlIjoiNzI5NjU4MDg0NTY4MSJ9'}}"'}}}}], 'required': ['user_id']}}, 'shopify-search_products-search_products': {{'name': 'shopify-search_products-search_products', 'type': 'tool', 'description': 'Search products by string query. If no products are found, the function will return an error message.', 'parameters': [{{'product_query': {{'type': 'string', 'description': "The string query to search products, such as 'Hats'. If query is empty string, it returns all products."}}}}, {{'limit': {{'type': 'string', 'description': 'Maximum number of entries to show.'}}}}, {{'navigate': {{'type': 'string', 'description': "navigate relative to previous view. 'next' to search after previous view, 'prev' to search before the previous view. 'stay' or None to remain.'"}}}}, {{'pageInfo': {{'type': 'string', 'description': 'The previous pageInfo object, such as "{{'endCursor': 'eyJsYXN0X2lkIjo3Mjk2NTgxODk0MjU3LCJsYXN0X3ZhbHVlIjoiNzI5NjU4MTg5NDI1NyJ9', 'hasNextPage': True, 'hasPreviousPage': False, 'startCursor': 'eyJsYXN0X2lkIjo3Mjk2NTgwODQ1NjgxLCJsYXN0X3ZhbHVlIjoiNzI5NjU4MDg0NTY4MSJ9'}}"'}}}}], 'required': []}}, ...}}

Your current task is:


Thought:
To retrieve the user's details, I first need to find the user ID using the provided email address.

Action:
{{"name": "shopify-find_user_id_by_email-find_user_id_by_email", "arguments": {{"user_email": "sample-email@arklex.ai"}}}}

---

Select the next action based on the current task, conversation record, and available tools.

# User message
{user_message}

# Available actions
{available_actions}

Your current task is:
{task}

Thought:
"""

PLANNER_SUMMARIZE_TRAJECTORY_PROMPT = """
# Instruction
Please summarize the planning steps required to satisfy the user's request.
Your response must be formatted as a bulleted list where each line begins with a hyphen ("-"). Do not include any extraneous text.

# User message
{user_message}

# Available actions
To help determine what steps are required to satisfy the request, refer to the following descriptions of available actions: {resource_descriptions}

Your current task is:
{task}

Answer:
"""

PLANNER_REACT_INSTRUCTION_FEW_SHOT_WITH_RAG = """
# Instruction
Please act as an agent that selects the next appropriate action in a sequence of actions in order to satisfy the user's request.

Your generation should have exactly the following format:
Thought:
<A single line of reasoning to process the context and inform the decision making. Do not include extra lines.>

Action:
{{"name": <The name of the action>, "arguments": <Any arguments that the action requires in json format>}}

If you must return a message to the user or ask them for more information, your response must do so using the action "{respond_action_name}", for example:
Action:
{{"name": "{respond_action_name}", "arguments": {{"content": <message to return to the user>}}}}

Never provide a response that is not in this format.

For example: 
---

Select the next action based on the current task, conversation record, and available actions.

# User message
Can you please retrieve my user details? My email is sample-email@arklex.ai.

# Available actions
{{'shopify-find_user_id_by_email-find_user_id_by_email': {{'name': 'shopify-find_user_id_by_email-find_user_id_by_email', 'type': 'tool', 'description': 'Find user id by email. If the user is not found, the function will return an error message.', 'parameters': [{{'user_email': {{'type': 'string', 'description': "The email of the user, such as 'something@example.com'."}}}}], 'required': ['user_email'], 'returns': {{'user_id': "The user id of the user. such as 'gid://shopify/Customer/13573257450893'."}}}}, 'shopify-get_user_details_admin-get_user_details_admin': {{'name': 'shopify-get_user_details_admin-get_user_details_admin', 'type': 'tool', 'description': 'Get the details of a user with Admin API.', 'parameters': [{{'user_id': {{'type': 'string', 'description': "The user id, such as 'gid://shopify/Customer/13573257450893'."}}}}, {{'limit': {{'type': 'string', 'description': 'Maximum number of entries to show.'}}}}, {{'navigate': {{'type': 'string', 'description': "navigate relative to previous view. 'next' to search after previous view, 'prev' to search before the previous view. 'stay' or None to remain.'"}}}}, {{'pageInfo': {{'type': 'string', 'description': 'The previous pageInfo object, such as "{{\'endCursor\': \'eyJsYXN0X2lkIjo3Mjk2NTgxODk0MjU3LCJsYXN0X3ZhbHVlIjoiNzI5NjU4MTg5NDI1NyJ9\', \'hasNextPage\': True, \'hasPreviousPage\': False, \'startCursor\': \'eyJsYXN0X2lkIjo3Mjk2NTgwODQ1NjgxLCJsYXN0X3ZhbHVlIjoiNzI5NjU4MDg0NTY4MSJ9\'}}"'}}}}], 'required': ['user_id'], 'returns': {{'user_details': 'The user details of the user. such as \'{{"firstName": "John", "lastName": "Doe", "email": "example@gmail.com"}}\'.', 'pageInfo': 'Current pageInfo object, such as  "{{\'endCursor\': \'eyJsYXN0X2lkIjo3Mjk2NTgxODk0MjU3LCJsYXN0X3ZhbHVlIjoiNzI5NjU4MTg5NDI1NyJ9\', \'hasNextPage\': True, \'hasPreviousPage\': False, \'startCursor\': \'eyJsYXN0X2lkIjo3Mjk2NTgwODQ1NjgxLCJsYXN0X3ZhbHVlIjoiNzI5NjU4MDg0NTY4MSJ9\'}}"'}}}}, 'FaissRAGWorker': {{'name': 'FaissRAGWorker', 'type': 'worker', 'description': "Answer the user's questions based on the company's internal documentations (unstructured text data), such as the policies, FAQs, and product information", 'parameters': [], 'required': [], 'returns': {{}}}}, 'shopify-get_cart-get_cart': {{'name': 'shopify-get_cart-get_cart', 'type': 'tool', 'description': 'Get cart information', 'parameters': [{{'cart_id': {{'type': 'str', 'description': "The cart id, such as 'gid://shopify/Cart/2938501948327'."}}}}, {{'limit': {{'type': 'string', 'description': 'Maximum number of entries to show.'}}}}, {{'navigate': {{'type': 'string', 'description': "navigate relative to previous view. 'next' to search after previous view, 'prev' to search before the previous view. 'stay' or None to remain.'"}}}}, {{'pageInfo': {{'type': 'string', 'description': 'The previous pageInfo object, such as "{{\'endCursor\': \'eyJsYXN0X2lkIjo3Mjk2NTgxODk0MjU3LCJsYXN0X3ZhbHVlIjoiNzI5NjU4MTg5NDI1NyJ9\', \'hasNextPage\': True, \'hasPreviousPage\': False, \'startCursor\': \'eyJsYXN0X2lkIjo3Mjk2NTgwODQ1NjgxLCJsYXN0X3ZhbHVlIjoiNzI5NjU4MDg0NTY4MSJ9\'}}"'}}}}], 'required': ['cart_id'], 'returns': {{'get_cart': 'The cart details of the cart.', 'pageInfo': 'Current pageInfo object, such as  "{{\'endCursor\': \'eyJsYXN0X2lkIjo3Mjk2NTgxODk0MjU3LCJsYXN0X3ZhbHVlIjoiNzI5NjU4MTg5NDI1NyJ9\', \'hasNextPage\': True, \'hasPreviousPage\': False, \'startCursor\': \'eyJsYXN0X2lkIjo3Mjk2NTgwODQ1NjgxLCJsYXN0X3ZhbHVlIjoiNzI5NjU4MDg0NTY4MSJ9\'}}"'}}}}, 'respond': {{'name': 'respond', 'type': 'worker', 'description': "Respond to the user if the user's request has been satisfied or if there is not enough information to do so.", 'parameters': [{{'content': {{'type': 'string', 'description': 'The message to return to the user.'}}}}], 'required': ['content'], 'returns': {{}}}}}}

Your current task is:


Thought:
To retrieve the user's details, I first need to find the user ID using the provided email address.

Action:
{{"name": "shopify-find_user_id_by_email-find_user_id_by_email", "arguments": {{"user_email": "sample-email@arklex.ai"}}}}

---

Select the next action based on the current task, conversation record, and available tools.

# User message
{user_message}

# Available actions
{available_actions}

Your current task is:
{task}

Thought:
"""
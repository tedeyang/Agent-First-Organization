from typing import List, Dict, Any
from arklex.memory.core import ShortTermMemory
from arklex.utils.graph_state import ResourceRecord, LLMConfig, MessageState, BotConfig
from arklex.utils.model_config import MODEL
import asyncio

# class ResourceRecord(BaseModel):
#     info: Dict
#     intent: str = Field(default="")
#     input: List = Field(default_factory=list)
#     output: str = Field(default="")
#     steps: List = Field(default_factory=list)
#     personalized_intent: str = Field(default="")


# Case 1: General product list → Agent lists multiple categories → User then specifies one
record_case1 = ResourceRecord(
    info={"attribute": {"task": "List available products", "direct": False}},
    intent="User is browsing product categories",
    input=[],
    output=(
        "• Pink Unicorn Boys & Girls Baseball Hat with Adjustable Buckle (id=gid://shopify/Product/7949281198127) – 3D unicorn design, ages 2‑12, adjustable buckle. Inventory: 547\n"
        "• Denim Apron with 5 Pockets | Multipurpose Canvas Apron (id=gid://shopify/Product/7949281624111) – 100% cotton canvas, 5 front pockets, one‑size fits all. Inventory: 349\n"
        "• Winter Flannel Blanket Solid Color Plaid Coral Blanket Fleece Bedspread (id=gid://shopify/Product/7949279526959) – Microfiber flannel, heated, anti‑pilling. Inventory: 0"
    ),
    steps=[
        {
            "context_generate": (
                "We have several products in our store. Are you looking for something particular? Let us know so we can find you the best match!"
            )
        }
    ],
)

# Case 1 Follow‑up: User specifies "Denim Apron with 5 Pockets"
record_case1_followup = ResourceRecord(
    info={"attribute": {"task": "Fetch product details", "direct": True}},
    intent="User wants product details",
    input=[],
    output=(
        "• Denim Apron with 5 Pockets | Multipurpose Canvas Apron (id=gid://shopify/Product/7949281624111) – 100% cotton canvas, 5 front pockets, one‑size fits all. Price: $19.99, Inventory: 349"
    ),
    steps=[
        {
            "context_generate": (
                "We have a versatile apron that could match your needs for various activities like BBQ, gardening, or painting. Are you interested in a particular color, like navy blue or black? Would you prefer an apron with specific features or price range?"
            )
        }
    ],
)


# Case 2: User asks about aprons → Agent lists aprons → User asks "Does it have pockets?"
record_case2_initial = ResourceRecord(
    info={"attribute": {"task": "List available products", "direct": False}},
    intent="User wants information about products",
    input=[],
    output=(
        "• Denim Apron with 5 Pockets | Multipurpose Canvas Apron (id=gid://shopify/Product/7949281624111) – 100% cotton canvas, 5 front pockets. Inventory: 349\n"
        "• Denim Apron with 3 Pockets | Multipurpose Canvas Apron (id=gid://shopify/Product/7949281722415) – 100% cotton canvas, 3 front pockets. Inventory: 84"
    ),
    steps=[
        {
            "context_generate": (
                "We found some aprons that might interest you. Are you looking for a specific color or need a certain number of pockets for your activities? Let us know how you'll be using the apron to better assist you in finding the perfect match!"
            )
        }
    ],
)

record_case2_followup = ResourceRecord(
    info={"attribute": {"task": "Fetch product details", "direct": True}},
    intent="User has inquiries related to product",
    input=[],
    output=(
        "• Denim Apron with 5 Pockets | Multipurpose Canvas Apron (id=gid://shopify/Product/7949281624111) – 100% cotton canvas, 5 front pockets. Inventory: 349\n"
        "• Denim Apron with 3 Pockets | Multipurpose Canvas Apron (id=gid://shopify/Product/7949281722415) – 100% cotton canvas, 3 front pockets. Inventory: 84"
    ),
    steps=[
        {
            "context_generate": "Yes—we have aprons with pockets for tools and accessories."
        }
    ],
)


# Case 3: Recommend aprons → then user switches to hats
record_case3_aprons = ResourceRecord(
    info={"attribute": {"task": "Recommend products", "direct": False}},
    intent="User wants recommendation",
    input=[],
    output=(
        "• Denim Apron with 5 Pockets | Multipurpose Canvas Apron (id=gid://shopify/Product/7949281624111) – 100% cotton canvas, 5 front pockets. Inventory: 349\n"
        "• Denim Apron with 3 Pockets | Multipurpose Canvas Apron (id=gid://shopify/Product/7949281722415) – 100% cotton canvas, 3 front pockets. Inventory: 84"
    ),
    steps=[
        {
            "context_generate": (
                "We have some aprons that might interest you! Do you have a color preference between navy blue, black, or blue? Also, are you looking for an apron with more pockets for functionality or fewer pockets for simplicity?"
            )
        }
    ],
)

record_case3_hats = ResourceRecord(
    info={"attribute": {"task": "Recommend products", "direct": False}},
    intent="User wanst recommendation",
    input=[],
    output=(
        "• Pink Unicorn Boys & Girls Baseball Hat with Adjustable Buckle (id=gid://shopify/Product/7949281198127) – unicorn design, ages 2‑12. Inventory: 547\n"
        "• Metallic Navy Blue Boys & Girls Baseball Hat with Adjustable Buckle (id=gid://shopify/Product/7949281427503) – breathable cotton. Sizes 2‑5 (27), 6‑12 (0)."
    ),
    steps=[
        {
            "context_generate": (
                "We have some colorful and fun hats available. Are you looking for a particular color or style? Would you prefer an adjustable buckle for a better fit?"
            )
        }
    ],
)


# Case 4: Recommend hats → then navy blue hats follow‑up
record_case4_hats = ResourceRecord(
    info={"attribute": {"task": "Recommend products", "direct": False}},
    intent="User wants recommendation",
    input=[],
    output=(
        "• Pink Unicorn Boys & Girls Baseball Hat with Adjustable Buckle (id=gid://shopify/Product/7949281198127) – 3D unicorn design. Inventory: 547\n"
        "• Green Boys & Girls Baseball Hat with Adjustable Buckle (id=gid://shopify/Product/7949281296431) – reflective strip. Inventory: 55\n"
        "• Metallic Navy Blue Boys & Girls Baseball Hat with Adjustable Buckle (id=gid://shopify/Product/7949281427503) – reflective strip, breathable cotton. Sizes 2‑5 (27), 6‑12 (0)."
    ),
    steps=[
        {
            "context_generate": (
                "We have a variety of baseball hats: unicorn, green no‑logo, and metallic navy blue. Interested in a specific color?"
            )
        }
    ],
)

record_case4_navy = ResourceRecord(
    info={"attribute": {"task": "Recommend products", "direct": False}},
    intent="User wants recommendation",
    input=[],
    output=(
        "• Metallic Navy Blue Boys & Girls Baseball Hat with Adjustable Buckle (id=gid://shopify/Product/7949281427503) – reflective strip, breathable cotton. Sizes 2‑5 (27), 6‑12 (0)."
    ),
    steps=[
        {
            "context_generate": (
                "Sure—here's the navy blue baseball hat we have in stock."
            )
        }
    ],
)


# Collect all records in a list for testing
sample_records = [
    record_case1,
    record_case1_followup,
    record_case2_initial,
    record_case2_followup,
    record_case3_aprons,
    record_case3_hats,
    record_case4_hats,
    record_case4_navy,
]


# Shopify-style grouped records function
def get_shopify_records() -> List[List[ResourceRecord]]:
    """
    Returns sample ResourceRecord groups simulating Shopify ecommerce assistant turns.
    """
    return [
        record_case1,
        record_case1_followup,
        record_case2_initial,
        record_case2_followup,
        record_case3_aprons,
        record_case3_hats,
        record_case4_hats,
        record_case4_navy,
    ]


# Test configuration
TEST_CONFIG: Dict[str, Any] = {
    "model": MODEL,
    "role": "test_assistant",
    "user_objective": "Test the short term memory functionality",
    "builder_objective": "Ensure proper memory retrieval and intent matching",
    "intro": "This is a test bot for short term memory",
    "bot_id": "test_bot",
    "version": "1.0",
    "language": "EN",
    "bot_type": "test",
}


def init_test_state() -> MessageState:
    # Initialize LLMConfig from test config
    llm_config: LLMConfig = LLMConfig(**TEST_CONFIG.get("model", MODEL))

    # Create BotConfig
    bot_config: BotConfig = BotConfig(
        bot_id=TEST_CONFIG.get("bot_id", "test_bot"),
        version=TEST_CONFIG.get("version", "1.0"),
        language=TEST_CONFIG.get("language", "EN"),
        bot_type=TEST_CONFIG.get("bot_type", "test"),
        llm_config=llm_config,
    )

    # Create MessageState
    sys_instruct: str = (
        "You are a "
        + TEST_CONFIG["role"]
        + ". "
        + TEST_CONFIG["user_objective"]
        + TEST_CONFIG["builder_objective"]
        + TEST_CONFIG["intro"]
    )

    return MessageState(sys_instruct=sys_instruct, bot_config=bot_config)


def test_shopify_intent() -> None:
    # Initialize test state
    state = init_test_state()

    # Test Case 1: Product Selection Flow
    print("\n=== Case 1: Product Selection Flow ===")
    # Create trajectory with just the first two records
    trajectory_case1 = [[record_case1]]
    chat_history_case1 = """assistant: Hello! How can I help you today?
user: What products do you have?
assistant: We have several products in our store. Are you looking for something particular?"""

    # Create STM instance for case 1
    llm_config = LLMConfig(**TEST_CONFIG.get("model", MODEL))
    bot_config = BotConfig(
        bot_id="test_bot",
        version="1.0",
        language="EN",
        bot_type="test",
        llm_config=llm_config,
    )
    stm_case1 = ShortTermMemory(trajectory_case1, chat_history_case1, llm_config=bot_config.llm_config)
    asyncio.run(stm_case1.personalize())

    # Test case 1
    found_case1, records_case1 = stm_case1.retrieve_records("Show me the denim apron with 5 pockets")
    found_intent_case1, intent_case1 = stm_case1.retrieve_intent("I want to see the denim apron with 5 pockets")
    print(f"Expected Behavior: found_intent=True, allow_global_intent_switch=False")
    print(f"Actual Results:")
    print(f"- found_intent: {found_intent_case1} (Expected: True)")
    print(f"- allow_global_intent_switch: {not found_intent_case1} (Expected: False)")
    if found_case1:
        print("\nFound Records (first 100 tokens):")
        for record in records_case1:
            print(f"\nRecord Intent: {record.intent}")
            print(f"Record Personalized Intent: {record.personalized_intent}")
            print(f"Record Output (first 100 tokens): {record.output[:100]}")

    # Test Case 2: Follow-up Questions
    print("\n=== Case 2: Follow-up Questions ===")
    # Create trajectory with just the relevant records
    trajectory_case2 = [[record_case2_initial]]
    chat_history_case2 = """assistant: Hello! How can I help you today?
user: Do you have any aprons?
assistant: Yes, we have several aprons available."""

    # Create STM instance for case 2
    stm_case2 = ShortTermMemory(trajectory_case2, chat_history_case2, llm_config=bot_config.llm_config)
    asyncio.run(stm_case2.personalize())

    # Test case 2
    found_case2, records_case2 = stm_case2.retrieve_records("Does it have pockets?")
    found_intent_case2, intent_case2 = stm_case2.retrieve_intent("Does it have pockets?")
    print(f"Expected Behavior: found_intent=True, allow_global_intent_switch=False")
    print(f"Actual Results:")
    print(f"- found_intent: {found_intent_case2} (Expected: True)")
    print(f"- allow_global_intent_switch: {not found_intent_case2} (Expected: False)")
    if found_case2:
        print("\nFound Records (first 100 tokens):")
        for record in records_case2:
            print(f"\nRecord Intent: {record.intent}")
            print(f"Record Personalized Intent: {record.personalized_intent}")
            print(f"Record Output (first 100 tokens): {record.output[:100]}")

    # Test Case 3: Different Product Recommendations
    print("\n=== Case 3: Different Product Recommendations ===")
    # Create trajectory with just the relevant records
    trajectory_case3 = [[record_case3_aprons]]
    chat_history_case3 = """assistant: Hello! How can I help you today?
user: I need an apron.
assistant: I can help you find the perfect apron."""

    # Create STM instance for case 3
    stm_case3 = ShortTermMemory(trajectory_case3, chat_history_case3, llm_config=bot_config.llm_config)
    asyncio.run(stm_case3.personalize())

    # Test case 3
    found_case3, records_case3 = stm_case3.retrieve_records("I want bedframes")
    found_intent_case3, intent_case3 = stm_case3.retrieve_intent("I want bedframes")
    print(f"Expected Behavior: found_intent=False, allow_global_intent_switch=True")
    print(f"Actual Results:")
    print(f"- found_intent: {found_intent_case3} (Expected: False)")
    print(f"- intent: {intent_case3}")
    print(f"- allow_global_intent_switch: {not found_intent_case3} (Expected: True)")
    if found_case3:
        print("\nFound Records (first 100 tokens):")
        for record in records_case3:
            print(f"\nRecord Intent: {record.intent}")
            print(f"Record Personalized Intent: {record.personalized_intent}")
            print(f"Record Output (first 100 tokens): {record.output[:100]}")

    # Test Case 4: Same Product, Different Attributes
    print("\n=== Case 4: Same Product, Different Attributes ===")
    # Create trajectory with just the relevant records
    trajectory_case4 = [[record_case4_hats]]
    chat_history_case4 = """assistant: Hello! How can I help you today?
user: Show me hats.
assistant: Here are our hat collections."""

    # Create STM instance for case 4
    stm_case4 = ShortTermMemory(trajectory_case4, chat_history_case4, llm_config=bot_config.llm_config)
    asyncio.run(stm_case4.personalize())

    # Test case 4
    found_case4, records_case4 = stm_case4.retrieve_records("Do you have navy blue hats?")
    found_intent_case4, intent_case4 = stm_case4.retrieve_intent("Show me navy blue hats")
    print(f"Expected Behavior: found_intent=True, allow_global_intent_switch=False")
    print(f"Actual Results:")
    print(f"- found_intent: {found_intent_case4} (Expected: True)")
    print(f"- allow_global_intent_switch: {not found_intent_case4} (Expected: False)")
    if found_case4:
        print("\nFound Records (first 100 tokens):")
        for record in records_case4:
            print(f"\nRecord Intent: {record.intent}")
            print(f"Record Personalized Intent: {record.personalized_intent}")
            print(f"Record Output (first 100 tokens): {record.output[:100]}")

    # Summary of Test Results
    print("\n=== Test Summary ===")
    print("Case 1 (Product Selection):", "PASS" if found_intent_case1 else "FAIL")
    print("Case 2 (Follow-up):", "PASS" if found_intent_case2 else "FAIL")
    print("Case 3 (Different Product):", "PASS" if not found_intent_case3 else "FAIL")
    print("Case 4 (Same Product Attributes):", "PASS" if found_intent_case4 else "FAIL")


if __name__ == "__main__":
    test_shopify_intent()

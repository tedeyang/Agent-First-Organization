"""Entities for slot management.

This module defines the entities used in the slot management system,
including Slot for tracking slot states and relationships.

Key Components:
- Slot: Represents a slot in the conversation system.
- SlotInput: Represents the input format for slot filling operations.
- SlotInputList: Represents a list of slot inputs.
- Verification: Represents the result of verifying a slot value.
"""

from typing import Any

from pydantic import BaseModel, Field


class Slot(BaseModel):
    """Represents a slot in the conversation system.

    A slot is a named container for a value of a specific type, with optional
    validation rules and metadata. Slots are used to capture and validate user
    input during conversations.

    The class provides:
    1. Type-safe value storage
    2. Value validation through enums
    3. Metadata for slot description and prompting
    4. Verification status tracking

    Attributes:
        name (str): The name of the slot.
        type (str): The type of the slot value (default: "str").
        value (Any): The current value of the slot (can be primitive, list, dict, or list of dicts).
        enum (Optional[List[Union[str, int, float, bool, None]]]): List of valid values.
        description (str): Description of the slot's purpose.
        prompt (str): Prompt to use when filling the slot.
        required (bool): Whether the slot must be filled.
        verified (bool): Whether the slot's value has been verified.
    """

    name: str
    type: str = Field(default="str")
    value: Any = Field(default=None)
    enum: list[str | int | float | bool | None] | None = Field(default=[])
    description: str = Field(default="")
    prompt: str = Field(default="")
    required: bool = Field(default=False)
    verified: bool = Field(default=False)
    repeatable: bool = Field(default=False)
    schema: list[dict] | None = None
    items: dict | None = None
    target: str | None = None
    valueSource: str | None = Field(default=None)

    def to_openai_schema(self) -> dict | None:
        
        def _get_type_map() -> dict[str, str]:
            """Get the mapping from internal types to OpenAI schema types."""
            return {
                "str": "string",
                "int": "integer",
                "float": "number",
                "bool": "boolean",
            }

        def _build_group_schema() -> dict:
            """Build schema for group type fields."""
            properties = {}
            required = []
            for field in self.schema or []:
                field_slot = field if isinstance(field, Slot) else Slot(**field)
                if getattr(field_slot, "valueSource", None) == "fixed":
                    continue
                properties[field_slot.name] = field_slot.to_openai_schema()
                if getattr(field_slot, "required", False):
                    required.append(field_slot.name)
            
            return {
                "type": "object",
                "properties": properties,
                "required": required,
                "description": getattr(self, "description", ""),
            }

        def _build_primitive_schema() -> dict:
            """Build schema for primitive type fields."""
            type_map = _get_type_map()
            return {
                "type": type_map.get(self.type, "string"),
                "description": getattr(self, "description", ""),
            }
        
        if getattr(self, "valueSource", None) == "fixed":
            return None
        
        # Handle repeatable fields - they should be arrays
        if getattr(self, "repeatable", False):
            if self.type == "group":
                # For repeatable group, each item is an object with the group's schema
                return {
                    "type": "array",
                    "items": _build_group_schema(),
                    "description": getattr(self, "description", ""),
                }
            else:
                # For repeatable primitive types, define the item type
                return {
                    "type": "array",
                    "items": _build_primitive_schema(),
                    "description": getattr(self, "description", ""),
                }
        elif self.type == "group":
            return _build_group_schema()
        else:
            # Primitive type
            return _build_primitive_schema()


class SlotInput(BaseModel):
    """Input structure for slot filling operations.

    This class represents the input format for slot filling operations,
    containing the essential information needed to process a slot.

    The class provides:
    1. Structured input format for slot filling
    2. Type-safe value handling
    3. Support for enum-based validation
    4. Descriptive metadata

    Attributes:
        name (str): The name of the slot.
        value (Union[str, int, float, bool, List[str], None]): The current value.
        enum (Optional[List[Union[str, int, float, bool, None]]]): Valid values.
        description (str): Description of the slot's purpose.
    """

    name: str
    value: str | int | float | bool | list[str] | None
    enum: list[str | int | float | bool | None] | None
    description: str


class SlotInputList(BaseModel):
    """Container for a list of slot inputs.

    This class serves as a container for multiple slot inputs that need to be
    processed together in a slot filling operation.

    The class provides:
    1. Batch processing of multiple slots
    2. Structured input format for slot filling operations
    3. Type-safe handling of multiple slot inputs

    Attributes:
        slot_input_list (List[SlotInput]): List of slot inputs to process.
    """

    slot_input_list: list[SlotInput]


class Verification(BaseModel):
    """Verification result for a slot value.

    This class represents the result of verifying a slot value, including
    the reasoning behind the verification decision and whether additional
    verification is needed.

    The class provides:
    1. Structured representation of verification results
    2. Reasoning for verification decisions
    3. Status tracking for additional verification needs

    Attributes:
        thought (str): Reasoning behind the verification decision.
        verification_needed (bool): Whether additional verification is required.
    """

    thought: str
    verification_needed: bool

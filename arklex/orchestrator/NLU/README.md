# Arklex Natural Language Understanding (NLU)

This directory contains the Natural Language Understanding (NLU) components of the Arklex framework. The NLU system is responsible for intent detection, slot filling, and understanding user input in conversations.

## Components

### Core Components

- `core/intent.py`: Intent detection and classification
- `core/slot.py`: Slot filling and value extraction
- `core/base.py`: Base interfaces and abstract classes

### Services

- `services/model_service.py`: Model interaction and configuration
- `services/api_service.py`: API client and request handling

### API Layer

- `api/routes.py`: FastAPI endpoints for NLU services
- `api/schemas.py`: Request/response models and validation

## Features

### Intent Detection

- Multi-intent classification
- Confidence scoring
- Context-aware processing
- Fallback handling
- Custom intent support

### Slot Filling

- Entity extraction
- Value validation
- Type conversion
- Required slot handling
- Slot verification

### API Integration

- RESTful endpoints
- Streaming support
- Authentication
- Rate limiting
- Error responses

## Usage

### Intent Detection

```python
from arklex.orchestrator.NLU.core.intent import IntentDetector

# Initialize intent detector
intent_detector = IntentDetector(config={
    "api_url": "your_api_url",
    "api_key": "your_api_key"
})

# Detect intent
intent = intent_detector.execute(
    "I want to book a flight",
    candidate_intents={
        "book_flight": [...],
        "check_status": [...]
    }
)
```

### Slot Filling

```python
from arklex.orchestrator.NLU.core.slot import SlotFiller
from arklex.utils.slot import Slot

# Initialize the slot filler
slot_filler = SlotFiller(config={
    "api_url": "your_api_url",
    "api_key": "your_api_key"
})

# Define slots
slots = [
    Slot(
        name="name",
        description="The user's full name",
        required=True,
        type="string"
    ),
    Slot(
        name="age",
        description="The user's age",
        required=True,
        type="integer"
    )
]

# Fill slots from text
filled_slots = slot_filler.execute(slots, "My name is John and I am 30 years old")
```

### API Endpoints

```python
from fastapi import FastAPI
from arklex.orchestrator.NLU.api.routes import create_app

app = create_app()

# The app will have these endpoints:
# - POST /intent/predict: Intent detection
# - POST /slot/predict: Slot filling
# - POST /slot/verify: Slot verification
```

## Configuration

### Intent Detection Configuration

```json
{
    "api_url": "your_api_url",
    "api_key": "your_api_key",
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 100,
    "timeout": 30
}
```

### Slot Filling Configuration

```json
{
    "api_url": "your_api_url",
    "api_key": "your_api_key",
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 100,
    "timeout": 30
}
```

## Best Practices

1. **Intent Detection**
   - Use clear and specific intent names
   - Include fallback intents
   - Consider context in classification
   - Validate intent confidence

2. **Slot Filling**
   - Define clear slot types
   - Validate slot values
   - Handle missing slots
   - Provide clear prompts

3. **API Usage**
   - Implement proper error handling
   - Use appropriate timeouts
   - Validate input/output
   - Monitor API usage

4. **Performance**
   - Cache common intents
   - Batch process when possible
   - Optimize model parameters
   - Monitor response times

## Error Handling

The NLU system handles various error cases:

- Invalid input format
- Missing required slots
- Invalid slot values
- API timeouts
- Model errors
- Configuration issues

## Integration

The NLU system integrates with:

- Task Graph
- Orchestrator
- Message Processing
- State Management
- Resource Management

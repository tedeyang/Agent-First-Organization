import pytest
from arklex.orchestrator.generator.docs.document_validator import DocumentValidator


@pytest.fixture
def validator():
    return DocumentValidator()


@pytest.fixture
def valid_document():
    return {
        "title": "Test Document",
        "sections": [
            {"name": "Section 1", "content": "Some content", "requirements": ["req1"]},
            {"name": "Section 2", "content": "Other content", "requirements": ["req2"]},
        ],
    }


def test_validate_structure_valid(validator, valid_document) -> None:
    assert validator.validate_structure(valid_document) is True


def test_validate_structure_missing_field(validator, valid_document) -> None:
    doc = valid_document.copy()
    del doc["title"]
    assert validator.validate_structure(doc) is False


def test_validate_structure_invalid_sections_type(validator, valid_document) -> None:
    doc = valid_document.copy()
    doc["sections"] = "not a list"
    assert validator.validate_structure(doc) is False


def test_validate_structure_invalid_section(validator, valid_document) -> None:
    doc = valid_document.copy()
    doc["sections"][0] = {"name": "Section 1"}  # missing content
    assert validator.validate_structure(doc) is False


def test_validate_required_fields_valid(validator, valid_document) -> None:
    assert validator.validate_required_fields(valid_document) is True


def test_validate_required_fields_missing_doc_field(validator, valid_document) -> None:
    doc = valid_document.copy()
    del doc["title"]
    assert validator.validate_required_fields(doc) is False


def test_validate_required_fields_missing_section_field(
    validator, valid_document
) -> None:
    doc = valid_document.copy()
    doc["sections"][0] = {"content": "no name"}
    assert validator.validate_required_fields(doc) is False


def test_validate_consistency_valid(validator, valid_document) -> None:
    assert validator.validate_consistency(valid_document) is True


def test_validate_consistency_duplicate_section_names(
    validator, valid_document
) -> None:
    doc = valid_document.copy()
    doc["sections"][1]["name"] = "Section 1"
    assert validator.validate_consistency(doc) is False


def test_validate_consistency_empty_requirements(validator, valid_document) -> None:
    doc = valid_document.copy()
    doc["sections"][0]["requirements"] = []
    assert validator.validate_consistency(doc) is False


def test_get_error_messages_all_valid(validator, valid_document) -> None:
    assert validator.get_error_messages(valid_document) == []


def test_get_error_messages_invalid_structure(validator, valid_document) -> None:
    doc = valid_document.copy()
    del doc["title"]
    errors = validator.get_error_messages(doc)
    assert "Invalid document structure" in errors


def test_get_error_messages_missing_required_fields(validator, valid_document) -> None:
    doc = valid_document.copy()
    doc["sections"][0] = {"content": "no name"}
    errors = validator.get_error_messages(doc)
    assert "Missing required fields" in errors


def test_get_error_messages_consistency(validator, valid_document) -> None:
    doc = valid_document.copy()
    doc["sections"][1]["name"] = "Section 1"
    errors = validator.get_error_messages(doc)
    assert "Document consistency issues" in errors


def test__validate_section_valid(validator) -> None:
    section = {"name": "Section", "content": "Content", "requirements": ["req"]}
    assert validator._validate_section(section) is True


def test__validate_section_missing_field(validator) -> None:
    section = {"content": "Content"}
    assert validator._validate_section(section) is False


def test__validate_section_empty_content(validator) -> None:
    section = {"name": "Section", "content": "", "requirements": ["req"]}
    assert validator._validate_section(section) is False


def test__validate_section_invalid_requirements(validator) -> None:
    section = {"name": "Section", "content": "Content", "requirements": "notalist"}
    assert validator._validate_section(section) is False

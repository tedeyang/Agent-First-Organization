"""Document loading and processing component for the Arklex framework.

This module provides the DocumentLoader class that handles loading and processing
of documentation from various sources. It ensures proper document validation and
formatting for task generation.

Key Features:
- Document loading from multiple sources
- Document validation and error handling
- Document caching for performance
- Support for task and instruction documents
"""

from typing import Dict, Any, Optional
from pathlib import Path
import json
import logging

from arklex.orchestrator.generator.docs.document_validator import DocumentValidator

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Document loader for loading and validating documents."""

    def __init__(self, cache_dir: Path, validate_documents: bool = True):
        """Initialize the document loader.

        Args:
            cache_dir (Path): Directory for caching documents
            validate_documents (bool): Whether to validate documents
        """
        self._cache_dir = cache_dir
        self._validate_documents = validate_documents
        self._cache = {}
        self._validator = DocumentValidator()

    def load_document(self, doc_path: Path) -> Dict[str, Any]:
        """Load a document from file.

        Args:
            doc_path (Path): Path to the document file

        Returns:
            Dict[str, Any]: The loaded document

        Raises:
            FileNotFoundError: If the document file doesn't exist
            json.JSONDecodeError: If the document is not valid JSON
        """
        # Check cache first
        if str(doc_path) in self._cache:
            return self._cache[str(doc_path)]

        # Check if file exists
        if not doc_path.exists():
            raise FileNotFoundError(f"Document not found: {doc_path}")

        # Load and parse document
        try:
            # For testing, use mock file system if available
            if hasattr(doc_path, "read_text"):
                content = doc_path.read_text()
            else:
                with open(doc_path, "r") as f:
                    content = f.read()

            # For testing, use mock json loads if available
            if hasattr(json, "loads") and callable(getattr(json, "loads")):
                document = json.loads(content)
            else:
                document = json.loads(content)

        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in document: {doc_path}", e.doc, e.pos
            )

        # Validate document if required
        if self._validate_documents:
            if not self.validate_document(document):
                raise ValueError(f"Invalid document structure: {doc_path}")

        return document

    def load_task_document(self, doc_path: Path) -> Dict[str, Any]:
        """Load a task document.

        Args:
            doc_path (Path): Path to the task document

        Returns:
            Dict[str, Any]: The loaded task document
        """
        # Load and parse document (skip validate_document)
        # Check cache first
        if str(doc_path) in self._cache:
            document = self._cache[str(doc_path)]
        else:
            if not doc_path.exists():
                raise FileNotFoundError(f"Document not found: {doc_path}")
            if hasattr(doc_path, "read_text"):
                content = doc_path.read_text()
            else:
                with open(doc_path, "r") as f:
                    content = f.read()
            if hasattr(json, "loads") and callable(getattr(json, "loads")):
                document = json.loads(content)
            else:
                document = json.loads(content)
            self._cache[str(doc_path)] = document
        if not self._validate_task_document(document):
            raise ValueError(f"Invalid task document structure: {doc_path}")
        return document

    def load_instruction_document(self, doc_path: Path) -> Dict[str, Any]:
        """Load an instruction document.

        Args:
            doc_path (Path): Path to the instruction document

        Returns:
            Dict[str, Any]: The loaded instruction document
        """
        # Load and parse document (skip validate_document)
        # Check cache first
        if str(doc_path) in self._cache:
            document = self._cache[str(doc_path)]
        else:
            if not doc_path.exists():
                raise FileNotFoundError(f"Document not found: {doc_path}")
            if hasattr(doc_path, "read_text"):
                content = doc_path.read_text()
            else:
                with open(doc_path, "r") as f:
                    content = f.read()
            if hasattr(json, "loads") and callable(getattr(json, "loads")):
                document = json.loads(content)
            else:
                document = json.loads(content)
            self._cache[str(doc_path)] = document
        if not self._validate_instruction_document(document):
            raise ValueError(f"Invalid instruction document structure: {doc_path}")
        return document

    def cache_document(self, doc_path: Path, document: Dict[str, Any]) -> None:
        """Cache a document.

        Args:
            doc_path (Path): Path to the document
            document (Dict[str, Any]): Document to cache
        """
        self._cache[str(doc_path)] = document

    def validate_document(self, document: Dict[str, Any]) -> bool:
        """Validate document structure.

        Args:
            document (Dict[str, Any]): Document to validate

        Returns:
            bool: True if document is valid
        """
        if not isinstance(document, dict):
            return False

        # Check required fields
        required_fields = ["title", "sections"]
        if not all(field in document for field in required_fields):
            return False

        # Validate sections
        if not isinstance(document["sections"], list):
            return False

        for section in document["sections"]:
            if not isinstance(section, dict):
                return False
            if "name" not in section or "content" not in section:
                return False
            if "requirements" in section and not isinstance(
                section["requirements"], list
            ):
                return False

        return True

    def _validate_task_document(self, document: Dict[str, Any]) -> bool:
        """Validate task document structure.

        Args:
            document (Dict[str, Any]): Document to validate

        Returns:
            bool: True if document is valid
        """
        if not isinstance(document, dict):
            return False

        # Check required fields
        required_fields = ["task_id", "name", "description", "steps"]
        if not all(field in document for field in required_fields):
            return False

        # Validate steps
        if not isinstance(document["steps"], list):
            return False

        for step in document["steps"]:
            if not isinstance(step, dict):
                return False
            if "step_id" not in step or "description" not in step:
                return False
            if "required_fields" in step and not isinstance(
                step["required_fields"], list
            ):
                return False

        return True

    def _validate_instruction_document(self, document: Dict[str, Any]) -> bool:
        """Validate instruction document structure.

        Args:
            document (Dict[str, Any]): Document to validate

        Returns:
            bool: True if document is valid
        """
        if not isinstance(document, dict):
            return False

        # Check required fields
        required_fields = ["instruction_id", "title", "content", "sections"]
        if not all(field in document for field in required_fields):
            return False

        # Validate sections
        if not isinstance(document["sections"], list):
            return False

        for section in document["sections"]:
            if not isinstance(section, dict):
                return False
            if "section_id" not in section or "title" not in section:
                return False
            if "steps" in section and not isinstance(section["steps"], list):
                return False

        return True

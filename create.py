"""Task graph generation and worker initialization for the Arklex framework.

This module provides functionality for generating task graphs and initializing
workers in the Arklex framework. It includes utilities for creating task graphs
based on configuration files, setting up workers like FaissRAGWorker and
DataBaseWorker, and managing the overall system initialization process.
"""

import argparse
import json
import logging
import os
import tempfile
import zipfile
from typing import Any, Dict, List, Optional, Set

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from arklex.utils.utils import init_logger
from arklex.orchestrator.generator.generator import Generator
from arklex.env.tools.RAG.build_rag import build_rag
from arklex.env.tools.database.build_database import build_database
from arklex.utils.model_config import MODEL
from arklex.utils.model_provider_config import LLM_PROVIDERS, PROVIDER_MAP
from arklex.utils.loader import Loader

logger = init_logger(
    log_level=logging.INFO,
    filename=os.path.join(os.path.dirname(__file__), "logs", "arklex.log"),
)
load_dotenv()


def generate_taskgraph(args: argparse.Namespace) -> None:
    """Generate a task graph based on the provided configuration.

    This function initializes a language model, loads the configuration, and uses the Generator
    to create a task graph. It then saves the task graph to a file and updates it with API URLs.

    Args:
        args (argparse.Namespace): Command-line arguments containing configuration and output settings.
    """
    model = PROVIDER_MAP.get(MODEL["llm_provider"], ChatOpenAI)(
        model=MODEL["model_type_or_path"], timeout=30000
    )
    config: Dict[str, Any] = json.load(open(args.config))
    generator = Generator(config, model, args.output_dir)
    taskgraph = generator.generate()
    taskgraph_filepath: str = generator.save_task_graph(taskgraph)
    # Update the task graph with the API URLs
    task_graph: Dict[str, Any] = json.load(
        open(os.path.join(os.path.dirname(__file__), taskgraph_filepath))
    )
    task_graph["nluapi"] = ""
    task_graph["slotfillapi"] = ""
    with open(taskgraph_filepath, "w") as f:
        json.dump(task_graph, f, indent=4)


def init_worker(args: argparse.Namespace) -> None:
    """Initialize workers based on the provided configuration.

    This function initializes and sets up various workers required by the Arklex framework
    based on the configuration file. It supports different types of workers including:
    - FaissRAGWorker: For RAG (Retrieval-Augmented Generation) functionality
    - DataBaseWorker: For database operations including search, booking, and cancellation

    The function reads the worker configurations from the provided config file and
    initializes the appropriate workers based on their names. Each worker type
    requires specific setup procedures and dependencies.

    Args:
        args (argparse.Namespace): Command-line arguments containing:
            - config: Path to the configuration file
            - output_dir: Directory where worker data will be stored
    """
    # Load configuration from the specified file
    config: Dict[str, Any] = json.load(open(args.config))
    workers: List[Dict[str, Any]] = config["workers"]
    worker_names: Set[str] = set([worker["name"] for worker in workers])

    # Initialize FaissRAGWorker if specified in configuration
    if "FaissRAGWorker" in worker_names:
        logger.info("Initializing FaissRAGWorker...")
        build_rag(args.output_dir, config["rag_docs"])

    # Initialize DataBaseWorker and related workers if specified
    elif any(
        node in worker_names
        for node in (
            "DataBaseWorker",
            "search_show",
            "book_show",
            "check_booking",
            "cancel_booking",
        )
    ):
        logger.info("Initializing DataBaseWorker...")
        build_database(args.output_dir)


def load_documents(
    config: Dict[str, Any], document_dir: Optional[str] = None
) -> List[Dict[str, str]]:
    """Load documents from various sources specified in the config.

    Args:
        config (Dict[str, Any]): Configuration containing document sources
        document_dir (Optional[str]): Directory containing documents

    Returns:
        List[Dict[str, str]]: List of loaded documents
    """
    loader = Loader()
    all_docs = []

    # Process all document types consistently
    doc_types = ["instructions", "task_docs", "rag_docs"]
    for doc_type in doc_types:
        if doc_type in config:
            docs = config[doc_type]
            if isinstance(docs, list):
                for doc in docs:
                    source = doc.get("source")
                    doc_type = doc.get(
                        "type", "text"
                    )  # Default to text if type not specified
                    num_docs = doc.get("num", 1)

                    try:
                        if doc_type == "url":
                            urls = loader.get_all_urls(source, num_docs)
                            crawled_docs = loader.to_crawled_url_objs(urls)
                            all_docs.extend(crawled_docs)
                        elif doc_type == "file":
                            if os.path.isfile(source):
                                if source.lower().endswith(".zip"):
                                    with tempfile.TemporaryDirectory() as temp_dir:
                                        with zipfile.ZipFile(source, "r") as zip_ref:
                                            zip_ref.extractall(temp_dir)
                                        file_list = []
                                        for root, _, files in os.walk(temp_dir):
                                            for file in files:
                                                file_list.append(
                                                    os.path.join(root, file)
                                                )
                                        all_docs.extend(
                                            loader.to_crawled_local_objs(file_list)
                                        )
                                else:
                                    all_docs.extend(
                                        loader.to_crawled_local_objs([source])
                                    )
                            elif os.path.isdir(source):
                                file_list = [
                                    os.path.join(source, f) for f in os.listdir(source)
                                ]
                                all_docs.extend(loader.to_crawled_local_objs(file_list))
                            else:
                                raise FileNotFoundError(
                                    f"Source path '{source}' does not exist"
                                )
                        elif doc_type == "text":
                            all_docs.extend(loader.to_crawled_text([source]))
                        else:
                            raise ValueError(f"Unsupported document type: {doc_type}")
                    except Exception as e:
                        logger.error(f"Error processing document {source}: {str(e)}")
                        continue

    # Convert CrawledObjects to dictionaries
    return [doc.to_dict() for doc in all_docs]


def main():
    parser = argparse.ArgumentParser(
        description="Create a task graph from a config file"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for cache and results",
    )
    args = parser.parse_args()

    # Set up logging
    logger.setLevel(getattr(logging, args.log_level.upper()))

    # Load config
    with open(args.config, "r") as f:
        config = json.load(f)

    # Load documents
    documents = load_documents(config)
    logger.info(f"Loaded {len(documents)} documents")

    # Instantiate model
    model = PROVIDER_MAP.get(MODEL["llm_provider"], ChatOpenAI)(
        model=MODEL["model_type_or_path"], timeout=30000
    )

    # Determine output directory
    output_dir = args.output_dir or os.path.join(os.path.dirname(args.config), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize generator with model and output_dir
    generator = Generator(config, model, output_dir)

    # Generate task graph
    task_graph = generator.generate()
    logger.info("Task graph generated successfully")

    # Save the generated task graph
    taskgraph_filepath = generator.save_task_graph(task_graph)
    logger.info(f"Task graph saved to {taskgraph_filepath}")

    # Build RAG if specified
    if "rag_docs" in config:
        build_rag(os.path.dirname(args.config), config["rag_docs"])
        logger.info("RAG system built successfully")

    # Build database if specified
    if "database" in config:
        build_database(config["database"])
        logger.info("Database built successfully")


if __name__ == "__main__":
    main()

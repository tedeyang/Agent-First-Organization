#!/usr/bin/env python3
"""
Test runner for HITL (Human-in-the-Loop) integration tests.

This script provides a convenient way to run the HITL integration tests
with proper configuration and environment setup.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def setup_environment() -> None:
    """Set up the test environment with required environment variables."""
    os.environ.setdefault("OPENAI_API_KEY", "test_key")
    os.environ.setdefault("DATA_DIR", "./examples/hitl_server")
    os.environ.setdefault("MYSQL_USERNAME", "test_user")
    os.environ.setdefault("MYSQL_PASSWORD", "test_password")
    os.environ.setdefault("MYSQL_HOSTNAME", "localhost")
    os.environ.setdefault("MYSQL_PORT", "3306")
    os.environ.setdefault("MYSQL_DB_NAME", "test_db")
    os.environ.setdefault("ARKLEX_TEST_ENV", "local")
    os.environ.setdefault("PYTHONPATH", str(project_root))
    os.environ.setdefault("TESTING", "true")
    os.environ.setdefault("LOG_LEVEL", "WARNING")


def check_test_dependencies() -> bool:
    """Check if all required test dependencies are available."""
    required_modules = [
        "langchain",
        "langchain_community",
        "langchain_openai",
        "openai",
        "pytest",
    ]

    missing_modules = []
    for module_name in required_modules:
        if importlib.util.find_spec(module_name) is None:
            missing_modules.append(module_name)

    if missing_modules:
        print(f"❌ Missing required dependencies: {', '.join(missing_modules)}")
        print(
            "Please install test dependencies with: pip install pytest openai langchain langchain-community langchain-openai"
        )
        return False

    return True


def check_test_files() -> bool:
    """Check if required test files exist."""
    required_files = [
        "tests/integration/test_hitl_server.py",
        "examples/hitl_server/taskgraph.json",
        "examples/hitl_server/conftest.py",
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print("❌ Missing required test files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False

    return True


def run_tests(
    test_file: str | None = None, verbose: bool = False, markers: str | None = None
) -> bool:
    """Run the HITL integration tests."""
    setup_environment()

    if not check_test_dependencies():
        return False

    if not check_test_files():
        return False

    # Determine which tests to run
    if test_file:
        test_path = f"tests/integration/{test_file}"
    else:
        test_path = "tests/integration/test_hitl_server.py"

    # Build pytest command
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        test_path,
        "-v" if verbose else "",
        f"-m {markers}" if markers else "",
        "--tb=short",
        "--strict-markers",
        "--disable-warnings",
        "--no-header",
    ]

    # Remove empty strings
    cmd = [arg for arg in cmd if arg]

    print(f"Running tests with command: {' '.join(cmd)}")
    print(f"Working directory: {os.getcwd()}")
    print("-" * 80)

    try:
        subprocess.run(cmd, check=True, capture_output=False)
        print("-" * 80)
        print("✅ All tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print("-" * 80)
        print(f"❌ Tests failed with exit code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Unexpected error running tests: {e}")
        return False


def main() -> None:
    """Main entry point for the test runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Run HITL integration tests")
    parser.add_argument(
        "--test-file",
        type=str,
        help="Specific test file to run (e.g., test_hitl_server.py)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Run tests in verbose mode"
    )
    parser.add_argument(
        "-m",
        "--markers",
        type=str,
        help="Run only tests matching the given markers (e.g., 'hitl and not slow')",
    )
    parser.add_argument(
        "--list-tests",
        action="store_true",
        help="List available tests without running them",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check dependencies and files without running tests",
    )

    args = parser.parse_args()

    if args.check_only:
        print("🔍 Checking test environment...")
        setup_environment()

        if check_test_dependencies():
            print("✅ All dependencies are available")
        else:
            print("❌ Missing dependencies")
            sys.exit(1)

        if check_test_files():
            print("✅ All required test files exist")
        else:
            print("❌ Missing test files")
            sys.exit(1)

        print("✅ Test environment is ready")
        return

    if args.list_tests:
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/integration/test_hitl_server.py",
            "--collect-only",
            "-q",
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error listing tests: {e}")
            sys.exit(1)
        return

    # Run the tests
    success = run_tests(
        test_file=args.test_file, verbose=args.verbose, markers=args.markers
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

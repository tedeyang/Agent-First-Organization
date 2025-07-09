"""
Test script to verify that create.py and run.py work with real API keys from .env file.
Tests with OPENAI_API_KEY, ANTHROPIC_API_KEY, and GOOGLE_API_KEY.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import requests

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arklex.utils.provider_utils import get_api_key_for_provider, get_provider_config

# Constants
API_KEYS = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]
PROVIDER_MAP = {
    "OPENAI_API_KEY": "openai",
    "ANTHROPIC_API_KEY": "anthropic",
    "GOOGLE_API_KEY": "gemini",
}
MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-haiku-20240307",
    "gemini": "gemini-1.5-flash",
}
API_ENDPOINTS = [
    ("https://api.openai.com", "OpenAI API"),
    ("https://api.anthropic.com", "Anthropic API"),
    ("https://generativelanguage.googleapis.com", "Google Gemini API"),
]


# Environment and Configuration
def load_env_file() -> dict[str, str]:
    """Load environment variables from .env file."""
    env_vars = {}
    env_file = Path(".env")

    if not env_file.exists():
        _print_env_file_instructions(env_file)
        return env_vars

    print(f"📁 Found .env file: {env_file.absolute()}")
    env_vars = _parse_env_file(env_file)
    print(f"✅ Loaded {len(env_vars)} environment variables")
    _print_api_key_details(env_vars)
    return env_vars


def _print_env_file_instructions(env_file: Path) -> None:
    """Print instructions for creating .env file."""
    print(f"⚠️  No .env file found at {env_file.absolute()}")
    print("💡 Please create a .env file with your API keys:")
    print("   OPENAI_API_KEY=your_openai_key_here")
    print("   ANTHROPIC_API_KEY=your_anthropic_key_here")
    print("   GOOGLE_API_KEY=your_google_key_here")


def _parse_env_file(env_file: Path) -> dict[str, str]:
    """Parse .env file and return environment variables."""
    env_vars = {}
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env_vars[key] = value
    return env_vars


def _print_api_key_details(env_vars: dict[str, str]) -> None:
    """Print details about found API keys."""
    found_keys = [key for key in API_KEYS if env_vars.get(key)]

    if not found_keys:
        _print_missing_api_keys_instructions()
        return

    print(f"🔑 Found API keys: {', '.join(found_keys)}")
    print("📊 API Key Details:")
    for key in found_keys:
        api_key = env_vars[key]
        print(f"   {key}: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else ''}")


def _print_missing_api_keys_instructions() -> None:
    """Print instructions for adding API keys."""
    print("⚠️  No API keys found in .env file")
    print("💡 Please add your API keys to the .env file:")
    print("   OPENAI_API_KEY=your_openai_key_here")
    print("   ANTHROPIC_API_KEY=your_anthropic_key_here")
    print("   GOOGLE_API_KEY=your_google_key_here")


# API Testing
def validate_api_key(provider: str, api_key: str) -> bool:
    """Test if an API key is valid by making a simple API call."""
    print(f"🔍 Validating {provider} API key...")
    print(f"🔑 API key starts with: {api_key[:10]}...")

    try:
        if provider == "openai":
            return _test_openai_api(api_key)
        elif provider == "anthropic":
            return _test_anthropic_api(api_key)
        elif provider == "gemini":
            return _test_gemini_api(api_key)
        else:
            print(f"❌ Unknown provider: {provider}")
            return False

    except requests.exceptions.Timeout:
        print(f"⏰ Timeout validating {provider} API key (10s timeout)")
        return False
    except requests.exceptions.ConnectionError:
        print(f"🌐 Connection error validating {provider} API key")
        print("💡 Check your internet connection and try again")
        return False
    except Exception as e:
        print(f"❌ Error validating {provider} API key: {e}")
        return False


def _test_openai_api(api_key: str) -> bool:
    """Test OpenAI API key."""
    print("🌐 Making OpenAI API call...")
    print("📡 Request URL: https://api.openai.com/v1/chat/completions")
    print("📡 Request model: gpt-3.5-turbo")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 5,
    }

    print("📤 Sending request...")
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=data,
        timeout=10,
    )

    return _handle_api_response(response, "OpenAI")


def _test_anthropic_api(api_key: str) -> bool:
    """Test Anthropic API key."""
    print("🌐 Making Anthropic API call...")
    print("📡 Request URL: https://api.anthropic.com/v1/messages")
    print("📡 Request model: claude-3-haiku-20240307")

    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    data = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 5,
        "messages": [{"role": "user", "content": "Hello"}],
    }

    print("📤 Sending request...")
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=data,
        timeout=10,
    )

    return _handle_api_response(response, "Anthropic")


def _test_gemini_api(api_key: str) -> bool:
    """Test Google Gemini API key."""
    print("🌐 Making Google Gemini API call...")
    print(
        "📡 Request URL: https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    )
    print("📡 Request model: gemini-1.5-flash")

    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": "Hello"}]}],
        "generationConfig": {"maxOutputTokens": 5},
    }

    print("📤 Sending request...")
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}",
        headers=headers,
        json=data,
        timeout=10,
    )

    return _handle_api_response(response, "Google Gemini")


def _handle_api_response(response: requests.Response, provider_name: str) -> bool:
    """Handle API response and print results."""
    print(f"📡 {provider_name} response status: {response.status_code}")

    if response.status_code != 200:
        print(f"❌ {provider_name} API key validation failed: {response.status_code}")
        print(f"📤 Error response: {response.text[:200]}...")
        return False

    print(f"✅ {provider_name} API key is valid")
    _print_response_preview(response, provider_name)
    return True


def _print_response_preview(response: requests.Response, provider_name: str) -> None:
    """Print a preview of the API response."""
    try:
        response_data = response.json()
        if (
            provider_name == "OpenAI"
            and "choices" in response_data
            and len(response_data["choices"]) > 0
        ):
            content = response_data["choices"][0].get("message", {}).get("content", "")
            print(f"📤 Response preview: {content[:50]}...")
        elif (
            provider_name == "Anthropic"
            and "content" in response_data
            and len(response_data["content"]) > 0
        ):
            content = response_data["content"][0].get("text", "")
            print(f"📤 Response preview: {content[:50]}...")
        elif (
            provider_name == "Google Gemini"
            and "candidates" in response_data
            and len(response_data["candidates"]) > 0
        ):
            content = (
                response_data["candidates"][0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
            print(f"📤 Response preview: {content[:50]}...")
    except Exception:
        pass


# Provider Utilities Testing
def test_provider_utils_with_real_keys(env_vars: dict[str, str]) -> bool:
    """Test that provider utilities work with real API keys."""
    print("\n🔍 Testing provider utilities with real API keys...")
    print("📋 Testing order: OpenAI → Anthropic → Gemini")

    providers = ["openai", "anthropic", "gemini"]
    results = []

    for provider in providers:
        env_key = f"{provider.upper()}_API_KEY"
        api_key = env_vars.get(env_key, "")

        if not api_key:
            print(f"⚠️  No {provider} API key found in .env file")
            results.append(False)
            continue

        result = _test_provider_utility(provider, api_key)
        results.append(result)

    return all(results)


def _test_provider_utility(provider: str, api_key: str) -> bool:
    """Test a single provider's utilities."""
    print(f"\n🧪 Testing {provider}...")
    print(f"🔑 Found {provider} API key: {api_key[:10]}...")

    if not validate_api_key(provider, api_key):
        print(f"❌ {provider} API key is invalid")
        return False

    print(f"✅ {provider} API key is valid")

    if not _test_get_api_key_for_provider(provider, api_key):
        return False

    return _test_get_provider_config(provider, api_key)


def _test_get_api_key_for_provider(provider: str, api_key: str) -> bool:
    """Test get_api_key_for_provider function."""
    try:
        retrieved_key = get_api_key_for_provider(provider)
        if retrieved_key != api_key:
            print(
                f"❌ get_api_key_for_provider('{provider}') failed - keys don't match"
            )
            return False
        print(f"✅ get_api_key_for_provider('{provider}') works correctly")
        return True
    except Exception as e:
        print(f"❌ Error testing {provider}: {e}")
        return False


def _test_get_provider_config(provider: str, api_key: str) -> bool:
    """Test get_provider_config function."""
    try:
        config = get_provider_config(provider, f"{provider}-test-model")
        if config["api_key"] != api_key:
            print(f"❌ get_provider_config('{provider}') failed - keys don't match")
            return False
        print(f"✅ get_provider_config('{provider}') works correctly")
        return True
    except Exception as e:
        print(f"❌ Error testing {provider} config: {e}")
        return False


# Process Management
def run_command_with_realtime_output(
    cmd: list[str],
    env: dict[str, str],
    input_text: str | None = None,
    timeout: int = 600,
) -> tuple[int, str, str]:
    """Run a command and show real-time output with enhanced logging."""
    print(f"🚀 Running: {' '.join(cmd)}")
    print(f"⏱️  Timeout: {timeout} seconds")
    print("=" * 80)

    if input_text:
        return _handle_process_with_input(cmd, env, input_text, timeout)
    else:
        return _handle_process_without_input(cmd, env, timeout)


def _handle_process_with_input(
    cmd: list[str], env: dict[str, str], input_text: str, timeout: int
) -> tuple[int, str, str]:
    """Handle process execution with input."""
    print("📝 Sending input to process...")
    print(f"📤 Input preview: {repr(input_text[:100])}...")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE,
        text=True,
        env=env,
        bufsize=0,
        universal_newlines=True,
    )

    try:
        start_time = time.time()
        print("🔄 Waiting for process output...")
        print("💡 Output will be displayed in real-time as it comes...")
        print("⏳ Process started, waiting for first output...")

        stdout, stderr = process.communicate(input=input_text, timeout=timeout)
        return _process_final_output(process, stdout, stderr, start_time)

    except subprocess.TimeoutExpired:
        print(f"⏰ Command timed out after {timeout} seconds")
        process.terminate()
        raise


def _handle_process_without_input(
    cmd: list[str], env: dict[str, str], timeout: int
) -> tuple[int, str, str]:
    """Handle process execution without input."""
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=None,
        text=True,
        env=env,
        bufsize=0,
        universal_newlines=True,
    )

    stdout_lines = []
    stderr_lines = []
    output_count = 0

    try:
        start_time = time.time()
        print("🔄 Waiting for process output...")
        print("💡 Output will be displayed in real-time as it comes...")
        print("⏳ Process started, waiting for first output...")

        stdout_lines, stderr_lines, output_count = _monitor_process_output(
            process, stdout_lines, stderr_lines, start_time, timeout
        )

        remaining_stdout, remaining_stderr = process.communicate()
        stdout_lines, stderr_lines = _add_remaining_output(
            remaining_stdout, remaining_stderr, stdout_lines, stderr_lines
        )

        return _process_final_output(
            process, "".join(stdout_lines), "".join(stderr_lines), start_time
        )

    except subprocess.TimeoutExpired:
        print(f"⏰ Command timed out after {timeout} seconds")
        process.terminate()
        raise


def _monitor_process_output(
    process: subprocess.Popen,
    stdout_lines: list,
    stderr_lines: list,
    start_time: float,
    timeout: int,
) -> tuple[list, list, int]:
    """Monitor process output in real-time."""
    last_output_time = time.time()
    output_count = 0
    last_progress_time = time.time()

    while process.poll() is None:
        if time.time() - start_time > timeout:
            print(f"⏰ Command timed out after {timeout} seconds")
            process.terminate()
            raise subprocess.TimeoutExpired(process.args, timeout)

        stdout_lines, stderr_lines, output_count, last_output_time = (
            _read_available_output(
                process, stdout_lines, stderr_lines, output_count, last_output_time
            )
        )

        _print_progress_if_needed(
            start_time, last_output_time, output_count, last_progress_time
        )
        time.sleep(0.01)

    return stdout_lines, stderr_lines, output_count


def _read_available_output(
    process: subprocess.Popen,
    stdout_lines: list,
    stderr_lines: list,
    output_count: int,
    last_output_time: float,
) -> tuple[list, list, int, float]:
    """Read available output from process streams."""
    import select

    reads = []
    if process.stdout:
        reads.append(process.stdout)
    if process.stderr:
        reads.append(process.stderr)

    if not reads:
        return stdout_lines, stderr_lines, output_count, last_output_time

    try:
        ready, _, _ = select.select(reads, [], [], 0.1)
        for stream in ready:
            if stream == process.stdout:
                line = stream.readline()
                if line:
                    output_count += 1
                    print(f"📤 STDOUT [{output_count}]: {line.rstrip()}")
                    stdout_lines.append(line)
                    last_output_time = time.time()
            elif stream == process.stderr:
                line = stream.readline()
                if line:
                    output_count += 1
                    print(f"📤 STDERR [{output_count}]: {line.rstrip()}")
                    stderr_lines.append(line)
                    last_output_time = time.time()
    except (OSError, ValueError):
        pass

    return stdout_lines, stderr_lines, output_count, last_output_time


def _print_progress_if_needed(
    start_time: float,
    last_output_time: float,
    output_count: int,
    last_progress_time: float,
) -> float:
    """Print progress message if needed."""
    current_time = time.time()
    if (
        current_time - last_output_time > 2
        and current_time - start_time > 3
        and current_time - last_progress_time > 5
    ):
        elapsed = int(current_time - start_time)
        print(f"⏳ Still running... (elapsed: {elapsed}s, outputs: {output_count})")
        return current_time
    return last_progress_time


def _add_remaining_output(
    remaining_stdout: str, remaining_stderr: str, stdout_lines: list, stderr_lines: list
) -> tuple[list, list]:
    """Add any remaining output to the lines."""
    if remaining_stdout:
        print(f"📤 FINAL STDOUT: {remaining_stdout}")
        stdout_lines.append(remaining_stdout)
    if remaining_stderr:
        print(f"📤 FINAL STDERR: {remaining_stderr}")
        stderr_lines.append(remaining_stderr)
    return stdout_lines, stderr_lines


def _process_final_output(
    process: subprocess.Popen, stdout: str, stderr: str, start_time: float
) -> tuple[int, str, str]:
    """Process final output and print results."""
    output_count = 0
    stdout_lines = []
    stderr_lines = []

    if stdout:
        lines = stdout.split("\n")
        for line in lines:
            if line.strip():
                output_count += 1
                print(f"📤 STDOUT [{output_count}]: {line}")
                stdout_lines.append(line + "\n")

    if stderr:
        lines = stderr.split("\n")
        for line in lines:
            if line.strip():
                output_count += 1
                print(f"📤 STDERR [{output_count}]: {line}")
                stderr_lines.append(line + "\n")

    elapsed_time = int(time.time() - start_time)
    print("=" * 80)
    print(
        f"🏁 Command finished with return code: {process.returncode} (elapsed: {elapsed_time}s, total outputs: {output_count})"
    )

    if process.returncode == 0:
        print("✅ Command completed successfully")
    else:
        print("❌ Command failed")

    return process.returncode, "".join(stdout_lines), "".join(stderr_lines)


# Environment Testing
def test_environment_loading() -> bool:
    """Test that .env file loading works correctly."""
    print("\n🔍 Testing .env file loading...")

    try:
        valid_keys, invalid_keys = _validate_loaded_keys()

        if valid_keys:
            print(f"✅ Valid API keys: {', '.join(valid_keys)}")
            return True
        else:
            print("❌ No valid API keys found")
            return False

    except Exception as e:
        print(f"❌ Error testing .env file loading: {e}")
        return False


def _validate_loaded_keys() -> tuple[list[str], list[str]]:
    """Validate loaded API keys and return valid and invalid keys."""
    from dotenv import load_dotenv

    load_dotenv()

    valid_keys = []
    invalid_keys = []

    for key_name in API_KEYS:
        api_key = os.getenv(key_name, "")
        if not api_key:
            continue

        provider = PROVIDER_MAP[key_name]
        if validate_api_key(provider, api_key):
            valid_keys.append(key_name)
            print(f"✅ {key_name} is valid")
        else:
            invalid_keys.append(key_name)
            print(f"❌ {key_name} is invalid")

    return valid_keys, invalid_keys


# Customer Service Testing
def test_customer_service_example_directly() -> bool:
    """Test the customer service example directly with real API keys."""
    print("\n🔍 Testing customer service example directly...")
    print("📋 Testing order: OpenAI → Anthropic → Gemini")
    print("🔧 Testing both create.py and run.py for each provider")

    env_vars = load_env_file()
    providers = ["openai", "anthropic", "gemini"]
    results = []

    for provider in providers:
        env_key = f"{provider.upper()}_API_KEY"
        api_key = env_vars.get(env_key, "")

        if not api_key:
            print(f"⚠️  No {provider} API key found in .env file")
            results.append(False)
            continue

        result = _test_customer_service_with_provider(provider, api_key, env_vars)
        results.append(result)

    return any(results)


def _test_customer_service_with_provider(
    provider: str, api_key: str, env_vars: dict[str, str]
) -> bool:
    """Test customer service example with a specific provider."""
    print(f"\n🧪 Testing customer service example with {provider}...")
    print(f"🔧 Testing both create.py and run.py with {provider}")

    if not validate_api_key(provider, api_key):
        print(f"❌ Skipping {provider} - API key is invalid")
        return False

    env_key = f"{provider.upper()}_API_KEY"
    env = os.environ.copy()
    env[env_key] = api_key

    config_path = "./examples/customer_service/customer_service_config.json"
    if not os.path.exists(config_path):
        print(f"❌ Customer service config not found: {config_path}")
        return False

    try:
        if not _test_create_py(provider, api_key, env, config_path):
            return False

        if not _test_run_py(provider, api_key, env):
            return False

        print(f"🎉 Both create.py and run.py work with {provider}!")
        return True

    except subprocess.TimeoutExpired:
        print(f"❌ Test timed out with {provider}")
        return False
    except Exception as e:
        print(f"❌ Error testing {provider}: {e}")
        return False


def _test_create_py(provider: str, api_key: str, env: dict, config_path: str) -> bool:
    """Test create.py with a specific provider."""
    print(f"📝 Step 1: Testing create.py with {provider}...")

    create_cmd = [
        sys.executable,
        "create.py",
        "--config",
        config_path,
        "--output-dir",
        "./examples/customer_service",
        "--llm-provider",
        provider,
        "--model",
        MODELS[provider],
        "--no-ui",
    ]

    create_returncode, create_stdout, create_stderr = run_command_with_realtime_output(
        create_cmd, env, timeout=600
    )

    if create_returncode != 0:
        print(f"❌ create.py failed with {provider}: {create_stderr}")
        return False

    print(f"✅ create.py succeeded with {provider}")
    return True


def _test_run_py(provider: str, api_key: str, env: dict) -> bool:
    """Test run.py with a specific provider."""
    print(f"🤖 Step 2: Testing run.py with {provider}...")
    print("💬 Sending test input: 'Tell me about your robots.'")

    run_cmd = [
        sys.executable,
        "run.py",
        "--input-dir",
        "./examples/customer_service",
        "--llm-provider",
        provider,
        "--model",
        MODELS[provider],
    ]

    test_input = "Tell me about your robots.\nquit\n"
    run_returncode, run_stdout, run_stderr = run_command_with_realtime_output(
        run_cmd, env=env, input_text=test_input, timeout=60
    )

    if run_returncode != 0:
        print(f"❌ run.py failed with {provider}: {run_stderr}")
        return False

    print(f"✅ run.py succeeded with {provider}")
    print(f"📤 Agent response preview: {run_stdout[:500]}...")
    return True


# Connectivity Testing
def check_connectivity(url: str, name: str, timeout: int = 5) -> bool:
    """Check connectivity to a specific URL."""
    try:
        requests.get(url, timeout=timeout)
        print(f"✅ Network connectivity to {name} is working")
        return True
    except Exception as e:
        print(f"⚠️  {name} connectivity test failed: {e}")
        return False


def _check_package_availability() -> list[str]:
    """Check if required packages are installed and return missing ones."""
    required_packages = ["requests", "python-dotenv"]
    missing_packages = []

    for package in required_packages:
        try:
            import importlib.util

            if package == "requests":
                if importlib.util.find_spec("requests") is None:
                    raise ImportError
                print("✅ requests is installed")
            elif package == "python-dotenv":
                if importlib.util.find_spec("dotenv") is None:
                    raise ImportError
                print("✅ python-dotenv is installed")
        except ImportError:
            print(f"❌ {package} is not installed")
            missing_packages.append(package)

    return missing_packages


def _test_api_connectivity() -> tuple[int, int]:
    """Test connectivity to API endpoints and return success/total counts."""
    connectivity_results = []
    for url, name in API_ENDPOINTS:
        result = check_connectivity(url, name)
        connectivity_results.append(result)

    successful_connections = sum(connectivity_results)
    total_connections = len(connectivity_results)

    if successful_connections == total_connections:
        print(f"✅ All {total_connections} API endpoints are reachable")
    elif successful_connections > 0:
        print(
            f"⚠️  {successful_connections}/{total_connections} API endpoints are reachable"
        )
    else:
        print(f"❌ None of the {total_connections} API endpoints are reachable")

    print("💡 If any connectivity tests failed, API key validation might be affected")
    return successful_connections, total_connections


def check_requirements() -> bool:
    """Check if required packages are installed."""
    print("🔍 Checking requirements...")

    missing_packages = _check_package_availability()
    if missing_packages:
        print(
            f"\n⚠️  Please install missing packages: pip install {' '.join(missing_packages)}"
        )
        return False

    print("✅ All required packages are installed")

    print("🌐 Testing network connectivity...")
    successful, total = _test_api_connectivity()
    return True


# Test Execution
def _run_test_step(
    step_name: str, test_func: callable, *args: object
) -> tuple[str, bool]:
    """Run a test step and return the result."""
    print(f"\n{'=' * 60}")
    print(f"🔍 STEP: {step_name}")
    print(f"{'=' * 60}")
    result = test_func(*args)
    return step_name, result


def main() -> bool:
    """Run all tests."""
    print("🚀 Testing create.py and run.py with real API keys...\n")
    print("This script will:")
    print("1. Check required packages")
    print("2. Load API keys from .env file")
    print("3. Validate each API key with actual API calls")
    print("4. Test provider utilities with real keys (OpenAI → Anthropic → Gemini)")
    print("5. Test customer service example end-to-end (OpenAI → Anthropic → Gemini)")
    print("   - Tests both create.py and run.py for each provider")

    if not check_requirements():
        return False

    env_vars = load_env_file()

    test_steps = [
        ("Environment Loading", test_environment_loading),
        ("Provider Utilities", test_provider_utils_with_real_keys, env_vars),
        ("Customer Service Example", test_customer_service_example_directly),
    ]

    results = []
    for step in test_steps:
        step_name = step[0]
        test_func = step[1]
        args = step[2:] if len(step) > 2 else []
        result = _run_test_step(step_name, test_func, *args)
        results.append(result)

    _print_final_summary(results)
    return _all_tests_passed(results)


def _print_final_summary(results: list[tuple[str, bool]]) -> None:
    """Print final test summary."""
    print(f"\n{'=' * 60}")
    print("📊 FINAL TEST SUMMARY")
    print(f"{'=' * 60}")

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1

    print(f"{'=' * 60}")
    print(f"Total: {total}, Passed: {passed}, Failed: {total - passed}")


def _all_tests_passed(results: list[tuple[str, bool]]) -> bool:
    """Check if all tests passed and print appropriate message."""
    passed = sum(1 for _, result in results if result)
    total = len(results)

    if passed == total:
        print("🎉 All tests passed! create.py and run.py work with real API keys.")
        print("✅ API key validation successful")
        print("✅ Provider utilities working correctly")
        print("✅ create.py working with --no-ui flag")
        print("✅ run.py working in interactive mode")
        print("✅ End-to-end customer service example working")
        print("✅ All providers tested in order: OpenAI → Anthropic → Gemini")
        return True
    else:
        print("⚠️  Some tests failed. Please check:")
        print("   - Your .env file contains valid API keys")
        print("   - API keys have sufficient credits/permissions")
        print("   - Network connectivity for API calls")
        print("   - Required packages are installed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

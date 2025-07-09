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


def load_env_file() -> dict[str, str]:
    """Load environment variables from .env file."""
    env_vars = {}
    env_file = Path(".env")

    if env_file.exists():
        print(f"📁 Found .env file: {env_file.absolute()}")
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key] = value
        print(f"✅ Loaded {len(env_vars)} environment variables")

        # Show which API keys are present
        api_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]
        found_keys = []
        for key in api_keys:
            if env_vars.get(key):
                found_keys.append(key)

        if found_keys:
            print(f"🔑 Found API keys: {', '.join(found_keys)}")
            print("📊 API Key Details:")
            for key in found_keys:
                api_key = env_vars[key]
                print(
                    f"   {key}: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else ''}"
                )
        else:
            print("⚠️  No API keys found in .env file")
            print("💡 Please add your API keys to the .env file:")
            print("   OPENAI_API_KEY=your_openai_key_here")
            print("   ANTHROPIC_API_KEY=your_anthropic_key_here")
            print("   GOOGLE_API_KEY=your_google_key_here")
    else:
        print(f"⚠️  No .env file found at {env_file.absolute()}")
        print("💡 Please create a .env file with your API keys:")
        print("   OPENAI_API_KEY=your_openai_key_here")
        print("   ANTHROPIC_API_KEY=your_anthropic_key_here")
        print("   GOOGLE_API_KEY=your_google_key_here")

    return env_vars


def validate_api_key(provider: str, api_key: str) -> bool:
    """Test if an API key is valid by making a simple API call."""
    print(f"🔍 Validating {provider} API key...")
    print(f"🔑 API key starts with: {api_key[:10]}...")

    try:
        if provider == "openai":
            # Test OpenAI API
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
            print(f"📡 OpenAI response status: {response.status_code}")
            if response.status_code == 200:
                print("✅ OpenAI API key is valid")
                try:
                    response_data = response.json()
                    if "choices" in response_data and len(response_data["choices"]) > 0:
                        content = (
                            response_data["choices"][0]
                            .get("message", {})
                            .get("content", "")
                        )
                        print(f"📤 Response preview: {content[:50]}...")
                except Exception:
                    pass
                return True
            else:
                print(f"❌ OpenAI API key validation failed: {response.status_code}")
                print(f"📤 Error response: {response.text[:200]}...")
                return False

        elif provider == "anthropic":
            # Test Anthropic API
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
            print(f"📡 Anthropic response status: {response.status_code}")
            if response.status_code == 200:
                print("✅ Anthropic API key is valid")
                try:
                    response_data = response.json()
                    if "content" in response_data and len(response_data["content"]) > 0:
                        content = response_data["content"][0].get("text", "")
                        print(f"📤 Response preview: {content[:50]}...")
                except Exception:
                    pass
                return True
            else:
                print(f"❌ Anthropic API key validation failed: {response.status_code}")
                print(f"📤 Error response: {response.text[:200]}...")
                return False

        elif provider == "gemini":
            # Test Google Gemini API
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
            print(f"📡 Google Gemini response status: {response.status_code}")
            if response.status_code == 200:
                print("✅ Google Gemini API key is valid")
                try:
                    response_data = response.json()
                    if (
                        "candidates" in response_data
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
                return True
            else:
                print(
                    f"❌ Google Gemini API key validation failed: {response.status_code}"
                )
                print(f"📤 Error response: {response.text[:200]}...")
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

    return False


def test_provider_utils_with_real_keys(env_vars: dict[str, str]) -> bool:
    """Test that provider utilities work with real API keys."""
    print("\n🔍 Testing provider utilities with real API keys...")

    providers = ["openai", "anthropic", "gemini"]
    results = []

    for provider in providers:
        env_key = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GOOGLE_API_KEY",
        }[provider]

        api_key = env_vars.get(env_key, "")

        if api_key:
            print(f"\n🧪 Testing {provider}...")
            print(f"🔑 Found {provider} API key: {api_key[:10]}...")

            # First validate the API key
            if validate_api_key(provider, api_key):
                print(f"✅ {provider} API key is valid")

                # Test get_api_key_for_provider
                try:
                    retrieved_key = get_api_key_for_provider(provider)
                    if retrieved_key == api_key:
                        print(
                            f"✅ get_api_key_for_provider('{provider}') works correctly"
                        )
                        results.append(True)
                    else:
                        print(
                            f"❌ get_api_key_for_provider('{provider}') failed - keys don't match"
                        )
                        results.append(False)
                except Exception as e:
                    print(f"❌ Error testing {provider}: {e}")
                    results.append(False)

                # Test get_provider_config
                try:
                    config = get_provider_config(provider, f"{provider}-test-model")
                    if config["api_key"] == api_key:
                        print(f"✅ get_provider_config('{provider}') works correctly")
                        results.append(True)
                    else:
                        print(
                            f"❌ get_provider_config('{provider}') failed - keys don't match"
                        )
                        results.append(False)
                except Exception as e:
                    print(f"❌ Error testing {provider} config: {e}")
                    results.append(False)
            else:
                print(f"❌ {provider} API key is invalid")
                results.append(False)
        else:
            print(f"⚠️  No {provider} API key found in .env file")
            results.append(False)

    return all(results)


def run_command_with_realtime_output(
    cmd: list[str], env: dict[str, str], input_text: str = None, timeout: int = 600
) -> tuple[int, str, str]:
    """Run a command and show real-time output with enhanced logging."""
    print(f"🚀 Running: {' '.join(cmd)}")
    print(f"⏱️  Timeout: {timeout} seconds")
    print("=" * 80)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE if input_text else None,
        text=True,
        env=env,
        bufsize=0,  # Unbuffered for immediate output
        universal_newlines=True,
    )

    stdout_lines = []
    stderr_lines = []
    last_output_time = time.time()
    output_count = 0
    last_progress_time = time.time()

    try:
        start_time = time.time()

        # If we have input, write it to stdin
        if input_text:
            print("📝 Sending input to process...")
            print(f"📤 Input preview: {repr(input_text[:100])}...")
            process.stdin.write(input_text)
            process.stdin.flush()
            process.stdin.close()

        print("🔄 Waiting for process output...")
        print("💡 Output will be displayed in real-time as it comes...")
        print("⏳ Process started, waiting for first output...")

        while process.poll() is None:
            # Check for timeout
            if time.time() - start_time > timeout:
                print(f"⏰ Command timed out after {timeout} seconds")
                process.terminate()
                raise subprocess.TimeoutExpired(cmd, timeout)

            # Use select for non-blocking I/O (if available)
            import select

            # Check if we can read from stdout or stderr
            reads = []
            if process.stdout:
                reads.append(process.stdout)
            if process.stderr:
                reads.append(process.stderr)

            if reads:
                try:
                    ready, _, _ = select.select(reads, [], [], 0.1)  # 100ms timeout

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
                    # select might fail on some systems, fall back to polling
                    pass

            # Show progress indicator if no output for a while
            current_time = time.time()
            if (
                current_time - last_output_time > 2
                and current_time - start_time > 3
                and current_time - last_progress_time > 5
            ):
                elapsed = int(current_time - start_time)
                print(
                    f"⏳ Still running... (elapsed: {elapsed}s, outputs: {output_count})"
                )
                last_progress_time = current_time
            time.sleep(0.01)  # Very short sleep for more responsive output

        # Read any remaining output
        remaining_stdout, remaining_stderr = process.communicate()
        if remaining_stdout:
            print(f"📤 FINAL STDOUT: {remaining_stdout}")
            stdout_lines.append(remaining_stdout)
        if remaining_stderr:
            print(f"📤 FINAL STDERR: {remaining_stderr}")
            stderr_lines.append(remaining_stderr)

    except subprocess.TimeoutExpired:
        print(f"⏰ Command timed out after {timeout} seconds")
        process.terminate()
        raise

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


def test_environment_loading() -> bool:
    """Test that .env file loading works correctly."""
    print("\n🔍 Testing .env file loading...")

    try:
        from dotenv import load_dotenv

        # Load .env file
        load_dotenv()

        # Check if API keys are loaded
        openai_key = os.getenv("OPENAI_API_KEY", "")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        google_key = os.getenv("GOOGLE_API_KEY", "")

        keys_found = []
        if openai_key:
            keys_found.append("OPENAI_API_KEY")
        if anthropic_key:
            keys_found.append("ANTHROPIC_API_KEY")
        if google_key:
            keys_found.append("GOOGLE_API_KEY")

        if keys_found:
            print(f"✅ Loaded API keys: {', '.join(keys_found)}")

            # Validate each key
            valid_keys = []
            for key_name in keys_found:
                provider_map = {
                    "OPENAI_API_KEY": "openai",
                    "ANTHROPIC_API_KEY": "anthropic",
                    "GOOGLE_API_KEY": "gemini",
                }
                provider = provider_map[key_name]
                api_key = os.getenv(key_name, "")
                if validate_api_key(provider, api_key):
                    valid_keys.append(key_name)
                    print(f"✅ {key_name} is valid")
                else:
                    print(f"❌ {key_name} is invalid")

            if valid_keys:
                print(f"✅ Valid API keys: {', '.join(valid_keys)}")
                return True
            else:
                print("❌ No valid API keys found")
                return False
        else:
            print("⚠️  No API keys found in .env file")
            return False

    except Exception as e:
        print(f"❌ Error testing .env file loading: {e}")
        return False


def test_customer_service_example_directly() -> bool:
    """Test the customer service example directly with real API keys."""
    print("\n🔍 Testing customer service example directly...")

    # Check if customer service example exists
    config_path = "./examples/customer_service/customer_service_config.json"
    if not os.path.exists(config_path):
        print(f"❌ Customer service config not found: {config_path}")
        return False

    # Load environment variables
    env_vars = load_env_file()

    # Test with OpenAI (most likely to be available)
    providers = ["openai", "anthropic", "gemini"]
    models = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-haiku-20240307",
        "gemini": "gemini-1.5-flash",
    }

    for provider in providers:
        env_key = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GOOGLE_API_KEY",
        }[provider]

        api_key = env_vars.get(env_key, "")
        if not api_key:
            continue

        print(f"\n🧪 Testing customer service example with {provider}...")

        # Validate API key first
        if not validate_api_key(provider, api_key):
            print(f"❌ Skipping {provider} - API key is invalid")
            continue

        # Set environment variable
        env = os.environ.copy()
        env[env_key] = api_key

        try:
            # Test create.py
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
                models[provider],
                "--no-ui",
            ]

            print(f"📝 Creating customer service agent with {provider}...")
            create_returncode, create_stdout, create_stderr = (
                run_command_with_realtime_output(create_cmd, env, timeout=600)
            )

            if create_returncode == 0:
                print(f"✅ create.py succeeded with {provider}")

                # Test run.py
                run_cmd = [
                    sys.executable,
                    "run.py",
                    "--input-dir",
                    "./examples/customer_service",
                    "--llm-provider",
                    provider,
                    "--model",
                    models[provider],
                    "--no-ui",
                ]

                print(f"🤖 Running customer service agent with {provider}...")
                print("💬 Sending test input: 'Tell me about your robots.'")

                # Test with a simple input
                test_input = "Tell me about your robots.\nquit\n"

                run_returncode, run_stdout, run_stderr = (
                    run_command_with_realtime_output(
                        run_cmd, env=env, input_text=test_input, timeout=60
                    )
                )

                if run_returncode == 0:
                    print(f"✅ run.py succeeded with {provider}")
                    print(f"📤 Agent response preview: {run_stdout[:500]}...")
                    return True
                else:
                    print(f"❌ run.py failed with {provider}: {run_stderr}")
            else:
                print(f"❌ create.py failed with {provider}: {create_stderr}")

        except subprocess.TimeoutExpired:
            print(f"❌ Test timed out with {provider}")
        except Exception as e:
            print(f"❌ Error testing {provider}: {e}")

    print("❌ No working provider found for customer service example")
    return False


def check_requirements() -> bool:
    """Check if required packages are installed."""
    print("🔍 Checking requirements...")

    required_packages = ["requests", "python-dotenv"]
    missing_packages = []

    for package in required_packages:
        try:
            if package == "requests":
                import importlib.util

                if importlib.util.find_spec("requests") is not None:
                    print("✅ requests is installed")
                else:
                    raise ImportError
            elif package == "python-dotenv":
                import importlib.util

                if importlib.util.find_spec("dotenv") is not None:
                    print("✅ python-dotenv is installed")
                else:
                    raise ImportError
        except ImportError:
            print(f"❌ {package} is not installed")
            missing_packages.append(package)

    if missing_packages:
        print(
            f"\n⚠️  Please install missing packages: pip install {' '.join(missing_packages)}"
        )
        return False

    print("✅ All required packages are installed")

    # Test network connectivity
    print("🌐 Testing network connectivity...")
    try:
        requests.get("https://api.openai.com", timeout=5)
        print("✅ Network connectivity to OpenAI API is working")
    except Exception as e:
        print(f"⚠️  Network connectivity test failed: {e}")
        print("   This might affect API key validation")

    return True


def main() -> bool:
    """Run all tests."""
    print("🚀 Testing create.py and run.py with real API keys...\n")
    print("This script will:")
    print("1. Check required packages")
    print("2. Load API keys from .env file")
    print("3. Validate each API key with actual API calls")
    print("4. Test provider utilities with real keys")
    print("5. Test customer service example end-to-end")

    # Check requirements first
    print("=" * 60)
    print("🔍 STEP 1: Checking requirements...")
    print("=" * 60)
    if not check_requirements():
        return False

    # Load environment variables
    print("\n" + "=" * 60)
    print("🔍 STEP 2: Loading environment variables...")
    print("=" * 60)
    env_vars = load_env_file()

    # Track test results
    results = []

    # Test 1: Environment loading
    print("\n" + "=" * 60)
    print("🔍 STEP 3: Testing environment loading...")
    print("=" * 60)
    results.append(("Environment Loading", test_environment_loading()))

    # Test 2: Provider utilities with real keys
    print("\n" + "=" * 60)
    print("🔍 STEP 4: Testing provider utilities...")
    print("=" * 60)
    results.append(("Provider Utilities", test_provider_utils_with_real_keys(env_vars)))

    # Test 3: Customer service example directly
    print("\n" + "=" * 60)
    print("🔍 STEP 5: Testing customer service example end-to-end...")
    print("=" * 60)
    results.append(
        ("Customer Service Example", test_customer_service_example_directly())
    )

    # Print summary
    print("\n" + "=" * 60)
    print("📊 FINAL TEST SUMMARY")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1

    print("=" * 60)
    print(f"Total: {total}, Passed: {passed}, Failed: {total - passed}")

    if passed == total:
        print("🎉 All tests passed! create.py and run.py work with real API keys.")
        print("✅ API key validation successful")
        print("✅ Provider utilities working correctly")
        print("✅ create.py working with --no-ui flag")
        print("✅ run.py working in interactive mode")
        print("✅ End-to-end customer service example working")
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

"""
Checker module for C++ task.
This module provides functions to check the syntax, compilability, and functional correctness of C++ code.

The checker first checks if the C++ code is syntactically correct using g++ with -fsyntax-only flag.
Then it compiles the code using g++ with C++17 standard and runs the compiled executable.
It checks both if the code compiles correctly (syntax_ok) and if it passes all the tests (passed_tests).
Tests are considered passed if the program runs without any assertion failures (exit code 0).
"""

import concurrent.futures
import subprocess
import os
from tempfile import NamedTemporaryFile

from datasets import load_dataset

from constrained_diffusion.cfgs.cpp import cpp_grammar_preprocessed
from constrained_diffusion.constrain_utils import (
    prelex_word,
    lex,
    reconstruct_word_boundaries,
)

GRAMMAR, LEXING = cpp_grammar_preprocessed()
DATASET = load_dataset("zai-org/humaneval-x", "cpp", split="test", trust_remote_code=True)


def cpp_syntax_check(cpp_program):
    """
    Check the syntax of a C++ program using our syntax definition
    """
    prelexed_program = prelex_word(cpp_program, "\x02\x03", is_first=True, is_last=True)
    lexed = lex(prelexed_program, LEXING, is_first=True)
    return any(
        GRAMMAR.accepts(lexied[0])
        for lexied in lexed
        if not lexied[1] and not lexied[2]
    )


def cpp_compile_and_run(cpp_program, timeout=40):
    """
    Compile and run a C++ program.

    Args:
        cpp_program: The C++ code to compile and run
        timeout: Maximum time in seconds to wait for compilation and execution

    Returns:
        A tuple (compile_success, run_success, output)
        - compile_success: True if compilation succeeded, False otherwise
        - run_success: True if execution succeeded and returned 0, False otherwise
        - output: Compiler/execution output or error message
    """
    # Create a temporary file for the C++ source code
    with NamedTemporaryFile(suffix=".cpp", delete=False) as source_file:
        source_path = source_file.name
        source_file.write(cpp_program.encode())
        source_file.flush()

    executable_path = source_path + ".exe"

    run_process = None
    try:
        # Compile the program with C++17 standard
        run_process = subprocess.Popen(
            ["g++", "-std=c++17", source_path, "-o", executable_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        run_process.wait(timeout=timeout)

        # Check if compilation was successful
        if run_process.returncode != 0:
            return False, False, run_process.stderr.read().decode()

        # Run the compiled program
        run_process = subprocess.Popen(
            [executable_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        run_process.wait(timeout=timeout)

        # Check if the program ran successfully (return code 0)
        # A return code of 0 means all assertions passed
        run_success = run_process.returncode == 0

        # Combine stdout and stderr for the output
        output = run_process.stdout.read().decode()
        if run_process.stderr:
            output += "\n" + run_process.stderr.read().decode()
        run_process = None

        return True, run_success, output

    except subprocess.TimeoutExpired:
        return False, False, "Timeout during compilation or execution"
    except Exception as e:
        return False, False, f"Error: {str(e)}"
    finally:
        # clean up the process
        if run_process:
            try:
                run_process.terminate()
            except subprocess.TimeoutExpired:
                try:
                    # Ensure the process is killed if it exceeds the timeout
                    run_process.kill()
                except subprocess.TimeoutExpired:
                    pass
                except Exception:
                    pass
            except Exception:
                pass
        # Clean up temporary files
        if os.path.exists(source_path):
            os.remove(source_path)
        if os.path.exists(executable_path):
            try:
                os.remove(executable_path)
            except OSError:
                pass  # Ignore errors when removing the executable


def check_instance(output, timeout=40):
    """
    Check a single instance from a JSONL file.

    Args:
        line: A JSON string containing an instance to check

    Returns:
        A dictionary with the check results
    """
    cpp_code = output["extracted"]
    if "\x02" in cpp_code:
        # If the code contains prelexed tokens, decode them
        output["extracted"] = reconstruct_word_boundaries(cpp_code)
        output["code"] = reconstruct_word_boundaries(output["code"])
        cpp_code = output["extracted"]

    if cpp_code.strip().startswith("/*"):
        declaration: str = DATASET[
            int(output["instance_id"].split("/")[1].split("_")[0])
        ]["declaration"]
        cpp_code_no_tests = declaration + output["code"]
    else:
        cpp_code_no_tests = cpp_code
    cpp_code_no_tests = cpp_code_no_tests.split("#undef NDEBUG")[0]

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # First, check if the code compiles (syntax-only check)
        syntax_ok_fut = executor.submit(cpp_syntax_check, cpp_code_no_tests)
        # compile and run the code to check functional correctness
        compile_fut = executor.submit(cpp_compile_and_run, cpp_code, timeout - 5)
        syntax_output_message = ""
        try:
            syntax_ok = syntax_ok_fut.result(timeout)
        except concurrent.futures.TimeoutError:
            syntax_ok = False
        except Exception as e:
            syntax_ok = False
            syntax_output_message = f"Syntax check failed: {str(e)}"
        try:
            _, run_success, output_message = compile_fut.result(timeout)
        except concurrent.futures.TimeoutError:
            _, run_success, output_message = (
                False,
                False,
                "Compilation or execution timed out",
            )
        except Exception as e:
            _, run_success, output_message = (
                False,
                False,
                f"Compilation or execution failed: {str(e)}",
            )

    return {
        "instance_id": output["instance_id"],
        "extracted": output["extracted"],
        "syntax_ok": syntax_ok,
        "passed_tests": run_success,  # True if the program compiled and ran with return code 0
        "compiler_output": syntax_output_message + output_message,
    }

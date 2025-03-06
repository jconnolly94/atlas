#!/usr/bin/env python
import unittest
import sys
import os
import argparse

# Add root directory to Python path for imports
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

def run_tests(test_type=None):
    """Discover and run tests based on the specified type.
    
    Args:
        test_type: Type of tests to run (unit, integration, system, performance, or None for all)
    """
    if test_type:
        print(f"Running {test_type} tests...")
        start_dir = os.path.join(os.path.dirname(__file__), test_type)
    else:
        print("Running all tests...")
        start_dir = os.path.dirname(__file__)
    
    # Discover tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(start_dir)
    
    # Run tests with verbosity=2 for detailed output
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Atlas traffic system tests")
    parser.add_argument("--type", choices=["unit", "integration", "system", "performance"], 
                        help="Type of tests to run (default: all)")
    args = parser.parse_args()
    
    result = run_tests(args.type)
    
    # Report overall statistics
    print("\n----- Test Results -----")
    print(f"Tests Run: {result.testsRun}")
    print(f"Errors: {len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    
    # Exit with non-zero code if tests failed
    sys.exit(len(result.errors) + len(result.failures))
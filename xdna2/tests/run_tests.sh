#!/bin/bash
# Run BF16 integration tests for Unicorn-Orator XDNA2

set -e

echo "=========================================="
echo "BF16 Workaround Integration Tests"
echo "=========================================="
echo ""

# Navigate to test directory
cd "$(dirname "$0")"

# Run tests with pytest
echo "Running pytest..."
python -m pytest test_bf16_integration.py -v -s

echo ""
echo "=========================================="
echo "All tests completed!"
echo "=========================================="

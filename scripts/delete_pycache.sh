#!/bin/bash

find . -type d -name "__pycache__" -exec rm -rf {} +
echo "Successfully deleted all __pycache__ folders."
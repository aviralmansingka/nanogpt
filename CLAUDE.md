# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository

## Commands

- Run locally: `python train_gpt2.py`
- Run with Modal: `modal run train_gpt2.py`
- Lint code: `ruff check .`
- Format code: `ruff format .`

## Code Guidelines

- Imports: Standard library first, then third-party libs, group related imports
- Indentation: 4 spaces
- Line length: 80 characters
- Types: Use type annotations, dataclasses for configuration
- Naming: PascalCase for classes, snake_case for functions/variables,
  UPPER_SNAKE_CASE for constants
- Documentation: Docstrings for classes and functions, inline comments for complex
  logic
- Error handling: Use assertions with explicit error messages
- Performance: Use torch.compile, kernel fusion, and autocast where appropriate


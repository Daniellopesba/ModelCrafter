# Makefile for linting, formatting, and testing Python code.

.PHONY: code-lint code-format code-test all smart

all: code-lint code-format code-test

# Lint the code using flake8 in an interactive manner.
code-lint:
	@echo "Linting the code..."
	@flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	@flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --ignore=E501,E712,W503 --statistics

# Preview and optionally format the code using the black code formatter.
code-format:
	@echo "Formatting the code..."
	@echo "Previewing changes..."
	@black . --exclude=venv --diff
	@read -p "Apply the above changes? [Y/n]: " choice; \
	if [ "$$choice" = "Y" ] || [ "$$choice" = "y" ] || [ -z "$$choice" ]; then \
		black . --exclude=venv; \
	else \
		echo "Changes were not applied."; \
	fi

code-test:
	@echo "Running tests..."
	@pytest

smart:
	@echo "Checking for modified files..."
	@MODIFIED_FILES=$$(git diff --name-only --cached); \
	if [ "$$MODIFIED_FILES" != "" ]; then \
		echo "Modified files: $$MODIFIED_FILES"; \
		flake8 $$MODIFIED_FILES --count --select=E9,F63,F7,F82 --show-source --statistics; \
		flake8 $$MODIFIED_FILES --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics; \
		echo "Previewing format changes for modified files..."; \
		if black $$MODIFIED_FILES --diff; then \
			read -p "Apply the above format changes to modified files? [Y/n]: " choice; \
			if [ "$$choice" = "Y" ] || [ "$$choice" = "y" ] || [ -z "$$choice" ]; then \
				black $$MODIFIED_FILES; \
				pytest; \
			else \
				echo "Format changes were not applied to modified files."; \
			fi; \
		else \
			echo "Formatting issues detected in modified files. Fix them before running tests."; \
		fi; \
	else \
		echo "No modified files found."; \
	fi
.PHONY: help create_env apt_deps run clean help

.DEFAULT_GOAL := help

VENV_DIR       := ./.venv
REQUIREMENTS   := requirements.txt
PYTHON         := python -u

create_env:
	@echo "Creating conda environment in $(VENV_DIR)..."
	conda create --prefix $(VENV_DIR) python=3.9 -y
	@echo "Activating environment and installing dependencies..."
	@bash -c "source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $(VENV_DIR) && pip install --upgrade pip && pip install -r $(REQUIREMENTS) && echo 'Done.'"

apt_deps:
	@echo "Installing system dependencies..."
	sudo apt-get update && sudo apt-get install -y cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-dri mesa-utils ccache
	@bash -c "conda install libpython-static -y"

run:
	@echo "Running the main script..."
	$(PYTHON) ./src/main.py

clean:
	@echo "Cleaning build files and caches..."
	rm -rf $(VENV_DIR) ./.pytest_cache
	find . -type f -name '*.py[co]' -delete
	find . -type d -name __pycache__ -delete
	@echo "Clean complete."

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

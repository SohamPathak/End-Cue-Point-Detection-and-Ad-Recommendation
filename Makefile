.PHONY: setup format lint test run docker-build docker-run clean all
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

$(VENV):
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

setup: $(VENV)  ## Create venv and install deps (dev)
	$(PIP) install -r requirements-dev.txt

format:  ## Auto-format
	$(VENV)/bin/black src app tests
	$(VENV)/bin/ruff check --fix src app tests

lint:  ## Static analysis
	$(VENV)/bin/ruff check src app tests
	$(VENV)/bin/mypy src

test:  ## Run tests with coverage
	$(VENV)/bin/pytest --cov=src/brahma --cov-report=term-missing

run:  ## Launch the Streamlit UI
	$(VENV)/bin/streamlit run app/streamlit_app.py

docker-build:  ## Build the container (bundles ffmpeg)
	docker build -t brahma-ai .

docker-run:  ## Run the container (mount creds + assets)
	docker run --rm -p 8501:8501 \
		-v $(PWD)/configs:/app/configs \
		-v $(PWD)/adv_video:/app/adv_video \
		-v $(PWD)/sample_video:/app/sample_video \
		-v $(PWD)/outputs:/app/outputs \
		brahma-ai

clean:  ## Remove venv and generated outputs
	rm -rf $(VENV) outputs/*.mp4 outputs/cache

all: format lint test  ## Format, lint, test

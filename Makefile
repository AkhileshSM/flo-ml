# 5-command inner loop from the product plan: up, train, ui, promote, down
COMPOSE ?= docker compose
PYTHON ?= python3
ENV ?= local
INSTANCE ?= local
STAGE ?= Staging
NAME ?= churn

.PHONY: up down train train-process ui console promote smoke test fixtures build-train score

fixtures:
	$(PYTHON) scripts/generate_fixtures.py

up:
	$(COMPOSE) up -d --build

down:
	$(COMPOSE) down

build-train:
	$(COMPOSE) --profile train build trainer

train: build-train
	$(PYTHON) jobs/estimator.py --env $(ENV) --instance $(INSTANCE)

train-process:
	$(PYTHON) jobs/estimator.py --env $(ENV) --instance process

ui:
	$(PYTHON) jobs/cli.py ui --open

console:
	$(PYTHON) -m uvicorn ui.app:app --host 0.0.0.0 --port 8088

promote:
	$(PYTHON) jobs/cli.py promote --name $(NAME) --stage $(STAGE) --env $(ENV)

score:
	$(PYTHON) -m client score --json client/examples/likely_churn.json

smoke:
	bash scripts/smoke.sh

test:
	$(PYTHON) -m pytest

ps:
	$(COMPOSE) ps

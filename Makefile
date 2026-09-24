PY := .venv/bin/python
CFG := config/plan_config.yaml

.PHONY: all ingest metrics match forecast plan audit report test
all: ingest match forecast plan audit report
ingest:
	$(PY) -m optimizer.ingest.run --config $(CFG)
match:
	$(PY) -m optimizer.match.run --config $(CFG)
forecast:
	$(PY) -m optimizer.forecast.run --config $(CFG)
plan:
	$(PY) -m optimizer.plan.run --config $(CFG)
audit:
	$(PY) -m audit.run --config $(CFG)
report:
	$(PY) -m optimizer.report.run --config $(CFG)
test:
	$(PY) -m pytest -q

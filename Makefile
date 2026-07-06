# Phase-E pipeline + Phase-F monitoring — orchestration only (no new logic here).
# unix / WSL / Git Bash:  make all      Windows PowerShell/cmd:  .\\make all   (uses make.bat)
PYTHON ?= python
CUTOFF ?= 2017-02-01

.PHONY: all features train export score serve monitor calibration smoke clean help

all: features train export score   ## full pipeline: features -> train+register -> export -> score

features:   ## build model_data from the phase-A cohorts (reproduces the notebook feature table)
	$(PYTHON) -m src.features

train:      ## fit base12 + isotonic, log to mlflow, register churn-base12@prod
	$(PYTHON) -m src.train

export:     ## materialise the prod model to api/model/ (serving has no registry dependency)
	$(PYTHON) -m src.export_model

score:      ## batch-score one cutoff month -> ranked contact list   (override: make score CUTOFF=2017-01-01)
	$(PYTHON) -m src.score $(CUTOFF)

serve:      ## run the on-demand scoring API (open http://127.0.0.1:8000/docs)
	$(PYTHON) -m uvicorn api.main:app --reload

monitor:    ## leading monitors at score time: tier-0 data-quality gate then tier-1 drift (override CUTOFF=)
	$(PYTHON) -m monitoring.data_quality $(CUTOFF)
	$(PYTHON) -m monitoring.score_drift $(CUTOFF)

calibration: ## lagging monitor: tier-2 predicted-vs-realized on the matured test cohort (the retrain trigger)
	$(PYTHON) -m monitoring.calibration

smoke:      ## run offline smoke tests (synthetic fixtures, no real data / registry)
	$(PYTHON) -m tests.smoke_test
	$(PYTHON) -m tests.smoke_train
	$(PYTHON) -m tests.smoke_score
	$(PYTHON) -m tests.smoke_full_scorer
	$(PYTHON) -m tests.smoke_api
	$(PYTHON) -m tests.smoke_data_quality
	$(PYTHON) -m tests.smoke_score_drift
	$(PYTHON) -m tests.smoke_calibration

clean:      ## remove generated artifacts (mlflow store, exported model, reports)
	rm -rf mlflow.db mlruns mlartifacts api/model reports

help:       ## list targets
	@grep -E "^[a-z]+:.*##" $(MAKEFILE_LIST) | sed -E "s/:.*## / -- /"

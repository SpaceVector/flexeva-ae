SHELL := /bin/bash
PYTHON ?= python3
PYTHON_BIN := $(if $(findstring /,$(PYTHON)),$(abspath $(PYTHON)),$(PYTHON))

.PHONY: build test

build:
	@cd flexmaya_ras && $(PYTHON_BIN) setup.py build_ext --inplace

test: build
	@cd flexmaya_ras && $(PYTHON_BIN) -m pytest -q
	@cd backends/maya && sha256sum --quiet -c SOURCE_ANALYSIS_FILES.sha256
	@cd backends/maya && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python \
		$(PYTHON_BIN) -m pytest -q -p no:cacheprovider \
		tests/python/source_analysis/test_source_analysis.py

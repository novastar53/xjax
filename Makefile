################################################################################
# Makefile
################################################################################

################################################################################
# Settings
################################################################################

# Verify environment.sh
ifeq ($(strip $(PROJECT_ROOT)),)
$(error Environment not configured. Run `source environment.sh`)
endif

#-------------------------------------------------------------------------------
# Shell
#-------------------------------------------------------------------------------

# Bash
SHELL := /bin/bash
.SHELLFLAGS := -e -u -o pipefail -c

# Colors - Supports colorized messages
COLOR_H1=\033[38;5;12m
COLOR_OK=\033[38;5;02m
COLOR_COMMENT=\033[38;5;08m
COLOR_RESET=\033[0m

# EXCLUDE_SRC - Source patterns to ignore

EXCLUDE_SRC := __pycache__ \
			   .egg-info
EXCLUDE_SRC := $(subst $(eval ) ,|,$(EXCLUDE_SRC))

#-------------------------------------------------------------------------------
# Commands
#-------------------------------------------------------------------------------

RM := rm -rf

#-------------------------------------------------------------------------------
# Output Dirs
#-------------------------------------------------------------------------------

BUILD_DIR := .build

#-------------------------------------------------------------------------------
# Environment
#-------------------------------------------------------------------------------

VENV_ROOT := .venv
VENV := $(VENV_ROOT)/bin/activate

#-------------------------------------------------------------------------------
# Requirements
#-------------------------------------------------------------------------------

REQUIREMENTS := requirements.txt

#-------------------------------------------------------------------------------
# Dependencies
#-------------------------------------------------------------------------------

DEPENDENCIES := $(BUILD_DIR)/deps.ts

#-------------------------------------------------------------------------------
# Packages
#-------------------------------------------------------------------------------

PACKAGES_DIR := $(BUILD_DIR)/packages
PACKAGES :=

# Package: xjax

XJAX_PACKAGE_SRC := $(shell find src -type f | egrep -v '$(EXCLUDE_SRC)')
XJAX_PACKAGE_REQUIRES = $(XJAX_PACKAGE_SRC)
XJAX_PACKAGE := $(PACKAGES_DIR)/xjax-$(PY_VERSION)-py3-none-any.whl

PACKAGES := $(PACKAGES) $(XJAX_PACKAGE)

#-------------------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------------------

PYTEST_OPTS ?=

#-------------------------------------------------------------------------------
# Linters
#-------------------------------------------------------------------------------

RUFF_CHECK_OPTS ?= --preview
RUFF_FORMAT_OPTS ?= --preview

#-------------------------------------------------------------------------------
# Phonies
#-------------------------------------------------------------------------------

PHONIES :=

################################################################################
# Targets
################################################################################

all: deps

#-------------------------------------------------------------------------------
# Output Dirs
#-------------------------------------------------------------------------------

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

#-------------------------------------------------------------------------------
# Environment
#-------------------------------------------------------------------------------

$(VENV):
	uv venv --python 3.11 --seed

venv: $(VENV)
PHONIES := $(PHONIES) venv


#-------------------------------------------------------------------------------
# Requirements
#-------------------------------------------------------------------------------

$(REQUIREMENTS): pyproject.toml | $(VENV)
	@echo
	@echo -e "$(COLOR_H1)# $@$(COLOR_RESET)"
	@echo

	source $(VENV) && uv pip compile --python-platform $(PLATFORM) -o $@ pyproject.toml

	@echo -e "$(COLOR_COMMENT)# Add Project$(COLOR_RESET)"
	echo "-e file://." >> $@
	@echo

requirements: $(REQUIREMENTS)
PHONIES := $(PHONIES) requirements


#-------------------------------------------------------------------------------
# Dependencies
#-------------------------------------------------------------------------------

$(DEPENDENCIES): $(REQUIREMENTS) | $(BUILD_DIR)
	source $(VENV) && uv pip sync --python-platform $(PLATFORM) $(REQUIREMENTS)
	@echo
	@echo -e "$(COLOR_COMMENT)# Activate venv: $(COLOR_OK)source $(VENV)$(COLOR_RESET)"
	@echo -e "$(COLOR_COMMENT)# Deactivate venv: $(COLOR_OK)deactivate$(COLOR_RESET)"
	@echo
	touch $@

deps: $(DEPENDENCIES)
PHONIES := $(PHONIES) deps


#-------------------------------------------------------------------------------
# Packages
#-------------------------------------------------------------------------------

$(PACKAGES_DIR):
	mkdir -p $@

# Package: xjax

$(XJAX_PACKAGE): $(XJAX_PACKAGE_REQUIRES) | $(PACKAGES_DIR) $(DEPENDENCIES)
	@echo
	@echo -e "$(COLOR_H1)# Package: $$(basename $@)$(COLOR_RESET)"
	@echo

	@echo -e "$(COLOR_COMMENT)# Build Package$(COLOR_RESET)"
	source $(VENV) && python -m build --outdir $(PROJECT_ROOT)/$$(dirname $@)
	@echo

packages: $(PACKAGES)

PHONIES := $(PHONIES) packages


#-------------------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------------------

tests: $(DEPENDENCIES)
	@echo
	@echo -e "$(COLOR_H1)# Tests$(COLOR_RESET)"
	@echo

	source $(VENV) && pytest $(PYTEST_OPTS) tests

coverage: $(DEPENDENCIES)
	@echo
	@echo -e "$(COLOR_H1)# Coverage$(COLOR_RESET)"
	@echo

	source $(VENV) && pytest $(PYTEST_OPTS) --cov=xjax --cov-report=html:$(BUILD_DIR)/coverage tests

PHONIES := $(PHONIES) tests coverage


#-------------------------------------------------------------------------------
# Linters
#-------------------------------------------------------------------------------

lint-fmt: deps
	source $(VENV) && \
	  ruff format $(RUFF_FORMAT_OPTS) && \
	  ruff check --fix $(RUFF_CHECK_OPTS) && \
	  nbqa isort notebooks/* && \
	  nbqa black notebooks/* && \
	  make lint-style

lint-style: deps
	source $(VENV) && \
	  ruff check $(RUFF_CHECK_OPTS) && \
	  ruff format --check $(RUFF_FORMAT_OPTS)

PHONIES := $(PHONIES) lint-fmt lint-style


#-------------------------------------------------------------------------------
# Clean
#-------------------------------------------------------------------------------

clean-venv:
	$(RM) $(VENV_ROOT)

clean-requirements:
	$(RM) $(REQUIREMENTS)

clean: clean-venv clean-requirements
PHONIES := $(PHONIES) clean-venv clean-requirements clean


.PHONIES: $(PHONIES)

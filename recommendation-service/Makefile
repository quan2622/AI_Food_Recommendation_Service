# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PYTHON    := ./venv/bin/python
PIP       := ./venv/bin/pip
UVICORN   := ./venv/bin/python -m uvicorn
# HOST      := 192.168.30.128
HOST      := 0.0.0.0
PORT      := 8081

# Colors for terminal output
YELLOW    := \033[1;33m
GREEN     := \033[1;32m
CYAN      := \033[1;36m
RED       := \033[1;31m
NC        := \033[0m # No Color

.PHONY: help setup run dev stop clean test

# -----------------------------------------------------------------------------
# Help System
# -----------------------------------------------------------------------------
help: ## Hien thi huong dan su dung
	@echo ""
	@echo "$(CYAN)======================================================================$(NC)"
	@echo "$(CYAN)             FOOD RECOMMENDATION SERVICE - MAKEFILE$(NC)"
	@echo "$(CYAN)======================================================================$(NC)"
	@echo "$(YELLOW)Danh sach cac lenh co san:$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""

# -----------------------------------------------------------------------------
# Core Commands
# -----------------------------------------------------------------------------
setup: ## [1] Khoi tao moi truong ao va cai dat thu vien
	@echo "$(CYAN)[PROGRESS] Dang tao moi truong ao (Python 3.11)...$(NC)"
	@python3 -m venv venv
	@echo "$(CYAN)[PROGRESS] Dang nang cap pip va cai dat dependencies...$(NC)"
	@$(PIP) install --upgrade pip
	@$(PIP) install -e .[dev]
	@echo "$(GREEN)[SUCCESS] Hoan tat! Ban co the su dung 'make dev' de bat dau.$(NC)"

run: stop ## [2] Chay server (Mode: Production)
	@echo "$(GREEN)[RUN] Dang khoi dong Server tai http://$(HOST):$(PORT) ...$(NC)"
	@$(UVICORN) app.main:app --host $(HOST) --port $(PORT)

dev: stop ## [3] Chay server (Mode: Development/Reload)
	@echo "$(YELLOW)[DEV] Dang khoi dong Server o che do Hot-Reload...$(NC)"
	@$(UVICORN) app.main:app --host $(HOST) --port $(PORT) --reload

test: stop ## [4] Chay Unit Tests cho he thong
	@echo "$(CYAN)[TEST] Dang chay thu nghiem (Pytest)...$(NC)"
	@$(PYTHON) -m pytest

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
stop: ## Giai phong cong $(PORT) khoi cac tien trinh dang running
	@echo "$(RED)[STOP] Dang kiem tra va giai phong cong $(PORT)...$(NC)"
	@fuser -k $(PORT)/tcp || true

clean: ## Don dep moi truong ao, cache va cac file rac
	@echo "$(RED)[CLEAN] Dang don dep thu muc du an...$(NC)"
	@rm -rf venv
	@rm -rf .pytest_cache
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name "*.egg-info" -exec rm -rf {} +
	@echo "$(GREEN)[SUCCESS] Da don dep xong!$(NC)"

.PHONY: help venv install install-python install-react \
        train start-api start-ui start \
        stop clean reset

# ── config ────────────────────────────────────────────────────────────────────
API_DIR     := nlu
UI_DIR      := ui
VENV        := $(API_DIR)/.venv
PYTHON      := $(VENV)/bin/python
PIP         := $(VENV)/bin/pip
API_PORT    := 5001
UI_PORT     := 5173
API_PID     := .api.pid

# ── default target ────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Intentiq — Intent Understanding Engine"
	@echo "  ─────────────────────────────────────────────"
	@echo "  make install        Create venv + install all dependencies"
	@echo "  make install-python Create venv + install Python dependencies"
	@echo "  make install-react  Install Node dependencies only"
	@echo ""
	@echo "  make train          Train the NLU model"
	@echo "  make start-api      Start Flask API  (port $(API_PORT))"
	@echo "  make start-ui       Start React UI   (port $(UI_PORT))"
	@echo "  make start          Start both (API in background, UI in foreground)"
	@echo ""
	@echo "  make stop           Stop background API process"
	@echo "  make clean          Remove build artifacts and caches"
	@echo "  make reset          clean + delete venv + trained model"
	@echo ""

# ── virtual environment ───────────────────────────────────────────────────────
venv:
	@if [ ! -d "$(VENV)" ]; then \
		echo "→ Creating virtual environment at $(VENV)…"; \
		python3 -m venv $(VENV); \
	else \
		echo "→ Virtual environment already exists."; \
	fi

# ── install ───────────────────────────────────────────────────────────────────
install: install-python install-react

install-python: venv
	@echo "→ Installing Python dependencies into venv…"
	$(PIP) install --upgrade pip -q
	$(PIP) install -r $(API_DIR)/requirements.txt

install-react:
	@echo "→ Installing Node dependencies…"
	cd $(UI_DIR) && yarn install

# ── model ─────────────────────────────────────────────────────────────────────
train: venv
	@echo "→ Training NLU model on all data…"
	cd $(API_DIR) && ../$(PYTHON) train.py
 
train-eval: venv
	@echo "→ Training NLU model + accuracy report on held-out split…"
	cd $(API_DIR) && ../$(PYTHON) train.py --eval

# ── run ───────────────────────────────────────────────────────────────────────
start-api: venv
	@echo "→ Starting Flask API on port $(API_PORT)…"
	cd $(API_DIR) && ../$(PYTHON) app.py

start-ui:
	@echo "→ Starting React UI on port $(UI_PORT)…"
	cd $(UI_DIR) && npm run dev

# Start API in background, then UI in foreground (Ctrl+C stops UI; run make stop for API)
start: venv
	@echo "→ Starting Flask API in background…"
	cd $(API_DIR) && ../$(PYTHON) app.py & echo $$! > ../$(API_PID)
	@sleep 1
	@echo "→ Starting React UI (Ctrl+C to stop)…"
	cd $(UI_DIR) && npm run dev

# ── stop ──────────────────────────────────────────────────────────────────────
stop:
	@if [ -f $(API_PID) ]; then \
		kill $$(cat $(API_PID)) 2>/dev/null && echo "→ Flask API stopped" || echo "→ Process already gone"; \
		rm -f $(API_PID); \
	else \
		echo "→ No PID file found ($(API_PID))"; \
	fi

# ── clean ─────────────────────────────────────────────────────────────────────
clean:
	@echo "→ Cleaning build artifacts…"
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc"     -delete 2>/dev/null || true
	rm -rf $(UI_DIR)/dist $(UI_DIR)/.vite
	@echo "   done."

reset: clean
	@echo "→ Removing venv and trained model…"
	rm -rf $(VENV)
	rm -f $(API_DIR)/models/*.pkl
	@echo "   Run 'make install && make train' to start fresh."
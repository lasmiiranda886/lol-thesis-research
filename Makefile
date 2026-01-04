.PHONY: setup train evaluate figures clean help

# Default target
help:
	@echo "LoL Win Prediction - Available commands:"
	@echo ""
	@echo "  make setup      - Install dependencies"
	@echo "  make train      - Train final models (requires data/final/)"
	@echo "  make evaluate   - Evaluate models and generate metrics"
	@echo "  make figures    - Generate thesis figures"
	@echo "  make all        - Run full pipeline (train + evaluate + figures)"
	@echo ""
	@echo "Data collection (requires Riot API key in .env):"
	@echo "  make collect    - Collect match data from Riot API"

# Setup
setup:
	pip install -r requirements.txt

# Train models
train:
	python src/nn_siamese_final.py
	python src/train_final_models.py

# Evaluate
evaluate:
	python src/ensemble_with_siamese.py

# Generate figures
figures:
	python src/generate_thesis_figures.py
	python src/generate_champion_variability_charts.py

# Full pipeline
all: train evaluate figures

# Data collection (requires API key)
collect:
	python src/collect_match_ids_global.py
	python src/collector.py

# Clean generated files
clean:
	rm -rf __pycache__ src/__pycache__
	rm -rf catboost_info logs

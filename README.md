# League of Legends Win Prediction from Champion Select

**Bachelor Thesis Project** - Predicting match outcomes in League of Legends using pre-game information from a single player's perspective.

## Abstract

This project investigates how accurately the outcome of a League of Legends match can be predicted during champion select, when only the information available to a single player is used. Unlike previous research that assumed access to all ten players' statistics, this work simulates the realistic constraints of the ranked queue where players only know their own stats and the champion selections.

**Key Results:**
- **AUC: 0.58** | **Accuracy: 55%** (vs. 50% random baseline)
- The most important feature is a learned team composition score from a Siamese Neural Network
- Handcrafted team composition features (Engage, Tank, Scaling) do not contribute to prediction
- Prediction accuracy decreases with higher ranks (Silver: 0.59 AUC, Diamond: 0.55 AUC)

## Research Questions

1. **FF1:** How accurately can match outcomes be predicted from a single player's perspective with limited information?
2. **FF2:** Which machine learning methods are best suited for this constrained prediction task?
3. **FF3:** Which features are most relevant when teammate rank information is unavailable?

## Data

- **Source:** Riot Games API (Match-V5, League-V4, Champion-Mastery-V4)
- **Scope:** 320,000+ Ranked Solo/Duo matches from 12 regions
- **Period:** January - December 2025 (Patches 15.x)
- **Datasets:**
  - Random Hero Dataset: 256,453 training / 63,927 test matches
  - Main-Hero Dataset: 63,709 training / 17,738 test matches (specialized players)

*Note: Raw data is not included due to size (~54GB). See data collection scripts to reproduce.*

## Model

**Weighted Average Ensemble:**
- LightGBM (40%)
- XGBoost (30%)
- ExtraTrees (30%)

**Final Feature Set (21 features):**
- Hero Player Stats (11): rank, LP, winrate, mastery, etc.
- Champion Stats (7): champion winrates by elo/role
- Special Features (3): Siamese Network score, expected WR, smurf score

## Repository Structure

```
src/
├── collector.py                    # Main data collection from Riot API
├── collector_optimized.py          # Optimized collection with rate limiting
├── collect_match_ids_global.py     # Match ID collection across regions
├── build_*.py                      # Dataset building scripts
├── champion_stats_features.py      # Champion statistics computation
├── nn_siamese_final.py             # Siamese Neural Network for team composition
├── ensemble_model.py               # Base ensemble model
├── ensemble_with_siamese.py        # Final ensemble with Siamese integration
├── train_model.py                  # Model training
├── train_final_models.py           # Final model training
├── feature_ablation_overnight.py   # Feature ablation study (31,260 combinations)
├── generate_thesis_figures.py      # Figure generation for thesis
└── generate_champion_variability_charts.py  # Champion analysis charts

data/
├── final/                          # Final datasets (included in repo)
│   ├── hero_dataset_random_train_final.parquet (33 MB, 256k matches)
│   ├── hero_dataset_random_test_final.parquet (8.8 MB, 64k matches)
│   ├── hero_dataset_main_train_final.parquet (8.2 MB, 64k matches)
│   └── hero_dataset_main_test_final.parquet (2.4 MB, 18k matches)
└── static/                         # Champion mappings (included in repo)
    ├── champion_categories.json
    └── champion_id_to_name.json
```

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- Python 3.11+
- pandas, numpy, pyarrow
- scikit-learn, LightGBM, XGBoost
- PyTorch (for Siamese Network)
- matplotlib

## Reproducing Results

1. **Data Collection** (requires Riot API key):
   ```bash
   python src/collect_match_ids_global.py
   python src/collector.py
   ```

2. **Feature Engineering:**
   ```bash
   python src/build_random_hero_dataset.py
   python src/build_main_hero_dataset.py
   python src/champion_stats_features.py
   ```

3. **Train Siamese Network:**
   ```bash
   python src/nn_siamese_final.py
   ```

4. **Train Final Models:**
   ```bash
   python src/train_final_models.py
   ```

## Citation

If you use this code or methodology, please cite:

```
Aziri, A. (2025). Vorhersage der Gewinnwahrscheinlichkeit von League-of-Legends-Spielen
während der Championauswahl mittels Machine Learning. Bachelor Thesis, BFH.
```

## License

This project was created for academic purposes as part of a Bachelor thesis at BFH.

## Acknowledgments

- Riot Games for providing the API access
- Referenced studies: Costa et al. (2021), Do et al. (2021), Hitar-García et al. (2022), Junior et al. (2023)

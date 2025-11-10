# MODERN Diet for Dementia Prevention - MVP Implementation

## Overview

This is a Minimum Viable Product (MVP) implementation to reproduce the key findings from the paper:
**"Machine learning-assisted optimization of dietary intervention against dementia risk"**
Published in Nature Human Behaviour (2025)

## Paper Summary

The study developed the MODERN diet (Machine learning-assisted Optimizing Dietary intERvention against demeNtia risk) based on:
- 185,012 UK Biobank participants
- 1,987 dementia cases over 10 years of follow-up
- Machine learning approach (LightGBM) for feature selection

## MODERN Diet Components

The MODERN diet consists of 7 components with a total score of 0-7:

### Adequacy (Higher intake recommended)
1. **Olive oil**: >0 servings/day

### Moderation (Moderate intake recommended)
2. **Green leafy vegetables**: 0-1.5 servings/day
3. **Berries and citrus fruits**: 0-2 servings/day
4. **Potatoes**: 0-0.75 servings/day
5. **Eggs**: 0-1 servings/day
6. **Poultry**: 0-0.5 servings/day

### Restriction (Limited intake recommended)
7. **Sweetened beverages**: 0 servings/day

## Key Findings

- **MODERN diet** showed stronger association with lower dementia risk (HR: 0.64, 95% CI: 0.43-0.93) compared to
- **MIND diet** (HR: 0.75, 95% CI: 0.61-0.92)

## MVP Implementation

This MVP includes:

1. **Data preprocessing**: Handling food group data
2. **Cox regression**: Food-wide association analysis
3. **Machine learning**: LightGBM feature selection
4. **Score calculation**: MODERN and MIND diet scores
5. **Visualization**: Key results and comparisons

## Project Structure

```
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── src/
│   ├── data_preprocessing.py # Data cleaning and preparation
│   ├── cox_regression.py     # Cox proportional hazards models
│   ├── ml_feature_selection.py # LightGBM feature importance
│   ├── diet_scores.py        # MODERN and MIND score calculation
│   └── visualization.py      # Plotting functions
├── notebooks/
│   └── analysis_workflow.ipynb # Step-by-step analysis
├── data/
│   ├── simulated/            # Simulated data for demo
│   └── processed/            # Processed datasets
└── results/
    ├── figures/              # Generated plots
    └── tables/               # Statistical results
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the complete analysis pipeline
python src/main.py

# Or use the Jupyter notebook
jupyter notebook notebooks/analysis_workflow.ipynb
```

## Dependencies

- Python 3.9+
- pandas
- numpy
- scikit-learn
- lifelines (for Cox regression)
- lightgbm
- matplotlib
- seaborn
- scipy

## Citation

If you use this code, please cite the original paper:

```
Chen, S.J., Chen, H., You, J. et al.
Machine learning-assisted optimization of dietary intervention against dementia risk.
Nat Hum Behav (2025).
https://doi.org/10.1038/s41562-025-02255-w
```

## License

This implementation is for educational and research purposes.

## Authors

MVP Implementation by Claude Code
Original Paper by Chen et al. (2025)

## Disclaimer

This is a simplified implementation for educational purposes. For clinical applications,
please refer to the original paper and consult with healthcare professionals.

"""
Cox Proportional Hazards Analysis

Perform survival analysis for diet-dementia associations.

Author: MVP Implementation
Date: 2025
"""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
import warnings
warnings.filterwarnings('ignore')


class CoxAnalysis:
    """Cox proportional hazards analysis for diet and dementia"""

    def __init__(self, data: pd.DataFrame):
        """
        Initialize Cox analysis

        Parameters:
        -----------
        data : DataFrame
            Dataset with exposure, outcome, and covariate data
        """
        self.data = data
        self.models = {}

    def prepare_data(self, score_column: str,
                    outcome_col: str = 'dementia_event',
                    time_col: str = 'time_to_event') -> pd.DataFrame:
        """
        Prepare data for Cox regression

        Parameters:
        -----------
        score_column : str
            Name of the diet score column
        outcome_col : str
            Name of the event column
        time_col : str
            Name of the time column

        Returns:
        --------
        DataFrame ready for Cox regression
        """
        required_cols = [score_column, outcome_col, time_col]
        return self.data[required_cols + ['age', 'sex', 'bmi',
                                         'education_years', 'smoking_status',
                                         'apoe4_carrier']].copy()

    def fit_model(self, score_column: str,
                 outcome_col: str = 'dementia_event',
                 time_col: str = 'time_to_event',
                 adjust_covariates: bool = True) -> CoxPHFitter:
        """
        Fit Cox proportional hazards model

        Parameters:
        -----------
        score_column : str
            Name of the diet score column
        outcome_col : str
            Name of the outcome column
        time_col : str
            Name of the time column
        adjust_covariates : bool
            Whether to adjust for covariates

        Returns:
        --------
        CoxPHFitter : Fitted Cox model
        """
        # Prepare analysis dataset
        analysis_data = self.data[[score_column, outcome_col, time_col,
                                  'age', 'sex', 'bmi', 'education_years',
                                  'smoking_status', 'apoe4_carrier']].copy()

        # Convert categorical variables
        analysis_data = pd.get_dummies(analysis_data,
                                      columns=['sex', 'smoking_status'],
                                      drop_first=True)

        # Define formula
        if adjust_covariates:
            covariates = [score_column, 'age', 'bmi', 'education_years',
                        'apoe4_carrier'] + \
                       [col for col in analysis_data.columns
                        if col.startswith('sex_') or col.startswith('smoking_status_')]
        else:
            covariates = [score_column]

        # Fit model
        cph = CoxPHFitter()
        cph.fit(analysis_data[covariates + [time_col, outcome_col]],
               duration_col=time_col,
               event_col=outcome_col)

        self.models[score_column] = cph
        return cph

    def get_hazard_ratio(self, model: CoxPHFitter,
                        variable: str) -> dict:
        """
        Extract hazard ratio and confidence intervals

        Parameters:
        -----------
        model : CoxPHFitter
            Fitted Cox model
        variable : str
            Variable name

        Returns:
        --------
        dict : HR, CI, and p-value
        """
        summary = model.summary

        if variable in summary.index:
            coef = summary.loc[variable, 'coef']
            se = summary.loc[variable, 'se(coef)']
            p_value = summary.loc[variable, 'p']

            hr = np.exp(coef)
            ci_lower = np.exp(coef - 1.96 * se)
            ci_upper = np.exp(coef + 1.96 * se)

            return {
                'variable': variable,
                'HR': hr,
                'CI_lower': ci_lower,
                'CI_upper': ci_upper,
                'p_value': p_value
            }
        return None

    def compare_tertiles(self, score_column: str,
                        outcome_col: str = 'dementia_event',
                        time_col: str = 'time_to_event') -> pd.DataFrame:
        """
        Compare risk by tertiles of diet score

        Parameters:
        -----------
        score_column : str
            Name of the diet score column

        Returns:
        --------
        DataFrame with tertile comparison results
        """
        # Create tertiles
        data_tertiles = self.data.copy()
        data_tertiles['tertile'] = pd.qcut(data_tertiles[score_column],
                                          q=3, labels=['T1', 'T2', 'T3'])

        # Prepare for Cox regression
        analysis_data = data_tertiles[['tertile', outcome_col, time_col,
                                      'age', 'sex', 'bmi', 'education_years',
                                      'smoking_status', 'apoe4_carrier']].copy()

        analysis_data = pd.get_dummies(analysis_data,
                                      columns=['sex', 'smoking_status', 'tertile'],
                                      drop_first=True)

        # Fit model
        cph = CoxPHFitter()
        cph.fit(analysis_data,
               duration_col=time_col,
               event_col=outcome_col)

        # Extract tertile HRs
        results = []
        for tertile in ['tertile_T2', 'tertile_T3']:
            hr_dict = self.get_hazard_ratio(cph, tertile)
            if hr_dict:
                results.append(hr_dict)

        return pd.DataFrame(results)

    def generate_summary_table(self, models_dict: dict) -> pd.DataFrame:
        """
        Generate summary table for multiple models

        Parameters:
        -----------
        models_dict : dict
            Dictionary of {score_name: model}

        Returns:
        --------
        DataFrame with comparison of all models
        """
        results = []

        for score_name, model in models_dict.items():
            hr_dict = self.get_hazard_ratio(model, score_name)
            if hr_dict:
                hr_dict['diet_score'] = score_name
                results.append(hr_dict)

        return pd.DataFrame(results)


def run_cox_analysis(data: pd.DataFrame) -> dict:
    """
    Run complete Cox analysis for MODERN and MIND scores

    Parameters:
    -----------
    data : DataFrame
        Dataset with diet scores and outcomes

    Returns:
    --------
    dict : Analysis results
    """
    print("\n" + "="*60)
    print("COX PROPORTIONAL HAZARDS ANALYSIS")
    print("="*60)

    analyzer = CoxAnalysis(data)
    results = {}

    # Analyze MODERN score
    if 'modern_total_score' in data.columns:
        print("\n1. MODERN Diet Score Analysis")
        print("-" * 60)

        modern_model = analyzer.fit_model('modern_total_score')
        modern_hr = analyzer.get_hazard_ratio(modern_model, 'modern_total_score')

        print(f"\nHazard Ratio per 1-point increase:")
        print(f"  HR: {modern_hr['HR']:.3f}")
        print(f"  95% CI: ({modern_hr['CI_lower']:.3f}, {modern_hr['CI_upper']:.3f})")
        print(f"  P-value: {modern_hr['p_value']:.4f}")

        # Tertile analysis
        modern_tertiles = analyzer.compare_tertiles('modern_total_score')
        print("\nTertile Analysis (T1 as reference):")
        print(modern_tertiles.to_string(index=False))

        results['modern'] = {
            'model': modern_model,
            'hr': modern_hr,
            'tertiles': modern_tertiles
        }

    # Analyze MIND score
    if 'mind_total_score' in data.columns:
        print("\n2. MIND Diet Score Analysis")
        print("-" * 60)

        mind_model = analyzer.fit_model('mind_total_score')
        mind_hr = analyzer.get_hazard_ratio(mind_model, 'mind_total_score')

        print(f"\nHazard Ratio per 1-point increase:")
        print(f"  HR: {mind_hr['HR']:.3f}")
        print(f"  95% CI: ({mind_hr['CI_lower']:.3f}, {mind_hr['CI_upper']:.3f})")
        print(f"  P-value: {mind_hr['p_value']:.4f}")

        # Tertile analysis
        mind_tertiles = analyzer.compare_tertiles('mind_total_score')
        print("\nTertile Analysis (T1 as reference):")
        print(mind_tertiles.to_string(index=False))

        results['mind'] = {
            'model': mind_model,
            'hr': mind_hr,
            'tertiles': mind_tertiles
        }

    # Comparison summary
    if 'modern' in results and 'mind' in results:
        print("\n3. MODERN vs MIND Comparison")
        print("-" * 60)
        print(f"MODERN HR per point: {results['modern']['hr']['HR']:.3f}")
        print(f"MIND HR per point: {results['mind']['hr']['HR']:.3f}")

    return results


if __name__ == "__main__":
    # Example usage
    print("This module provides Cox regression analysis functions.")
    print("Import and use with: from cox_analysis import run_cox_analysis")

"""
Simulated Data Generator

Generate synthetic data resembling UK Biobank structure for testing purposes.

Author: MVP Implementation
Date: 2025
"""

import numpy as np
import pandas as pd
from typing import Tuple


class DataGenerator:
    """Generate simulated dietary and dementia data"""

    def __init__(self, n_participants: int = 10000, random_state: int = 42):
        """
        Initialize data generator

        Parameters:
        -----------
        n_participants : int
            Number of participants to simulate
        random_state : int
            Random seed for reproducibility
        """
        self.n_participants = n_participants
        self.random_state = random_state
        np.random.seed(random_state)

    def generate_demographics(self) -> pd.DataFrame:
        """
        Generate demographic data

        Returns:
        --------
        DataFrame with demographic variables
        """
        data = {
            'participant_id': np.arange(self.n_participants),
            'age': np.random.normal(59.2, 7.97, self.n_participants).clip(40, 75),
            'sex': np.random.choice(['Male', 'Female'], self.n_participants, p=[0.447, 0.553]),
            'education_years': np.random.choice([10, 12, 14, 16, 18, 20], self.n_participants),
            'bmi': np.random.normal(27.4, 4.8, self.n_participants).clip(15, 50),
            'physical_activity': np.random.lognormal(6.5, 1.2, self.n_participants),
            'smoking_status': np.random.choice(['Never', 'Former', 'Current'],
                                             self.n_participants, p=[0.55, 0.35, 0.10]),
            'apoe4_carrier': np.random.choice([0, 1], self.n_participants, p=[0.75, 0.25])
        }
        return pd.DataFrame(data)

    def generate_food_intake(self) -> pd.DataFrame:
        """
        Generate food intake data (servings per day)

        Returns:
        --------
        DataFrame with food group intake levels
        """
        # Generate food intake with realistic distributions
        np.random.seed(self.random_state)

        data = {
            'participant_id': np.arange(self.n_participants),

            # MODERN components
            'olive_oil': np.random.exponential(0.3, self.n_participants).clip(0, 5),
            'green_leafy_vegetables': np.random.lognormal(0, 0.8, self.n_participants).clip(0, 5),
            'berries': np.random.exponential(0.2, self.n_participants).clip(0, 3),
            'citrus_fruits': np.random.exponential(0.3, self.n_participants).clip(0, 3),
            'potatoes': np.random.lognormal(0, 0.6, self.n_participants).clip(0, 3),
            'eggs': np.random.exponential(0.5, self.n_participants).clip(0, 3),
            'poultry': np.random.exponential(0.3, self.n_participants).clip(0, 2),
            'sweetened_beverages': np.random.exponential(0.4, self.n_participants).clip(0, 5),

            # Additional MIND components
            'other_vegetables': np.random.lognormal(0.5, 0.7, self.n_participants).clip(0, 5),
            'nuts': np.random.exponential(0.3, self.n_participants).clip(0, 3),
            'whole_grains': np.random.lognormal(0.8, 0.6, self.n_participants).clip(0, 5),
            'fish': np.random.exponential(0.2, self.n_participants).clip(0, 2),
            'beans': np.random.exponential(0.3, self.n_participants).clip(0, 3),
            'wine': np.random.exponential(0.3, self.n_participants).clip(0, 3),
            'red_meat': np.random.exponential(0.4, self.n_participants).clip(0, 3),
            'butter': np.random.exponential(0.5, self.n_participants).clip(0, 3),
            'cheese': np.random.exponential(0.3, self.n_participants).clip(0, 2),
            'pastries': np.random.exponential(0.4, self.n_participants).clip(0, 3),
            'fried_food': np.random.exponential(0.2, self.n_participants).clip(0, 2),
        }

        df = pd.DataFrame(data)

        # Combine berries and citrus for MODERN score
        df['berries_citrus'] = df['berries'] + df['citrus_fruits']

        return df

    def generate_outcomes(self, food_df: pd.DataFrame,
                         demographics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate dementia outcomes based on diet and demographics

        Parameters:
        -----------
        food_df : DataFrame
            Food intake data
        demographics_df : DataFrame
            Demographic data

        Returns:
        --------
        DataFrame with outcome variables
        """
        np.random.seed(self.random_state)

        # Calculate diet scores (simplified)
        protective_foods = (
            food_df['olive_oil'] * 0.5 +
            food_df['green_leafy_vegetables'] * 0.8 +
            food_df['berries_citrus'] * 0.7 +
            food_df['fish'] * 0.6
        )

        harmful_foods = (
            food_df['sweetened_beverages'] * 0.8 +
            food_df['red_meat'] * 0.5 +
            food_df['fried_food'] * 0.7
        )

        # Risk score based on diet and demographics
        age_effect = (demographics_df['age'] - 50) * 0.08
        apoe4_effect = demographics_df['apoe4_carrier'] * 0.7

        baseline_hazard = -4.5  # Corresponds to ~1% incidence
        risk_score = (
            baseline_hazard +
            age_effect +
            apoe4_effect -
            protective_foods * 0.15 +
            harmful_foods * 0.12 +
            np.random.normal(0, 0.5, self.n_participants)
        )

        # Generate dementia events
        dementia_probability = 1 / (1 + np.exp(-risk_score))
        dementia_event = np.random.binomial(1, dementia_probability)

        # Generate time to event (years)
        follow_up_time = np.random.uniform(0.5, 12, self.n_participants)

        # For non-events, they survived the full follow-up
        event_time = np.where(dementia_event == 1,
                            follow_up_time * np.random.uniform(0.3, 1, self.n_participants),
                            follow_up_time)

        outcome_data = {
            'participant_id': np.arange(self.n_participants),
            'dementia_event': dementia_event,
            'time_to_event': event_time,
            'follow_up_years': follow_up_time
        }

        return pd.DataFrame(outcome_data)

    def generate_full_dataset(self) -> pd.DataFrame:
        """
        Generate complete dataset with all variables

        Returns:
        --------
        DataFrame with demographics, food intake, and outcomes
        """
        print(f"Generating data for {self.n_participants} participants...")

        # Generate components
        demographics = self.generate_demographics()
        food_intake = self.generate_food_intake()
        outcomes = self.generate_outcomes(food_intake, demographics)

        # Merge all data
        full_data = demographics.merge(food_intake, on='participant_id')
        full_data = full_data.merge(outcomes, on='participant_id')

        print(f"✓ Generated {len(full_data)} complete records")
        print(f"✓ Dementia cases: {full_data['dementia_event'].sum()} "
              f"({full_data['dementia_event'].mean()*100:.2f}%)")

        return full_data


def save_simulated_data(output_path: str, n_participants: int = 10000):
    """
    Generate and save simulated dataset

    Parameters:
    -----------
    output_path : str
        Path to save the CSV file
    n_participants : int
        Number of participants to generate
    """
    generator = DataGenerator(n_participants=n_participants)
    data = generator.generate_full_dataset()
    data.to_csv(output_path, index=False)
    print(f"\n✓ Data saved to: {output_path}")
    return data


if __name__ == "__main__":
    # Generate and save sample data
    output_file = "../data/simulated/ukb_simulated_data.csv"
    data = save_simulated_data(output_file, n_participants=10000)

    # Display summary statistics
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)

    print("\nDemographics:")
    print(f"  Age: {data['age'].mean():.1f} ± {data['age'].std():.1f} years")
    print(f"  Female: {(data['sex']=='Female').mean()*100:.1f}%")
    print(f"  BMI: {data['bmi'].mean():.1f} ± {data['bmi'].std():.1f}")

    print("\nKey food groups (servings/day):")
    food_cols = ['olive_oil', 'green_leafy_vegetables', 'berries_citrus',
                 'sweetened_beverages', 'fish']
    for col in food_cols:
        print(f"  {col}: {data[col].mean():.2f} ± {data[col].std():.2f}")

    print("\nOutcomes:")
    print(f"  Total dementia cases: {data['dementia_event'].sum()}")
    print(f"  Incidence rate: {data['dementia_event'].mean()*100:.2f}%")
    print(f"  Mean follow-up: {data['follow_up_years'].mean():.1f} years")

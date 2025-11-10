"""
Diet Score Calculation Module

This module implements the MODERN and MIND diet scoring systems
for dementia risk assessment.

Author: MVP Implementation
Date: 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class MODERNScore:
    """
    Calculate MODERN diet score (0-7 points)

    Components:
    1. Olive oil (Adequacy): >0 servings/day = 1 point
    2. Green leafy vegetables (Moderation): 0-1.5 servings/day = 1 point
    3. Berries and citrus fruits (Moderation): 0-2 servings/day = 1 point
    4. Potatoes (Moderation): 0-0.75 servings/day = 1 point
    5. Eggs (Moderation): 0-1 servings/day = 1 point
    6. Poultry (Moderation): 0-0.5 servings/day = 1 point
    7. Sweetened beverages (Restriction): 0 servings/day = 1 point
    """

    def __init__(self):
        """Initialize MODERN score calculator with component rules"""
        self.components = {
            'olive_oil': {'type': 'adequacy', 'optimal': (0.001, np.inf)},
            'green_leafy_vegetables': {'type': 'moderation', 'optimal': (0, 1.5)},
            'berries_citrus': {'type': 'moderation', 'optimal': (0, 2.0)},
            'potatoes': {'type': 'moderation', 'optimal': (0, 0.75)},
            'eggs': {'type': 'moderation', 'optimal': (0, 1.0)},
            'poultry': {'type': 'moderation', 'optimal': (0, 0.5)},
            'sweetened_beverages': {'type': 'restriction', 'optimal': (0, 0)}
        }

    def score_component(self, value: float, component: str) -> float:
        """
        Score a single component based on intake level

        Parameters:
        -----------
        value : float
            Daily servings of the food component
        component : str
            Name of the food component

        Returns:
        --------
        float : Score for this component (0 or 1)
        """
        if component not in self.components:
            raise ValueError(f"Unknown component: {component}")

        comp_info = self.components[component]
        comp_type = comp_info['type']
        optimal_range = comp_info['optimal']

        if comp_type == 'adequacy':
            # Higher intake is better
            return 1.0 if value > optimal_range[0] else 0.0

        elif comp_type == 'moderation':
            # Within optimal range is better
            return 1.0 if optimal_range[0] <= value <= optimal_range[1] else 0.0

        elif comp_type == 'restriction':
            # Zero or minimal intake is better
            return 1.0 if value == optimal_range[0] else 0.0

        return 0.0

    def calculate_total_score(self, food_data: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """
        Calculate total MODERN score

        Parameters:
        -----------
        food_data : dict
            Dictionary with food component names as keys and daily servings as values

        Returns:
        --------
        tuple : (total_score, component_scores)
            total_score: Total MODERN score (0-7)
            component_scores: Dictionary of individual component scores
        """
        component_scores = {}

        for component in self.components.keys():
            if component in food_data:
                component_scores[component] = self.score_component(
                    food_data[component], component
                )
            else:
                component_scores[component] = 0.0

        total_score = sum(component_scores.values())
        return total_score, component_scores

    def calculate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MODERN scores for a dataframe

        Parameters:
        -----------
        df : DataFrame
            DataFrame with columns matching component names

        Returns:
        --------
        DataFrame : Original dataframe with added MODERN score columns
        """
        result_df = df.copy()

        # Calculate component scores
        for component in self.components.keys():
            if component in df.columns:
                result_df[f'modern_{component}_score'] = df[component].apply(
                    lambda x: self.score_component(x, component)
                )

        # Calculate total score
        score_columns = [f'modern_{comp}_score' for comp in self.components.keys()]
        result_df['modern_total_score'] = result_df[score_columns].sum(axis=1)

        return result_df


class MINDScore:
    """
    Calculate MIND diet score (0-15 points)

    The MIND diet (Mediterranean-DASH Intervention for Neurodegenerative Delay)
    includes 10 brain-healthy food groups and 5 unhealthy food groups.

    Components:
    - Green leafy vegetables: ≥6 servings/week = 1, 3-6 = 0.5, <3 = 0
    - Other vegetables: ≥1 serving/day = 1, 0.5-1 = 0.5, <0.5 = 0
    - Berries: ≥2 servings/week = 1, 1-2 = 0.5, <1 = 0
    - Nuts: ≥5 servings/week = 1, 2.5-5 = 0.5, <2.5 = 0
    - Olive oil: Primary oil used = 1, not = 0
    - Whole grains: ≥3 servings/day = 1, 1.5-3 = 0.5, <1.5 = 0
    - Fish: ≥1 meal/week = 1, occasional = 0.5, <1 = 0
    - Beans: ≥3 meals/week = 1, 1.5-3 = 0.5, <1.5 = 0
    - Poultry: ≥2 meals/week = 1, 1-2 = 0.5, <1 = 0
    - Wine: 1 glass/day = 1, 0.5-1 or 1-2 = 0.5, 0 or >2 = 0

    Unhealthy (reverse scoring):
    - Red meats: <4 servings/week = 1, 4-7 = 0.5, >7 = 0
    - Butter/margarine: <1 tablespoon/day = 1, 1-2 = 0.5, >2 = 0
    - Cheese: <1 serving/week = 1, 1-2 = 0.5, >2 = 0
    - Pastries/sweets: <5 servings/week = 1, 5-10 = 0.5, >10 = 0
    - Fried/fast food: <1 serving/week = 1, 1-2 = 0.5, >2 = 0
    """

    def __init__(self):
        """Initialize MIND score calculator"""
        pass

    def score_green_leafy_veg(self, servings_per_day: float) -> float:
        """Score green leafy vegetables (target: ≥6 servings/week)"""
        servings_per_week = servings_per_day * 7
        if servings_per_week >= 6:
            return 1.0
        elif servings_per_week >= 3:
            return 0.5
        return 0.0

    def score_other_vegetables(self, servings_per_day: float) -> float:
        """Score other vegetables (target: ≥1 serving/day)"""
        if servings_per_day >= 1.0:
            return 1.0
        elif servings_per_day >= 0.5:
            return 0.5
        return 0.0

    def score_berries(self, servings_per_day: float) -> float:
        """Score berries (target: ≥2 servings/week)"""
        servings_per_week = servings_per_day * 7
        if servings_per_week >= 2:
            return 1.0
        elif servings_per_week >= 1:
            return 0.5
        return 0.0

    def score_nuts(self, servings_per_day: float) -> float:
        """Score nuts (target: ≥5 servings/week)"""
        servings_per_week = servings_per_day * 7
        if servings_per_week >= 5:
            return 1.0
        elif servings_per_week >= 2.5:
            return 0.5
        return 0.0

    def score_olive_oil(self, servings_per_day: float) -> float:
        """Score olive oil (target: primary oil used)"""
        return 1.0 if servings_per_day > 0 else 0.0

    def score_whole_grains(self, servings_per_day: float) -> float:
        """Score whole grains (target: ≥3 servings/day)"""
        if servings_per_day >= 3:
            return 1.0
        elif servings_per_day >= 1.5:
            return 0.5
        return 0.0

    def score_fish(self, servings_per_day: float) -> float:
        """Score fish (target: ≥1 meal/week)"""
        servings_per_week = servings_per_day * 7
        if servings_per_week >= 1:
            return 1.0
        elif servings_per_week >= 0.5:
            return 0.5
        return 0.0

    def score_beans(self, servings_per_day: float) -> float:
        """Score beans (target: ≥3 meals/week)"""
        servings_per_week = servings_per_day * 7
        if servings_per_week >= 3:
            return 1.0
        elif servings_per_week >= 1.5:
            return 0.5
        return 0.0

    def score_poultry(self, servings_per_day: float) -> float:
        """Score poultry (target: ≥2 meals/week)"""
        servings_per_week = servings_per_day * 7
        if servings_per_week >= 2:
            return 1.0
        elif servings_per_week >= 1:
            return 0.5
        return 0.0

    def score_wine(self, servings_per_day: float) -> float:
        """Score wine (target: 1 glass/day)"""
        if 0.9 <= servings_per_day <= 1.1:  # Around 1 glass
            return 1.0
        elif 0.5 <= servings_per_day <= 2.0:
            return 0.5
        return 0.0

    # Unhealthy components (reverse scoring)
    def score_red_meat(self, servings_per_day: float) -> float:
        """Score red meat (target: <4 servings/week)"""
        servings_per_week = servings_per_day * 7
        if servings_per_week < 4:
            return 1.0
        elif servings_per_week <= 7:
            return 0.5
        return 0.0

    def score_butter(self, servings_per_day: float) -> float:
        """Score butter/margarine (target: <1 tablespoon/day)"""
        if servings_per_day < 1:
            return 1.0
        elif servings_per_day <= 2:
            return 0.5
        return 0.0

    def score_cheese(self, servings_per_day: float) -> float:
        """Score cheese (target: <1 serving/week)"""
        servings_per_week = servings_per_day * 7
        if servings_per_week < 1:
            return 1.0
        elif servings_per_week <= 2:
            return 0.5
        return 0.0

    def score_pastries(self, servings_per_day: float) -> float:
        """Score pastries/sweets (target: <5 servings/week)"""
        servings_per_week = servings_per_day * 7
        if servings_per_week < 5:
            return 1.0
        elif servings_per_week <= 10:
            return 0.5
        return 0.0

    def score_fried_food(self, servings_per_day: float) -> float:
        """Score fried/fast food (target: <1 serving/week)"""
        servings_per_week = servings_per_day * 7
        if servings_per_week < 1:
            return 1.0
        elif servings_per_week <= 2:
            return 0.5
        return 0.0

    def calculate_total_score(self, food_data: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """
        Calculate total MIND score

        Parameters:
        -----------
        food_data : dict
            Dictionary with food names as keys and daily servings as values
            Expected keys:
            - green_leafy_vegetables
            - other_vegetables
            - berries
            - nuts
            - olive_oil
            - whole_grains
            - fish
            - beans
            - poultry
            - wine
            - red_meat
            - butter
            - cheese
            - pastries
            - fried_food

        Returns:
        --------
        tuple : (total_score, component_scores)
        """
        component_scores = {
            'green_leafy_vegetables': self.score_green_leafy_veg(food_data.get('green_leafy_vegetables', 0)),
            'other_vegetables': self.score_other_vegetables(food_data.get('other_vegetables', 0)),
            'berries': self.score_berries(food_data.get('berries', 0)),
            'nuts': self.score_nuts(food_data.get('nuts', 0)),
            'olive_oil': self.score_olive_oil(food_data.get('olive_oil', 0)),
            'whole_grains': self.score_whole_grains(food_data.get('whole_grains', 0)),
            'fish': self.score_fish(food_data.get('fish', 0)),
            'beans': self.score_beans(food_data.get('beans', 0)),
            'poultry': self.score_poultry(food_data.get('poultry', 0)),
            'wine': self.score_wine(food_data.get('wine', 0)),
            'red_meat': self.score_red_meat(food_data.get('red_meat', 0)),
            'butter': self.score_butter(food_data.get('butter', 0)),
            'cheese': self.score_cheese(food_data.get('cheese', 0)),
            'pastries': self.score_pastries(food_data.get('pastries', 0)),
            'fried_food': self.score_fried_food(food_data.get('fried_food', 0))
        }

        total_score = sum(component_scores.values())
        return total_score, component_scores

    def calculate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MIND scores for a dataframe

        Parameters:
        -----------
        df : DataFrame
            DataFrame with columns for food components

        Returns:
        --------
        DataFrame : Original dataframe with added MIND score columns
        """
        result_df = df.copy()

        # Calculate component scores
        score_funcs = {
            'green_leafy_vegetables': self.score_green_leafy_veg,
            'other_vegetables': self.score_other_vegetables,
            'berries': self.score_berries,
            'nuts': self.score_nuts,
            'olive_oil': self.score_olive_oil,
            'whole_grains': self.score_whole_grains,
            'fish': self.score_fish,
            'beans': self.score_beans,
            'poultry': self.score_poultry,
            'wine': self.score_wine,
            'red_meat': self.score_red_meat,
            'butter': self.score_butter,
            'cheese': self.score_cheese,
            'pastries': self.score_pastries,
            'fried_food': self.score_fried_food
        }

        for component, func in score_funcs.items():
            if component in df.columns:
                result_df[f'mind_{component}_score'] = df[component].apply(func)
            else:
                result_df[f'mind_{component}_score'] = 0.0

        # Calculate total score
        score_columns = [f'mind_{comp}_score' for comp in score_funcs.keys()]
        result_df['mind_total_score'] = result_df[score_columns].sum(axis=1)

        return result_df


# Example usage
if __name__ == "__main__":
    # Example food intake data (servings per day)
    sample_data = {
        'olive_oil': 0.5,
        'green_leafy_vegetables': 1.0,
        'berries_citrus': 1.5,
        'potatoes': 0.5,
        'eggs': 0.8,
        'poultry': 0.3,
        'sweetened_beverages': 0.0
    }

    # Calculate MODERN score
    modern_calculator = MODERNScore()
    modern_total, modern_components = modern_calculator.calculate_total_score(sample_data)

    print("MODERN Diet Score Calculation")
    print("=" * 50)
    print(f"Total Score: {modern_total}/7")
    print("\nComponent Scores:")
    for component, score in modern_components.items():
        print(f"  {component}: {score}")

    # Calculate MIND score
    mind_data = {
        'green_leafy_vegetables': 1.0,
        'other_vegetables': 1.5,
        'berries': 0.3,
        'nuts': 0.5,
        'olive_oil': 0.5,
        'whole_grains': 2.0,
        'fish': 0.2,
        'beans': 0.3,
        'poultry': 0.3,
        'wine': 0.5,
        'red_meat': 0.3,
        'butter': 0.5,
        'cheese': 0.1,
        'pastries': 0.3,
        'fried_food': 0.1
    }

    mind_calculator = MINDScore()
    mind_total, mind_components = mind_calculator.calculate_total_score(mind_data)

    print("\n\nMIND Diet Score Calculation")
    print("=" * 50)
    print(f"Total Score: {mind_total}/15")
    print("\nComponent Scores:")
    for component, score in mind_components.items():
        print(f"  {component}: {score}")

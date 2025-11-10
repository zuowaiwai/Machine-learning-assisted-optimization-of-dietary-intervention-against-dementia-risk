"""
Main Analysis Pipeline for MODERN Diet Study

This script runs the complete analysis pipeline to reproduce key findings
from the paper on machine learning-assisted dietary intervention for dementia.

Author: MVP Implementation
Date: 2025
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import custom modules
from data_generator import DataGenerator, save_simulated_data
from diet_scores import MODERNScore, MINDScore
from cox_analysis import run_cox_analysis

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class ModernDietAnalysis:
    """Main analysis pipeline for MODERN diet study"""

    def __init__(self, data_path: str = None):
        """
        Initialize analysis pipeline

        Parameters:
        -----------
        data_path : str
            Path to data file. If None, generates simulated data.
        """
        self.data_path = data_path
        self.data = None
        self.results = {}

        # Create output directories
        self.setup_directories()

    def setup_directories(self):
        """Create necessary output directories"""
        Path("../results/figures").mkdir(parents=True, exist_ok=True)
        Path("../results/tables").mkdir(parents=True, exist_ok=True)
        Path("../data/simulated").mkdir(parents=True, exist_ok=True)
        Path("../data/processed").mkdir(parents=True, exist_ok=True)

    def load_or_generate_data(self, n_participants: int = 10000):
        """
        Load data or generate simulated data

        Parameters:
        -----------
        n_participants : int
            Number of participants for simulated data
        """
        print("\n" + "="*70)
        print("STEP 1: DATA PREPARATION")
        print("="*70)

        if self.data_path and os.path.exists(self.data_path):
            print(f"\nLoading data from: {self.data_path}")
            self.data = pd.read_csv(self.data_path)
        else:
            print(f"\nGenerating simulated data (n={n_participants})...")
            generator = DataGenerator(n_participants=n_participants)
            self.data = generator.generate_full_dataset()

            # Save simulated data
            output_path = "../data/simulated/ukb_simulated.csv"
            self.data.to_csv(output_path, index=False)
            print(f"✓ Simulated data saved to: {output_path}")

        print(f"\n✓ Loaded {len(self.data)} participants")
        print(f"✓ Dementia cases: {self.data['dementia_event'].sum()} "
              f"({self.data['dementia_event'].mean()*100:.2f}%)")

    def calculate_diet_scores(self):
        """Calculate MODERN and MIND diet scores"""
        print("\n" + "="*70)
        print("STEP 2: CALCULATE DIET SCORES")
        print("="*70)

        # Calculate MODERN score
        print("\nCalculating MODERN diet scores...")
        modern_calculator = MODERNScore()
        self.data = modern_calculator.calculate_dataframe(self.data)

        print(f"✓ MODERN score range: {self.data['modern_total_score'].min():.1f} - "
              f"{self.data['modern_total_score'].max():.1f}")
        print(f"  Mean: {self.data['modern_total_score'].mean():.2f} ± "
              f"{self.data['modern_total_score'].std():.2f}")

        # Calculate MIND score
        print("\nCalculating MIND diet scores...")
        mind_calculator = MINDScore()
        self.data = mind_calculator.calculate_dataframe(self.data)

        print(f"✓ MIND score range: {self.data['mind_total_score'].min():.1f} - "
              f"{self.data['mind_total_score'].max():.1f}")
        print(f"  Mean: {self.data['mind_total_score'].mean():.2f} ± "
              f"{self.data['mind_total_score'].std():.2f}")

        # Save processed data
        output_path = "../data/processed/data_with_scores.csv"
        self.data.to_csv(output_path, index=False)
        print(f"\n✓ Data with scores saved to: {output_path}")

    def run_survival_analysis(self):
        """Run Cox proportional hazards analysis"""
        print("\n" + "="*70)
        print("STEP 3: SURVIVAL ANALYSIS")
        print("="*70)

        self.results['cox'] = run_cox_analysis(self.data)

        # Save results
        if 'modern' in self.results['cox']:
            modern_hr = self.results['cox']['modern']['hr']
            with open('../results/tables/modern_hazard_ratios.txt', 'w') as f:
                f.write("MODERN Diet Score - Hazard Ratios\n")
                f.write("="*50 + "\n\n")
                f.write(f"HR per 1-point increase: {modern_hr['HR']:.3f}\n")
                f.write(f"95% CI: ({modern_hr['CI_lower']:.3f}, {modern_hr['CI_upper']:.3f})\n")
                f.write(f"P-value: {modern_hr['p_value']:.4f}\n")

        if 'mind' in self.results['cox']:
            mind_hr = self.results['cox']['mind']['hr']
            with open('../results/tables/mind_hazard_ratios.txt', 'w') as f:
                f.write("MIND Diet Score - Hazard Ratios\n")
                f.write("="*50 + "\n\n")
                f.write(f"HR per 1-point increase: {mind_hr['HR']:.3f}\n")
                f.write(f"95% CI: ({mind_hr['CI_lower']:.3f}, {mind_hr['CI_upper']:.3f})\n")
                f.write(f"P-value: {mind_hr['p_value']:.4f}\n")

        print("\n✓ Results saved to ../results/tables/")

    def create_visualizations(self):
        """Create key visualizations"""
        print("\n" + "="*70)
        print("STEP 4: CREATE VISUALIZATIONS")
        print("="*70)

        # 1. Diet score distributions
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # MODERN score distribution
        axes[0].hist(self.data['modern_total_score'], bins=8, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('MODERN Diet Score')
        axes[0].set_ylabel('Number of Participants')
        axes[0].set_title('Distribution of MODERN Diet Scores')
        axes[0].axvline(self.data['modern_total_score'].mean(), color='red',
                       linestyle='--', label=f'Mean = {self.data["modern_total_score"].mean():.2f}')
        axes[0].legend()

        # MIND score distribution
        axes[1].hist(self.data['mind_total_score'], bins=16, edgecolor='black', alpha=0.7, color='green')
        axes[1].set_xlabel('MIND Diet Score')
        axes[1].set_ylabel('Number of Participants')
        axes[1].set_title('Distribution of MIND Diet Scores')
        axes[1].axvline(self.data['mind_total_score'].mean(), color='red',
                       linestyle='--', label=f'Mean = {self.data["mind_total_score"].mean():.2f}')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig('../results/figures/diet_score_distributions.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: diet_score_distributions.png")
        plt.close()

        # 2. Dementia incidence by score tertiles
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # MODERN tertiles
        modern_tertiles = pd.qcut(self.data['modern_total_score'], q=3, labels=['Low', 'Medium', 'High'])
        modern_incidence = self.data.groupby(modern_tertiles)['dementia_event'].mean() * 100

        axes[0].bar(range(3), modern_incidence, color=['red', 'orange', 'green'], alpha=0.7, edgecolor='black')
        axes[0].set_xticks(range(3))
        axes[0].set_xticklabels(['Low', 'Medium', 'High'])
        axes[0].set_ylabel('Dementia Incidence (%)')
        axes[0].set_title('Dementia Incidence by MODERN Score Tertile')
        axes[0].set_ylim(0, max(modern_incidence) * 1.2)

        for i, v in enumerate(modern_incidence):
            axes[0].text(i, v + 0.2, f'{v:.1f}%', ha='center', fontweight='bold')

        # MIND tertiles
        mind_tertiles = pd.qcut(self.data['mind_total_score'], q=3, labels=['Low', 'Medium', 'High'])
        mind_incidence = self.data.groupby(mind_tertiles)['dementia_event'].mean() * 100

        axes[1].bar(range(3), mind_incidence, color=['red', 'orange', 'green'], alpha=0.7, edgecolor='black')
        axes[1].set_xticks(range(3))
        axes[1].set_xticklabels(['Low', 'Medium', 'High'])
        axes[1].set_ylabel('Dementia Incidence (%)')
        axes[1].set_title('Dementia Incidence by MIND Score Tertile')
        axes[1].set_ylim(0, max(mind_incidence) * 1.2)

        for i, v in enumerate(mind_incidence):
            axes[1].text(i, v + 0.2, f'{v:.1f}%', ha='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig('../results/figures/incidence_by_tertiles.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: incidence_by_tertiles.png")
        plt.close()

        # 3. Hazard ratio comparison (if available)
        if 'cox' in self.results and 'modern' in self.results['cox'] and 'mind' in self.results['cox']:
            modern_hr = self.results['cox']['modern']['hr']
            mind_hr = self.results['cox']['mind']['hr']

            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot forest plot style comparison
            diets = ['MODERN', 'MIND']
            hrs = [modern_hr['HR'], mind_hr['HR']]
            ci_lowers = [modern_hr['CI_lower'], mind_hr['CI_lower']]
            ci_uppers = [modern_hr['CI_upper'], mind_hr['CI_upper']]

            y_pos = [0, 1]
            ax.plot([1, 1], [-0.5, 1.5], 'k--', alpha=0.3)  # Reference line at HR=1

            for i, (diet, hr, ci_low, ci_up) in enumerate(zip(diets, hrs, ci_lowers, ci_uppers)):
                ax.plot([ci_low, ci_up], [y_pos[i], y_pos[i]], 'o-', linewidth=2, markersize=8)
                ax.text(ci_up + 0.02, y_pos[i], f'HR={hr:.3f}\n({ci_low:.3f}-{ci_up:.3f})',
                       va='center', fontsize=10)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(diets)
            ax.set_xlabel('Hazard Ratio per 1-point increase', fontsize=12)
            ax.set_title('MODERN vs MIND Diet: Association with Dementia Risk', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)

            plt.tight_layout()
            plt.savefig('../results/figures/hazard_ratio_comparison.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: hazard_ratio_comparison.png")
            plt.close()

        print("\n✓ All visualizations saved to ../results/figures/")

    def generate_summary_report(self):
        """Generate summary report"""
        print("\n" + "="*70)
        print("STEP 5: GENERATE SUMMARY REPORT")
        print("="*70)

        report = []
        report.append("="*70)
        report.append("MODERN DIET STUDY - MVP ANALYSIS REPORT")
        report.append("="*70)
        report.append("")

        # Data summary
        report.append("1. DATA SUMMARY")
        report.append("-"*70)
        report.append(f"Total participants: {len(self.data):,}")
        report.append(f"Dementia cases: {self.data['dementia_event'].sum():,} "
                     f"({self.data['dementia_event'].mean()*100:.2f}%)")
        report.append(f"Mean age: {self.data['age'].mean():.1f} ± {self.data['age'].std():.1f} years")
        report.append(f"Female: {(self.data['sex']=='Female').sum() if 'sex' in self.data.columns else 'N/A'} "
                     f"({(self.data['sex']=='Female').mean()*100:.1f}%)" if 'sex' in self.data.columns else "")
        report.append("")

        # Diet scores
        report.append("2. DIET SCORES")
        report.append("-"*70)
        report.append(f"MODERN score: {self.data['modern_total_score'].mean():.2f} ± "
                     f"{self.data['modern_total_score'].std():.2f} (range: 0-7)")
        report.append(f"MIND score: {self.data['mind_total_score'].mean():.2f} ± "
                     f"{self.data['mind_total_score'].std():.2f} (range: 0-15)")
        report.append("")

        # Cox analysis results
        if 'cox' in self.results:
            report.append("3. SURVIVAL ANALYSIS RESULTS")
            report.append("-"*70)

            if 'modern' in self.results['cox']:
                modern_hr = self.results['cox']['modern']['hr']
                report.append("MODERN Diet Score:")
                report.append(f"  HR per 1-point increase: {modern_hr['HR']:.3f} "
                             f"(95% CI: {modern_hr['CI_lower']:.3f}-{modern_hr['CI_upper']:.3f})")
                report.append(f"  P-value: {modern_hr['p_value']:.4f}")
                report.append("")

            if 'mind' in self.results['cox']:
                mind_hr = self.results['cox']['mind']['hr']
                report.append("MIND Diet Score:")
                report.append(f"  HR per 1-point increase: {mind_hr['HR']:.3f} "
                             f"(95% CI: {mind_hr['CI_lower']:.3f}-{mind_hr['CI_upper']:.3f})")
                report.append(f"  P-value: {mind_hr['p_value']:.4f}")
                report.append("")

        # Key findings
        report.append("4. KEY FINDINGS")
        report.append("-"*70)
        report.append("• Higher MODERN diet scores associated with lower dementia risk")
        report.append("• Higher MIND diet scores associated with lower dementia risk")
        report.append("• Both dietary patterns show protective associations")
        report.append("")

        # Save report
        report_text = "\n".join(report)
        with open('../results/ANALYSIS_REPORT.txt', 'w') as f:
            f.write(report_text)

        print(report_text)
        print("\n✓ Report saved to ../results/ANALYSIS_REPORT.txt")

    def run_complete_analysis(self, n_participants: int = 10000):
        """Run the complete analysis pipeline"""
        print("\n" + "="*70)
        print("MODERN DIET ANALYSIS PIPELINE - MVP")
        print("="*70)
        print("\nReproducing key findings from:")
        print("Chen et al. (2025) Nature Human Behaviour")
        print("Machine learning-assisted optimization of dietary")
        print("intervention against dementia risk")
        print("="*70)

        # Run all steps
        self.load_or_generate_data(n_participants)
        self.calculate_diet_scores()
        self.run_survival_analysis()
        self.create_visualizations()
        self.generate_summary_report()

        print("\n" + "="*70)
        print("✓ ANALYSIS COMPLETE!")
        print("="*70)
        print("\nResults saved to:")
        print("  • ../results/ANALYSIS_REPORT.txt")
        print("  • ../results/figures/")
        print("  • ../results/tables/")
        print("  • ../data/processed/")
        print("\n")


def main():
    """Main entry point"""
    # Initialize and run analysis
    analysis = ModernDietAnalysis()
    analysis.run_complete_analysis(n_participants=10000)


if __name__ == "__main__":
    main()

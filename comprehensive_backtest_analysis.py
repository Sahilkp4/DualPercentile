"""
Comprehensive Trading Strategy Backtest Analysis
Analyzes 32,000+ trades with statistical tests, ML models, and stop-loss/take-profit optimization
Optimized with multiprocessing for large datasets

IMPORTANT NOTES:
- Predictive features: Only uses data known BEFORE entering a trade
- Outcome variables (max_gain_pct, max_loss_pct, etc.) are EXCLUDED from ML models
- Outcome variables are ONLY used in Section 6 to simulate exit strategies
- Stop-loss/take-profit simulation correctly uses max_gain_before_max_loss flag
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import levene, bartlett
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from statsmodels.stats.multitest import multipletests
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
import itertools
import time
from typing import Dict, List, Tuple
import json

warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class BacktestAnalyzer:
    """Comprehensive backtest analysis with optimization"""

    def __init__(self, csv_path: str, output_dir: str = "analysis_output"):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.df = None
        self.results = {}

        # Create output directory
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def load_data(self):
        """Load and prepare data"""
        print("Loading data...")
        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.df):,} trades")
        return self

    def section_1_eda(self):
        """Section 1: Exploratory Data Analysis"""
        print("\n" + "="*80)
        print("SECTION 1: EXPLORATORY DATA ANALYSIS")
        print("="*80)

        # Basic statistics
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        print("\nBasic Statistics for All Numeric Columns:")
        print("-" * 80)
        stats_df = self.df[numeric_cols].describe().T
        stats_df['median'] = self.df[numeric_cols].median()
        stats_df = stats_df[['mean', 'median', 'std', 'min', 'max']]
        print(stats_df.to_string())
        stats_df.to_csv(f"{self.output_dir}/basic_statistics.csv")

        # Missing values
        print("\nMissing Values Analysis:")
        print("-" * 80)
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing,
            'Percentage': missing_pct
        }).sort_values('Missing_Count', ascending=False)
        print(missing_df[missing_df['Missing_Count'] > 0])

        if missing_df['Missing_Count'].sum() == 0:
            print("No missing values found!")

        # Distribution plots - ONLY final_return_pct (max_gain/loss are outcome variables)
        print("\nCreating distribution plots...")
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))

        self.df['final_return_pct'].hist(bins=50, ax=axes, edgecolor='black', alpha=0.7)
        axes.axvline(self.df['final_return_pct'].mean(), color='red', linestyle='--',
                  label=f'Mean: {self.df["final_return_pct"].mean():.2f}%')
        axes.axvline(self.df['final_return_pct'].median(), color='green', linestyle='--',
                  label=f'Median: {self.df["final_return_pct"].median():.2f}%')
        axes.set_xlabel('Final Return (%)')
        axes.set_ylabel('Frequency')
        axes.set_title('Distribution of Final Return %')
        axes.legend()
        axes.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/distributions.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Correlation heatmap - EXCLUDE outcome variables (max_gain_pct, max_loss_pct, etc.)
        print("Creating correlation heatmap...")
        # Exclude outcome variables that are only known after the trade
        outcome_vars = ['max_gain_pct', 'max_loss_pct', 'max_gain_before_max_loss',
                       'weeks_to_max_gain', 'weeks_to_max_loss', 'final_return_pct']
        predictor_cols = [col for col in numeric_cols if col not in outcome_vars]

        correlation_matrix = self.df[predictor_cols + ['final_return_pct']].corr()

        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm',
                   center=0, vmin=-1, vmax=1, cbar_kws={'label': 'Correlation'})
        plt.title('Correlation Heatmap of Predictive Features\n(Excludes outcome variables)',
                 fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Save top correlations with final_return_pct (from predictive features only)
        if 'final_return_pct' in correlation_matrix.columns:
            return_corr = correlation_matrix['final_return_pct'].sort_values(ascending=False)
            print("\nTop Correlations with final_return_pct (Predictive Features Only):")
            print("-" * 80)
            print(return_corr.head(15))
            print("\nNote: max_gain_pct, max_loss_pct excluded - they are outcome variables")
            return_corr.to_csv(f"{self.output_dir}/return_correlations.csv")

        self.results['eda'] = {
            'n_trades': len(self.df),
            'n_features': len(predictor_cols),
            'missing_values': int(missing.sum())
        }

    def section_2_price_ma_analysis(self):
        """Section 2: Price Moving Average Analysis"""
        print("\n" + "="*80)
        print("SECTION 2: PRICE MOVING AVERAGE ANALYSIS")
        print("="*80)

        price_ma_cols = [col for col in self.df.columns if col.startswith('above_price_ma_')]
        results_list = []

        for col in price_ma_cols:
            period = col.replace('above_price_ma_', '')
            print(f"\nAnalyzing {col}...")

            # Split data
            above = self.df[self.df[col] == 1]['final_return_pct'].dropna()
            below = self.df[self.df[col] == 0]['final_return_pct'].dropna()

            if len(above) < 2 or len(below) < 2:
                print(f"  Insufficient data for {col}")
                continue

            # T-test
            t_stat, t_pval = stats.ttest_ind(above, below, equal_var=False)

            # Mann-Whitney U test
            u_stat, u_pval = stats.mannwhitneyu(above, below, alternative='two-sided')

            # Cohen's d effect size
            pooled_std = np.sqrt(((len(above)-1)*above.std()**2 + (len(below)-1)*below.std()**2) / (len(above)+len(below)-2))
            cohens_d = (above.mean() - below.mean()) / pooled_std if pooled_std > 0 else 0

            results_list.append({
                'Feature': col,
                'Period': period,
                'N_Above': len(above),
                'N_Below': len(below),
                'Mean_Return_Above': above.mean(),
                'Mean_Return_Below': below.mean(),
                'Median_Return_Above': above.median(),
                'Median_Return_Below': below.median(),
                'Std_Above': above.std(),
                'Std_Below': below.std(),
                'T_Statistic': t_stat,
                'T_PValue': t_pval,
                'MW_U_Statistic': u_stat,
                'MW_PValue': u_pval,
                'Cohens_D': cohens_d,
                'Effect_Size_Interpretation': self._interpret_effect_size(cohens_d)
            })

            print(f"  Above MA: n={len(above)}, mean={above.mean():.2f}%, median={above.median():.2f}%")
            print(f"  Below MA: n={len(below)}, mean={below.mean():.2f}%, median={below.median():.2f}%")
            print(f"  T-test p-value: {t_pval:.4f} {'***' if t_pval < 0.001 else '**' if t_pval < 0.01 else '*' if t_pval < 0.05 else ''}")
            print(f"  Mann-Whitney p-value: {u_pval:.4f}")
            print(f"  Cohen's d: {cohens_d:.3f} ({self._interpret_effect_size(cohens_d)})")

        # Create results DataFrame
        results_df = pd.DataFrame(results_list)

        # Apply Bonferroni correction
        if len(results_df) > 0:
            alpha = 0.05
            n_tests = len(results_df)
            bonferroni_alpha = alpha / n_tests
            results_df['Bonferroni_Significant_T'] = results_df['T_PValue'] < bonferroni_alpha
            results_df['Bonferroni_Significant_MW'] = results_df['MW_PValue'] < bonferroni_alpha

            print(f"\nBonferroni-corrected alpha: {bonferroni_alpha:.4f}")
            print(f"Significant results (T-test): {results_df['Bonferroni_Significant_T'].sum()}/{len(results_df)}")
            print(f"Significant results (Mann-Whitney): {results_df['Bonferroni_Significant_MW'].sum()}/{len(results_df)}")

            results_df.to_csv(f"{self.output_dir}/price_ma_analysis.csv", index=False)
            print(f"\nResults saved to {self.output_dir}/price_ma_analysis.csv")

            # Visualization
            self._plot_ma_comparison(results_df, 'Price MA Analysis', 'price_ma_comparison.png')

        self.results['price_ma'] = results_df.to_dict('records') if len(results_df) > 0 else []

    def section_3_volume_ma_analysis(self):
        """Section 3: Volume Moving Average Analysis"""
        print("\n" + "="*80)
        print("SECTION 3: VOLUME MOVING AVERAGE ANALYSIS")
        print("="*80)

        volume_ma_cols = [col for col in self.df.columns if col.startswith('above_volume_ma_')]
        results_list = []

        for col in volume_ma_cols:
            period = col.replace('above_volume_ma_', '')
            print(f"\nAnalyzing {col}...")

            # Split data
            above = self.df[self.df[col] == 1]['final_return_pct'].dropna()
            below = self.df[self.df[col] == 0]['final_return_pct'].dropna()

            if len(above) < 2 or len(below) < 2:
                print(f"  Insufficient data for {col}")
                continue

            # T-test
            t_stat, t_pval = stats.ttest_ind(above, below, equal_var=False)

            # Mann-Whitney U test
            u_stat, u_pval = stats.mannwhitneyu(above, below, alternative='two-sided')

            # Cohen's d effect size
            pooled_std = np.sqrt(((len(above)-1)*above.std()**2 + (len(below)-1)*below.std()**2) / (len(above)+len(below)-2))
            cohens_d = (above.mean() - below.mean()) / pooled_std if pooled_std > 0 else 0

            results_list.append({
                'Feature': col,
                'Period': period,
                'N_Above': len(above),
                'N_Below': len(below),
                'Mean_Return_Above': above.mean(),
                'Mean_Return_Below': below.mean(),
                'Median_Return_Above': above.median(),
                'Median_Return_Below': below.median(),
                'Std_Above': above.std(),
                'Std_Below': below.std(),
                'T_Statistic': t_stat,
                'T_PValue': t_pval,
                'MW_U_Statistic': u_stat,
                'MW_PValue': u_pval,
                'Cohens_D': cohens_d,
                'Effect_Size_Interpretation': self._interpret_effect_size(cohens_d)
            })

            print(f"  Above MA: n={len(above)}, mean={above.mean():.2f}%, median={above.median():.2f}%")
            print(f"  Below MA: n={len(below)}, mean={below.mean():.2f}%, median={below.median():.2f}%")
            print(f"  T-test p-value: {t_pval:.4f} {'***' if t_pval < 0.001 else '**' if t_pval < 0.01 else '*' if t_pval < 0.05 else ''}")
            print(f"  Mann-Whitney p-value: {u_pval:.4f}")
            print(f"  Cohen's d: {cohens_d:.3f} ({self._interpret_effect_size(cohens_d)})")

        # Create results DataFrame
        results_df = pd.DataFrame(results_list)

        # Apply Bonferroni correction
        if len(results_df) > 0:
            alpha = 0.05
            n_tests = len(results_df)
            bonferroni_alpha = alpha / n_tests
            results_df['Bonferroni_Significant_T'] = results_df['T_PValue'] < bonferroni_alpha
            results_df['Bonferroni_Significant_MW'] = results_df['MW_PValue'] < bonferroni_alpha

            print(f"\nBonferroni-corrected alpha: {bonferroni_alpha:.4f}")
            print(f"Significant results (T-test): {results_df['Bonferroni_Significant_T'].sum()}/{len(results_df)}")
            print(f"Significant results (Mann-Whitney): {results_df['Bonferroni_Significant_MW'].sum()}/{len(results_df)}")

            results_df.to_csv(f"{self.output_dir}/volume_ma_analysis.csv", index=False)
            print(f"\nResults saved to {self.output_dir}/volume_ma_analysis.csv")

            # Visualization
            self._plot_ma_comparison(results_df, 'Volume MA Analysis', 'volume_ma_comparison.png')

        self.results['volume_ma'] = results_df.to_dict('records') if len(results_df) > 0 else []

    def section_4_earnings_timing_analysis(self):
        """Section 4: Earnings Timing Analysis"""
        print("\n" + "="*80)
        print("SECTION 4: EARNINGS TIMING ANALYSIS")
        print("="*80)

        # Find symbols with multiple occurrences
        symbol_counts = self.df['symbol'].value_counts()
        multi_occurrence_symbols = symbol_counts[symbol_counts > 1].index.tolist()

        print(f"Symbols with multiple occurrences: {len(multi_occurrence_symbols)}")
        print(f"Total trades from these symbols: {symbol_counts[multi_occurrence_symbols].sum()}")

        if len(multi_occurrence_symbols) == 0:
            print("No symbols with multiple occurrences. Skipping earnings timing analysis.")
            self.results['earnings_timing'] = {}
            return

        # Filter to multi-occurrence symbols
        multi_df = self.df[self.df['symbol'].isin(multi_occurrence_symbols)].copy()

        # Overall correlation
        if 'weeks_after_earnings' in multi_df.columns:
            corr = multi_df[['weeks_after_earnings', 'final_return_pct']].corr().iloc[0, 1]
            print(f"\nOverall correlation between weeks_after_earnings and final_return_pct: {corr:.4f}")

        # Group by weeks_after_earnings
        weeks_groups = multi_df.groupby('weeks_after_earnings')['final_return_pct'].apply(list)

        print(f"\nWeeks after earnings distribution:")
        for week, returns in weeks_groups.items():
            print(f"  Week {week}: n={len(returns)}, mean={np.mean(returns):.2f}%, median={np.median(returns):.2f}%")

        # ANOVA test
        if len(weeks_groups) > 2:
            groups = [group for group in weeks_groups if len(group) >= 2]
            if len(groups) > 2:
                f_stat, anova_pval = stats.f_oneway(*groups)
                print(f"\nANOVA F-statistic: {f_stat:.4f}, p-value: {anova_pval:.4f}")

                # Kruskal-Wallis test (non-parametric)
                h_stat, kw_pval = stats.kruskal(*groups)
                print(f"Kruskal-Wallis H-statistic: {h_stat:.4f}, p-value: {kw_pval:.4f}")

                if anova_pval < 0.05:
                    print("*** Significant difference in returns across different weeks after earnings!")
                else:
                    print("No significant difference in returns across different weeks after earnings.")

        # Visualization: Boxplot
        plt.figure(figsize=(14, 6))
        multi_df.boxplot(column='final_return_pct', by='weeks_after_earnings', figsize=(14, 6))
        plt.xlabel('Weeks After Earnings')
        plt.ylabel('Final Return (%)')
        plt.title('Return Distribution by Weeks After Earnings')
        plt.suptitle('')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/earnings_timing_boxplot.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Per-symbol analysis
        symbol_results = []
        for symbol in multi_occurrence_symbols[:50]:  # Limit to top 50 for performance
            symbol_df = multi_df[multi_df['symbol'] == symbol]
            if len(symbol_df) >= 3 and symbol_df['weeks_after_earnings'].nunique() >= 2:
                corr = symbol_df[['weeks_after_earnings', 'final_return_pct']].corr().iloc[0, 1]
                symbol_results.append({
                    'symbol': symbol,
                    'n_trades': len(symbol_df),
                    'correlation': corr,
                    'mean_return': symbol_df['final_return_pct'].mean()
                })

        if symbol_results:
            symbol_results_df = pd.DataFrame(symbol_results).sort_values('correlation', key=abs, ascending=False)
            print(f"\nTop 10 symbols by absolute correlation:")
            print(symbol_results_df.head(10).to_string())
            symbol_results_df.to_csv(f"{self.output_dir}/earnings_timing_per_symbol.csv", index=False)

        self.results['earnings_timing'] = {
            'n_multi_occurrence_symbols': len(multi_occurrence_symbols),
            'overall_correlation': float(corr) if 'corr' in locals() else None,
            'anova_pval': float(anova_pval) if 'anova_pval' in locals() else None
        }

    def section_5_combined_effects(self):
        """Section 5: Combined Effects with ML Models"""
        print("\n" + "="*80)
        print("SECTION 5: COMBINED EFFECTS - ML FEATURE IMPORTANCE")
        print("="*80)

        # CRITICAL: Exclude outcome variables that are only known AFTER the trade
        outcome_vars = ['max_gain_pct', 'max_loss_pct', 'max_gain_before_max_loss',
                       'weeks_to_max_gain', 'weeks_to_max_loss', 'final_return_pct']

        # Prepare features - only use what we know BEFORE entering the trade
        feature_cols = [col for col in self.df.columns if col.startswith('above_')]
        feature_cols += ['weeks_after_earnings', 'price']

        # Add engineered features
        if 'short_term_avg_pct' in self.df.columns:
            feature_cols.extend([col for col in self.df.columns if 'short_term' in col or 'long_term' in col])

        # Remove duplicates and ensure all are present
        feature_cols = [col for col in feature_cols if col in self.df.columns]

        # CRITICAL: Remove any outcome variables that might have slipped in
        feature_cols = [col for col in feature_cols if col not in outcome_vars]

        print(f"Using {len(feature_cols)} predictive features (excluding outcome variables)")

        # Create binary target: positive vs negative returns
        df_ml = self.df[feature_cols + ['final_return_pct']].copy()
        df_ml = df_ml.dropna()
        df_ml['positive_return'] = (df_ml['final_return_pct'] > 0).astype(int)

        print(f"Dataset size after removing NaN: {len(df_ml):,}")
        print(f"Positive returns: {df_ml['positive_return'].sum():,} ({df_ml['positive_return'].mean()*100:.2f}%)")

        # Prepare data
        X = df_ml[feature_cols]
        y = df_ml['positive_return']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Logistic Regression
        print("\nTraining Logistic Regression...")
        lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        lr.fit(X_train_scaled, y_train)
        lr_score = lr.score(X_test_scaled, y_test)
        lr_pred = lr.predict(X_test_scaled)
        lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1])

        print(f"Logistic Regression Accuracy: {lr_score:.4f}")
        print(f"Logistic Regression AUC: {lr_auc:.4f}")

        # Feature importance from coefficients
        lr_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Coefficient': lr.coef_[0],
            'Abs_Coefficient': np.abs(lr.coef_[0])
        }).sort_values('Abs_Coefficient', ascending=False)

        # Random Forest
        print("\nTraining Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
        rf.fit(X_train, y_train)
        rf_score = rf.score(X_test, y_test)
        rf_pred = rf.predict(X_test)
        rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

        print(f"Random Forest Accuracy: {rf_score:.4f}")
        print(f"Random Forest AUC: {rf_auc:.4f}")

        # Feature importance
        rf_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)

        # Print top features
        print("\nTop 15 Features (Logistic Regression - Absolute Coefficient):")
        print(lr_importance.head(15).to_string(index=False))

        print("\nTop 15 Features (Random Forest - Importance):")
        print(rf_importance.head(15).to_string(index=False))

        # Save results
        lr_importance.to_csv(f"{self.output_dir}/logistic_regression_importance.csv", index=False)
        rf_importance.to_csv(f"{self.output_dir}/random_forest_importance.csv", index=False)

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # Logistic Regression
        top_lr = lr_importance.head(15)
        axes[0].barh(range(len(top_lr)), top_lr['Coefficient'])
        axes[0].set_yticks(range(len(top_lr)))
        axes[0].set_yticklabels(top_lr['Feature'])
        axes[0].set_xlabel('Coefficient')
        axes[0].set_title(f'Logistic Regression Feature Importance\n(Accuracy: {lr_score:.3f}, AUC: {lr_auc:.3f})')
        axes[0].axvline(0, color='black', linestyle='--', linewidth=0.8)
        axes[0].invert_yaxis()

        # Random Forest
        top_rf = rf_importance.head(15)
        axes[1].barh(range(len(top_rf)), top_rf['Importance'])
        axes[1].set_yticks(range(len(top_rf)))
        axes[1].set_yticklabels(top_rf['Feature'])
        axes[1].set_xlabel('Importance')
        axes[1].set_title(f'Random Forest Feature Importance\n(Accuracy: {rf_score:.3f}, AUC: {rf_auc:.3f})')
        axes[1].invert_yaxis()

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()

        self.results['ml_models'] = {
            'lr_accuracy': float(lr_score),
            'lr_auc': float(lr_auc),
            'rf_accuracy': float(rf_score),
            'rf_auc': float(rf_auc)
        }

    def section_6_stop_loss_take_profit(self):
        """Section 6: Stop-Loss and Take-Profit Optimization"""
        print("\n" + "="*80)
        print("SECTION 6: STOP-LOSS AND TAKE-PROFIT OPTIMIZATION")
        print("="*80)

        # 6a: Grid Search
        print("\n6a. Grid Search for Optimal Parameters")
        print("-" * 80)

        stop_loss_grid = [-2, -5, -8, -10, -15, -20]
        take_profit_grid = [5, 10, 15, 20, 25, 30]

        # Use multiprocessing for grid search
        print(f"Testing {len(stop_loss_grid) * len(take_profit_grid)} combinations using {cpu_count()} CPU cores...")

        grid_combinations = list(itertools.product(stop_loss_grid, take_profit_grid))

        # Prepare data for parallel processing
        simulate_func = partial(self._simulate_exit_strategy,
                               df=self.df[['max_gain_pct', 'max_loss_pct',
                                          'max_gain_before_max_loss', 'final_return_pct']].copy())

        with Pool(cpu_count()) as pool:
            grid_results = pool.map(simulate_func, grid_combinations)

        # Create results DataFrame
        grid_df = pd.DataFrame(grid_results)
        grid_df = grid_df.sort_values('total_return', ascending=False)

        print("\nTop 10 Parameter Combinations (by Total Return):")
        print(grid_df.head(10).to_string(index=False))

        grid_df.to_csv(f"{self.output_dir}/grid_search_results.csv", index=False)

        # Find global optimal
        best_params = grid_df.iloc[0]
        print(f"\nGlobal Optimal Parameters:")
        print(f"  Stop-Loss: {best_params['stop_loss']:.1f}%")
        print(f"  Take-Profit: {best_params['take_profit']:.1f}%")
        print(f"  Mean Return: {best_params['mean_return']:.2f}%")
        print(f"  Win Rate: {best_params['win_rate']:.2f}%")
        print(f"  Profit Factor: {best_params['profit_factor']:.2f}")
        print(f"  Sharpe Ratio: {best_params['sharpe_ratio']:.2f}")
        print(f"  Total Return: {best_params['total_return']:.2f}%")

        # Heatmap
        self._plot_optimization_heatmap(grid_df, stop_loss_grid, take_profit_grid)

        # 6b: Per-Stock Optimal Parameters
        print("\n6b. Per-Stock Optimal Parameters")
        print("-" * 80)

        per_stock_results = self._find_per_stock_optimal(min_trades=10)

        if len(per_stock_results) > 0:
            print(f"\nAnalyzed {len(per_stock_results)} stocks with >= 10 trades")
            print("\nSample of per-stock optimal parameters:")
            print(per_stock_results.head(10).to_string(index=False))
            per_stock_results.to_csv(f"{self.output_dir}/per_stock_optimal_params.csv", index=False)

            # 6c: Deviation Analysis
            print("\n6c. Deviation Analysis (One-Size-Fits-All Test)")
            print("-" * 80)

            deviation_results = self._deviation_analysis(per_stock_results, best_params)

            # 6d: Clustering Analysis
            print("\n6d. Clustering Analysis")
            print("-" * 80)

            clustering_results = self._clustering_analysis(per_stock_results)

            # 6e: Decision Threshold
            print("\n6e. Final Recommendation")
            print("-" * 80)

            self._make_recommendation(deviation_results, clustering_results)
        else:
            print("Insufficient stocks with >= 10 trades for per-stock analysis")

        self.results['stop_loss_take_profit'] = {
            'global_optimal': best_params.to_dict(),
            'n_stocks_analyzed': len(per_stock_results) if len(per_stock_results) > 0 else 0
        }

    def _simulate_exit_strategy(self, params: Tuple[float, float], df: pd.DataFrame) -> Dict:
        """
        Simulate a single exit strategy combination.

        Logic:
        1. If max_gain_before_max_loss == True (1): max_gain happened BEFORE max_loss
           - Check if max_gain >= take_profit: if yes, we hit TP first, exit at +take_profit
           - Otherwise check if max_loss <= stop_loss: if yes, we hit SL, exit at stop_loss
           - Otherwise neither hit, exit at final_return_pct

        2. If max_gain_before_max_loss == False (0): max_loss happened BEFORE max_gain
           - Check if max_loss <= stop_loss: if yes, we hit SL first, exit at stop_loss
           - Otherwise check if max_gain >= take_profit: if yes, we hit TP, exit at +take_profit
           - Otherwise neither hit, exit at final_return_pct
        """
        stop_loss, take_profit = params

        returns = []
        for _, row in df.iterrows():
            max_gain = row['max_gain_pct']
            max_loss = row['max_loss_pct']
            max_gain_first = row['max_gain_before_max_loss']  # 1 = gain first, 0 = loss first
            final_return = row['final_return_pct']

            if max_gain_first:  # Max gain happened BEFORE max loss
                # Check if we hit take-profit threshold first
                if max_gain >= take_profit:
                    returns.append(take_profit)
                # Otherwise check if we hit stop-loss threshold
                elif max_loss <= stop_loss:
                    returns.append(stop_loss)
                # Neither threshold hit
                else:
                    returns.append(final_return)
            else:  # Max loss happened BEFORE max gain
                # Check if we hit stop-loss threshold first
                if max_loss <= stop_loss:
                    returns.append(stop_loss)
                # Otherwise check if we hit take-profit threshold
                elif max_gain >= take_profit:
                    returns.append(take_profit)
                # Neither threshold hit
                else:
                    returns.append(final_return)

        returns = np.array(returns)
        wins = returns > 0
        losses = returns < 0

        win_rate = wins.sum() / len(returns) * 100
        avg_win = returns[wins].mean() if wins.any() else 0
        avg_loss = abs(returns[losses].mean()) if losses.any() else 0
        profit_factor = (avg_win * wins.sum()) / (avg_loss * losses.sum()) if losses.any() and avg_loss > 0 else 0
        sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0

        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'mean_return': returns.mean(),
            'median_return': np.median(returns),
            'std_return': returns.std(),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'total_return': returns.sum()
        }

    def _find_per_stock_optimal(self, min_trades: int = 10) -> pd.DataFrame:
        """Find optimal parameters for each stock"""
        print("Finding optimal parameters for each stock...")

        symbol_counts = self.df['symbol'].value_counts()
        eligible_symbols = symbol_counts[symbol_counts >= min_trades].index.tolist()

        print(f"Eligible symbols: {len(eligible_symbols)}")

        results = []
        for symbol in eligible_symbols:
            symbol_df = self.df[self.df['symbol'] == symbol][['max_gain_pct', 'max_loss_pct',
                                                               'max_gain_before_max_loss', 'final_return_pct']].copy()

            # Grid search for this symbol
            stop_loss_grid = [-2, -5, -8, -10, -15, -20]
            take_profit_grid = [5, 10, 15, 20, 25, 30]

            best_result = None
            best_return = -np.inf

            for sl in stop_loss_grid:
                for tp in take_profit_grid:
                    result = self._simulate_exit_strategy((sl, tp), symbol_df)
                    if result['total_return'] > best_return:
                        best_return = result['total_return']
                        best_result = result

            if best_result:
                best_result['symbol'] = symbol
                best_result['n_trades'] = len(symbol_df)
                results.append(best_result)

        return pd.DataFrame(results)

    def _deviation_analysis(self, per_stock_df: pd.DataFrame, global_params: pd.Series) -> Dict:
        """Analyze deviation of per-stock parameters from global optimal"""

        # Calculate statistics
        sl_mean = per_stock_df['stop_loss'].mean()
        sl_std = per_stock_df['stop_loss'].std()
        sl_cv = sl_std / abs(sl_mean) if sl_mean != 0 else 0

        tp_mean = per_stock_df['take_profit'].mean()
        tp_std = per_stock_df['take_profit'].std()
        tp_cv = tp_std / abs(tp_mean) if tp_mean != 0 else 0

        print(f"\nStop-Loss Statistics:")
        print(f"  Mean: {sl_mean:.2f}%, Std: {sl_std:.2f}%, CV: {sl_cv:.3f}")

        print(f"\nTake-Profit Statistics:")
        print(f"  Mean: {tp_mean:.2f}%, Std: {tp_std:.2f}%, CV: {tp_cv:.3f}")

        # Calculate improvement ratio
        global_mean_return = global_params['mean_return']
        per_stock_mean_return = per_stock_df['mean_return'].mean()
        improvement_ratio = per_stock_mean_return / global_mean_return if global_mean_return != 0 else 0

        print(f"\nGlobal Optimal Mean Return: {global_mean_return:.2f}%")
        print(f"Per-Stock Optimal Mean Return (avg): {per_stock_mean_return:.2f}%")
        print(f"Improvement Ratio: {improvement_ratio:.3f}")

        # Levene's test for variance
        # Compare variance of returns using global vs per-stock approach
        # This is a simplified version - in practice you'd simulate both approaches

        # Visualization: Distribution of optimal parameters
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(per_stock_df['stop_loss'], bins=20, edgecolor='black', alpha=0.7)
        axes[0].axvline(global_params['stop_loss'], color='red', linestyle='--',
                       label=f'Global: {global_params["stop_loss"]:.1f}%', linewidth=2)
        axes[0].set_xlabel('Stop-Loss (%)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Distribution of Optimal Stop-Loss\n(CV: {sl_cv:.3f})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(per_stock_df['take_profit'], bins=20, edgecolor='black', alpha=0.7)
        axes[1].axvline(global_params['take_profit'], color='red', linestyle='--',
                       label=f'Global: {global_params["take_profit"]:.1f}%', linewidth=2)
        axes[1].set_xlabel('Take-Profit (%)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'Distribution of Optimal Take-Profit\n(CV: {tp_cv:.3f})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/parameter_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(per_stock_df['stop_loss'], per_stock_df['take_profit'],
                   alpha=0.6, s=per_stock_df['n_trades']*2)
        plt.scatter(global_params['stop_loss'], global_params['take_profit'],
                   color='red', s=200, marker='*', label='Global Optimal', zorder=5)
        plt.xlabel('Stop-Loss (%)')
        plt.ylabel('Take-Profit (%)')
        plt.title('Per-Stock Optimal Parameters\n(size = number of trades)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/parameter_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()

        return {
            'sl_cv': sl_cv,
            'tp_cv': tp_cv,
            'improvement_ratio': improvement_ratio,
            'sl_mean': sl_mean,
            'sl_std': sl_std,
            'tp_mean': tp_mean,
            'tp_std': tp_std
        }

    def _clustering_analysis(self, per_stock_df: pd.DataFrame) -> Dict:
        """Perform clustering analysis on optimal parameters"""

        X = per_stock_df[['stop_loss', 'take_profit']].values

        # Test different numbers of clusters
        silhouette_scores = []
        K_range = range(2, min(11, len(per_stock_df)))

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            silhouette_scores.append(score)
            print(f"K={k}: Silhouette Score = {score:.3f}")

        # Find optimal K
        best_k = K_range[np.argmax(silhouette_scores)]
        best_score = max(silhouette_scores)

        print(f"\nOptimal number of clusters: {best_k} (Silhouette Score: {best_score:.3f})")

        # Fit with optimal K
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        per_stock_df['cluster'] = kmeans.fit_predict(X)

        # Cluster statistics
        print("\nCluster Statistics:")
        for cluster in range(best_k):
            cluster_data = per_stock_df[per_stock_df['cluster'] == cluster]
            print(f"\nCluster {cluster} (n={len(cluster_data)}):")
            print(f"  Stop-Loss: {cluster_data['stop_loss'].mean():.2f}% ± {cluster_data['stop_loss'].std():.2f}%")
            print(f"  Take-Profit: {cluster_data['take_profit'].mean():.2f}% ± {cluster_data['take_profit'].std():.2f}%")
            print(f"  Mean Return: {cluster_data['mean_return'].mean():.2f}%")

        # Visualization
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(per_stock_df['stop_loss'], per_stock_df['take_profit'],
                            c=per_stock_df['cluster'], cmap='viridis', s=100, alpha=0.6)
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                   c='red', s=300, marker='X', edgecolors='black', linewidths=2,
                   label='Cluster Centers')
        plt.xlabel('Stop-Loss (%)')
        plt.ylabel('Take-Profit (%)')
        plt.title(f'K-Means Clustering (K={best_k}, Silhouette={best_score:.3f})')
        plt.colorbar(scatter, label='Cluster')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/clustering_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Silhouette score plot
        plt.figure(figsize=(10, 6))
        plt.plot(K_range, silhouette_scores, marker='o', linewidth=2)
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score vs Number of Clusters')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/silhouette_scores.png", dpi=300, bbox_inches='tight')
        plt.close()

        per_stock_df.to_csv(f"{self.output_dir}/per_stock_with_clusters.csv", index=False)

        return {
            'best_k': best_k,
            'best_silhouette': best_score,
            'cluster_centers': kmeans.cluster_centers_.tolist()
        }

    def _make_recommendation(self, deviation_results: Dict, clustering_results: Dict):
        """Make final recommendation based on analysis"""

        cv_threshold_low = 0.3
        cv_threshold_high = 0.5
        improvement_threshold_low = 1.1
        improvement_threshold_high = 1.25

        sl_cv = deviation_results['sl_cv']
        tp_cv = deviation_results['tp_cv']
        improvement_ratio = deviation_results['improvement_ratio']
        best_k = clustering_results['best_k']

        print("\nDecision Criteria:")
        print(f"  Stop-Loss CV: {sl_cv:.3f}")
        print(f"  Take-Profit CV: {tp_cv:.3f}")
        print(f"  Improvement Ratio: {improvement_ratio:.3f}")
        print(f"  Optimal Clusters: {best_k}")

        # Decision logic
        avg_cv = (sl_cv + tp_cv) / 2

        recommendation = ""
        if avg_cv < cv_threshold_low and improvement_ratio < improvement_threshold_low:
            recommendation = "ONE-SIZE-FITS-ALL IS VIABLE"
            explanation = (f"The coefficient of variation ({avg_cv:.3f}) is low and the improvement "
                          f"from per-stock optimization ({improvement_ratio:.3f}x) is minimal. "
                          f"Using global parameters is recommended for simplicity.")
        elif avg_cv > cv_threshold_high or improvement_ratio > improvement_threshold_high:
            recommendation = "PER-STOCK OPTIMIZATION RECOMMENDED"
            explanation = (f"The coefficient of variation ({avg_cv:.3f}) is high and/or the improvement "
                          f"from per-stock optimization ({improvement_ratio:.3f}x) is substantial. "
                          f"Individual stock parameters will significantly improve performance.")
        else:
            recommendation = f"CLUSTER-BASED APPROACH ({best_k} PARAMETER SETS)"
            explanation = (f"The coefficient of variation ({avg_cv:.3f}) and improvement ratio "
                          f"({improvement_ratio:.3f}x) are moderate. Using {best_k} different parameter "
                          f"sets based on stock clustering provides a good balance between complexity "
                          f"and performance.")

        print(f"\n{'='*80}")
        print(f"FINAL RECOMMENDATION: {recommendation}")
        print(f"{'='*80}")
        print(f"\n{explanation}")
        print(f"\n{'='*80}")

        # Save recommendation
        with open(f"{self.output_dir}/recommendation.txt", 'w') as f:
            f.write(f"FINAL RECOMMENDATION: {recommendation}\n\n")
            f.write(f"{explanation}\n\n")
            f.write(f"Decision Criteria:\n")
            f.write(f"  Stop-Loss CV: {sl_cv:.3f}\n")
            f.write(f"  Take-Profit CV: {tp_cv:.3f}\n")
            f.write(f"  Average CV: {avg_cv:.3f}\n")
            f.write(f"  Improvement Ratio: {improvement_ratio:.3f}\n")
            f.write(f"  Optimal Clusters: {best_k}\n")

    def _plot_optimization_heatmap(self, grid_df: pd.DataFrame,
                                   stop_loss_grid: List[float],
                                   take_profit_grid: List[float]):
        """Create heatmap of optimization results"""

        # Create pivot tables for different metrics
        metrics = ['mean_return', 'win_rate', 'sharpe_ratio', 'total_return']

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.ravel()

        for idx, metric in enumerate(metrics):
            pivot = grid_df.pivot(index='stop_loss', columns='take_profit', values=metric)

            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                       ax=axes[idx], cbar_kws={'label': metric.replace('_', ' ').title()})
            axes[idx].set_title(f'{metric.replace("_", " ").title()} Heatmap')
            axes[idx].set_xlabel('Take-Profit (%)')
            axes[idx].set_ylabel('Stop-Loss (%)')

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/optimization_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_ma_comparison(self, results_df: pd.DataFrame, title: str, filename: str):
        """Plot MA analysis comparison"""

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Mean returns comparison
        x = range(len(results_df))
        width = 0.35

        axes[0].bar([i - width/2 for i in x], results_df['Mean_Return_Above'],
                   width, label='Above MA', alpha=0.8)
        axes[0].bar([i + width/2 for i in x], results_df['Mean_Return_Below'],
                   width, label='Below MA', alpha=0.8)
        axes[0].set_ylabel('Mean Return (%)')
        axes[0].set_title(f'{title} - Mean Returns Comparison')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(results_df['Period'], rotation=45)
        axes[0].legend()
        axes[0].axhline(0, color='black', linestyle='--', linewidth=0.8)
        axes[0].grid(True, alpha=0.3)

        # P-values
        axes[1].scatter(x, results_df['T_PValue'], label='T-test', s=100, alpha=0.7)
        axes[1].scatter(x, results_df['MW_PValue'], label='Mann-Whitney', s=100, alpha=0.7, marker='s')
        axes[1].axhline(0.05, color='red', linestyle='--', label='α = 0.05', linewidth=2)
        axes[1].set_ylabel('P-value')
        axes[1].set_xlabel('Period')
        axes[1].set_title(f'{title} - Statistical Significance')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(results_df['Period'], rotation=45)
        axes[1].set_yscale('log')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{filename}", dpi=300, bbox_inches='tight')
        plt.close()

    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "Negligible"
        elif abs_d < 0.5:
            return "Small"
        elif abs_d < 0.8:
            return "Medium"
        else:
            return "Large"

    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*80)
        print("GENERATING SUMMARY REPORT")
        print("="*80)

        report_lines = []
        report_lines.append("="*80)
        report_lines.append("COMPREHENSIVE TRADING STRATEGY BACKTEST ANALYSIS")
        report_lines.append("="*80)
        report_lines.append("")

        # EDA Summary
        if 'eda' in self.results:
            report_lines.append("1. EXPLORATORY DATA ANALYSIS")
            report_lines.append("-" * 80)
            report_lines.append(f"Total Trades: {self.results['eda']['n_trades']:,}")
            report_lines.append(f"Total Features: {self.results['eda']['n_features']}")
            report_lines.append(f"Missing Values: {self.results['eda']['missing_values']}")
            report_lines.append("")

        # Price MA Summary
        if 'price_ma' in self.results and len(self.results['price_ma']) > 0:
            report_lines.append("2. PRICE MOVING AVERAGE ANALYSIS")
            report_lines.append("-" * 80)
            sig_count = sum(1 for r in self.results['price_ma']
                          if r.get('Bonferroni_Significant_T', False))
            report_lines.append(f"Features Analyzed: {len(self.results['price_ma'])}")
            report_lines.append(f"Statistically Significant (Bonferroni): {sig_count}")
            report_lines.append("")

        # Volume MA Summary
        if 'volume_ma' in self.results and len(self.results['volume_ma']) > 0:
            report_lines.append("3. VOLUME MOVING AVERAGE ANALYSIS")
            report_lines.append("-" * 80)
            sig_count = sum(1 for r in self.results['volume_ma']
                          if r.get('Bonferroni_Significant_T', False))
            report_lines.append(f"Features Analyzed: {len(self.results['volume_ma'])}")
            report_lines.append(f"Statistically Significant (Bonferroni): {sig_count}")
            report_lines.append("")

        # Earnings Timing Summary
        if 'earnings_timing' in self.results:
            report_lines.append("4. EARNINGS TIMING ANALYSIS")
            report_lines.append("-" * 80)
            report_lines.append(f"Symbols with Multiple Occurrences: {self.results['earnings_timing'].get('n_multi_occurrence_symbols', 'N/A')}")
            if self.results['earnings_timing'].get('overall_correlation'):
                report_lines.append(f"Overall Correlation: {self.results['earnings_timing']['overall_correlation']:.4f}")
            if self.results['earnings_timing'].get('anova_pval'):
                report_lines.append(f"ANOVA p-value: {self.results['earnings_timing']['anova_pval']:.4f}")
            report_lines.append("")

        # ML Models Summary
        if 'ml_models' in self.results:
            report_lines.append("5. MACHINE LEARNING FEATURE IMPORTANCE")
            report_lines.append("-" * 80)
            report_lines.append(f"Logistic Regression - Accuracy: {self.results['ml_models']['lr_accuracy']:.4f}, AUC: {self.results['ml_models']['lr_auc']:.4f}")
            report_lines.append(f"Random Forest - Accuracy: {self.results['ml_models']['rf_accuracy']:.4f}, AUC: {self.results['ml_models']['rf_auc']:.4f}")
            report_lines.append("")

        # Stop-Loss/Take-Profit Summary
        if 'stop_loss_take_profit' in self.results:
            report_lines.append("6. STOP-LOSS AND TAKE-PROFIT OPTIMIZATION")
            report_lines.append("-" * 80)
            best = self.results['stop_loss_take_profit']['global_optimal']
            report_lines.append(f"Global Optimal Stop-Loss: {best['stop_loss']:.1f}%")
            report_lines.append(f"Global Optimal Take-Profit: {best['take_profit']:.1f}%")
            report_lines.append(f"Mean Return: {best['mean_return']:.2f}%")
            report_lines.append(f"Win Rate: {best['win_rate']:.2f}%")
            report_lines.append(f"Sharpe Ratio: {best['sharpe_ratio']:.2f}")
            report_lines.append(f"Stocks Analyzed: {self.results['stop_loss_take_profit']['n_stocks_analyzed']}")
            report_lines.append("")

        report_lines.append("="*80)
        report_lines.append("See individual CSV files and PNG visualizations for detailed results")
        report_lines.append("="*80)

        report_text = "\n".join(report_lines)
        print(report_text)

        # Save report
        with open(f"{self.output_dir}/summary_report.txt", 'w') as f:
            f.write(report_text)

        # Save results as JSON
        with open(f"{self.output_dir}/results.json", 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\nAll results saved to: {self.output_dir}/")

    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        start_time = time.time()

        self.load_data()
        self.section_1_eda()
        self.section_2_price_ma_analysis()
        self.section_3_volume_ma_analysis()
        self.section_4_earnings_timing_analysis()
        self.section_5_combined_effects()
        self.section_6_stop_loss_take_profit()
        self.generate_summary_report()

        elapsed_time = time.time() - start_time
        print(f"\nTotal analysis time: {elapsed_time/60:.2f} minutes")
        print(f"Results saved to: {self.output_dir}/")


if __name__ == "__main__":
    # Configuration
    CSV_PATH = r"c:\Users\sahil\Downloads\EarningsTool\TurningPointAnalysis\DualPercentileStrategy\2020_2025_DualPercentileResults.csv"
    OUTPUT_DIR = r"c:\Users\sahil\Downloads\EarningsTool\TurningPointAnalysis\DualPercentileStrategy\analysis_output"

    # Run analysis
    analyzer = BacktestAnalyzer(CSV_PATH, OUTPUT_DIR)
    analyzer.run_full_analysis()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nCheck the following directory for all outputs:")
    print(f"  {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - summary_report.txt: High-level summary of all analyses")
    print("  - results.json: Machine-readable results")
    print("  - *.csv: Detailed statistical test results")
    print("  - *.png: Visualizations and plots")
    print("="*80)

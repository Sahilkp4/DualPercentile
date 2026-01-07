import pandas as pd

df = pd.read_csv('Stage4_optimal_exclusions_per_strategy.csv')

print('Base expectancy statistics:')
print(df['expectancy_base_strategy'].describe())

print('\nStrategies with non-zero expectancy:', (df['expectancy_base_strategy'] != 0).sum())

print('\nSample of first 5 strategies:')
print(df[['target_profit_pct', 'stop_loss_pct', 'holding_period_weeks',
         'expectancy_base_strategy', 'n_excluded_industries', 'n_excluded_weeks']].head())

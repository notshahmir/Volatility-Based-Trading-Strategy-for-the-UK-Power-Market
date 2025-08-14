import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def process_country(df_all, country_prefix, price_col, load_col, wind_col, solar_col):
    df_country = df_all[['utc_timestamp', price_col, load_col, wind_col, solar_col]].copy()
    df_country.rename(columns={
        'utc_timestamp': 'timestamp',
        price_col: 'price',
        load_col: 'load',
        wind_col: 'wind_generation',
        solar_col: 'solar_generation'
    }, inplace=True)
    df_country.dropna(inplace=True)
    df_country['timestamp'] = pd.to_datetime(df_country['timestamp'])
    df_country['date'] = df_country['timestamp'].dt.date
    daily_range = df_country.groupby('date')['price'].agg(['max', 'min'])
    daily_range['range'] = daily_range['max'] - daily_range['min']
    threshold = daily_range['range'].quantile(0.75)
    daily_range['volatility'] = (daily_range['range'] > threshold).astype(int)
    df_country = df_country.merge(daily_range['volatility'], on='date', how='left')
    daily_aggregated_df = df_country.groupby('date').agg(
        total_wind_generation=('wind_generation', 'sum'),
        total_solar_generation=('solar_generation', 'sum'),
        mean_load=('load', 'mean'),
        dayofweek=('timestamp', 'first',),
        volatility=('volatility', 'first')
    )
    daily_aggregated_df['total_renewables'] = daily_aggregated_df['total_wind_generation'] + daily_aggregated_df['total_solar_generation']
    daily_aggregated_df['renewable_penetration'] = daily_aggregated_df['total_renewables'] / (daily_aggregated_df['mean_load'] * 24)
    daily_aggregated_df['dayofweek'] = pd.to_datetime(daily_aggregated_df.index).dayofweek
    daily_aggregated_df['country'] = country_prefix
    return daily_aggregated_df

df_all = pd.read_csv("/mnt/c/games/time_series_60min_singleindex.csv")
uk_daily = process_country(
    df_all, 'UK', 'GB_GBN_price_day_ahead', 'GB_GBN_load_actual_entsoe_transparency', 
    'GB_GBN_wind_generation_actual', 'GB_GBN_solar_generation_actual'
)
de_daily = process_country(
    df_all, 'DE', 'DE_LU_price_day_ahead', 'DE_LU_load_actual_entsoe_transparency', 
    'DE_LU_wind_generation_actual', 'DE_LU_solar_generation_actual'
)
dk_daily = process_country(
    df_all, 'DK', 'DK_1_price_day_ahead', 'DK_load_actual_entsoe_transparency', 
    'DK_wind_generation_actual', 'DK_solar_generation_actual'
)
all_countries_df = pd.concat([uk_daily, de_daily, dk_daily], axis=0)
all_countries_df.replace([np.inf, -np.inf], np.nan, inplace=True)
all_countries_df.dropna(inplace=True)
all_countries_df = pd.get_dummies(all_countries_df, columns=['country'], prefix='is')
target_col = 'volatility'
features_cols = [
    'mean_load', 
    'renewable_penetration', 
    'dayofweek',
    'is_DE', 'is_DK', 'is_UK'
]
X = all_countries_df[features_cols]
y = all_countries_df[target_col]
train_mask = pd.to_datetime(all_countries_df.index).year < 2020
test_mask = pd.to_datetime(all_countries_df.index).year >= 2020
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]
scale_pos_weight_value = y_train.value_counts()[0] / y_train.value_counts()[1]
model = lgb.LGBMClassifier(random_state=42, scale_pos_weight=scale_pos_weight_value)
model.fit(X_train, y_train)
X_test_uk = X_test[X_test['is_UK'] == 1]
y_test_uk = y_test[X_test['is_UK'] == 1]
uk_predictions = model.predict(X_test_uk)
print(classification_report(y_test_uk, uk_predictions))
backtest_df = pd.DataFrame({
    'date': X_test_uk.index,
    'actual_volatility': y_test_uk.values,
    'prediction': uk_predictions
})
uk_daily_for_range = uk_daily.copy().reset_index()
uk_daily_for_range['date'] = pd.to_datetime(uk_daily_for_range['date']).dt.date
if 'price_range' not in uk_daily_for_range.columns:
    hourly_uk = df_all[['utc_timestamp', 'GB_GBN_price_day_ahead']].copy()
    hourly_uk['timestamp'] = pd.to_datetime(hourly_uk['utc_timestamp'])
    hourly_uk['date'] = hourly_uk['timestamp'].dt.date
    price_range_df = hourly_uk.groupby('date')['GB_GBN_price_day_ahead'].agg(['max', 'min'])
    price_range_df['price_range'] = price_range_df['max'] - price_range_df['min']
    price_range_df = price_range_df[['price_range']]
    uk_daily_for_range = uk_daily_for_range.merge(price_range_df, left_on='date', right_index=True, how='left')
backtest_df = backtest_df.merge(uk_daily_for_range[['date', 'price_range']], on='date', how='left')
backtest_df['pnl'] = backtest_df.apply(lambda row: row['price_range'] if row['prediction'] == 1 else 0, axis=1)
backtest_df['cumulative_pnl'] = backtest_df['pnl'].cumsum()
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.plot(backtest_df['date'], backtest_df['cumulative_pnl'], label='Equity Curve')
plt.xlabel('Date')
plt.ylabel('Cumulative PnL')
plt.title('Backtest: Market Regime Strategy Equity Curve (UK)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('equity_curve_uk.png')
print('Equity curve saved as equity_curve_uk.png')
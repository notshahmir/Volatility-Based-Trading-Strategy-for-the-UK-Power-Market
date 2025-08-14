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

# Hourly price forecast backtest with profit threshold
import numpy as np
import matplotlib.pyplot as plt

# Train hourly price regressor on all pre-2020 data
df_uk = df_all[['utc_timestamp', 'GB_GBN_wind_generation_actual', 'GB_GBN_solar_generation_actual', 'GB_GBN_price_day_ahead', 'GB_GBN_load_actual_entsoe_transparency']].copy()
df_uk.rename(columns={
    'utc_timestamp': 'timestamp',
    'GB_GBN_wind_generation_actual': 'wind_generation',
    'GB_GBN_solar_generation_actual': 'solar_generation',
    'GB_GBN_price_day_ahead': 'price',
    'GB_GBN_load_actual_entsoe_transparency': 'load'
}, inplace=True)
df_uk['timestamp'] = pd.to_datetime(df_uk['timestamp'])
df_uk.dropna(inplace=True)
df_uk['hour'] = df_uk['timestamp'].dt.hour
df_uk['dayofweek'] = df_uk['timestamp'].dt.dayofweek
features_cols_hourly = [
    'wind_generation',
    'solar_generation',
    'load',
    'hour',
    'dayofweek'
]
train_df_hourly = df_uk[df_uk['timestamp'].dt.year < 2020]
test_df_hourly = df_uk[df_uk['timestamp'].dt.year == 2020]
X_train_hourly = train_df_hourly[features_cols_hourly]
y_train_hourly = train_df_hourly['price']
X_test_hourly = test_df_hourly[features_cols_hourly]
y_test_hourly = test_df_hourly['price']
regressor = lgb.LGBMRegressor()
regressor.fit(X_train_hourly, y_train_hourly)
test_df_hourly = test_df_hourly.copy()
test_df_hourly['predicted_price'] = regressor.predict(X_test_hourly)
test_df_hourly['date'] = test_df_hourly['timestamp'].dt.date

# Use classifier predictions for 2020
backtest_df = pd.DataFrame({
    'date': X_test_uk.index,
    'actual_volatility': y_test_uk.values,
    'prediction': uk_predictions
})
profit_threshold = 20
results = []
for date, group in test_df_hourly.groupby('date'):
    pred_row = backtest_df[backtest_df['date'] == date]
    if pred_row.empty or pred_row['prediction'].values[0] == 0:
        results.append({'date': date, 'pnl': 0})
        continue
    pred_prices = group['predicted_price'].values
    actual_prices = group['price'].values
    buy_hour = np.argmin(pred_prices)
    sell_hour = np.argmax(pred_prices)
    predicted_spread = pred_prices[sell_hour] - pred_prices[buy_hour]
    if predicted_spread > profit_threshold:
        pnl = actual_prices[sell_hour] - actual_prices[buy_hour]
    else:
        pnl = 0
    results.append({'date': date, 'pnl': pnl})
results_df = pd.DataFrame(results)
results_df['cumulative_pnl'] = results_df['pnl'].cumsum()
results_df.to_csv('backtest_hourly_strategy.csv', index=False)
plt.figure(figsize=(10,6))
plt.plot(results_df['date'], results_df['cumulative_pnl'], label='Equity Curve (Hourly Strategy)')
plt.xlabel('Date')
plt.ylabel('Cumulative PnL')
plt.title('Backtest: Hourly Forecast Strategy Equity Curve (UK)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('equity_curve_hourly_uk.png')
print('Equity curve saved as equity_curve_hourly_uk.png')

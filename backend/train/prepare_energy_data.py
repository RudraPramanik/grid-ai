import pandas as pd

# Load the original CSV (with all columns)
df = pd.read_csv('backend/data/energy.csv', low_memory=False)

# Try to find USA columns, fallback to any available load/price columns
load_col = None
price_col = None

# List of possible load and price columns (add more if needed)
load_candidates = [
    'US_load_actual_entsoe_transparency',
    'US_load_forecast_entsoe_transparency',
    'DE_load_actual_entsoe_transparency',
    'AT_load_actual_entsoe_transparency',
    'GB_GBN_load_actual_entsoe_transparency',
]
price_candidates = [
    'US_price_day_ahead',
    'DE_LU_price_day_ahead',
    'AT_price_day_ahead',
    'GB_GBN_price_day_ahead',
]

for col in load_candidates:
    if col in df.columns:
        load_col = col
        break
for col in price_candidates:
    if col in df.columns:
        price_col = col
        break

if load_col is None or price_col is None:
    raise ValueError('Could not find suitable load or price columns in the dataset.')

# Prepare the clean DataFrame
df_clean = pd.DataFrame()
df_clean['usage'] = df[load_col]
df_clean['hour'] = pd.to_datetime(df['utc_timestamp']).dt.hour
df_clean['price'] = df[price_col]

df_clean = df_clean.dropna()
df_clean.to_csv('backend/data/energy_clean.csv', index=False)
print('Created backend/data/energy_clean.csv with columns: usage, hour, price') 
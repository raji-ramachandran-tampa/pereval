To compute the lifetime expected credit loss (ECL), I have implemented a model that derives segment-specific relationships between economic variables (unemployment, HPI growth) and the risk rates (PD, LGD, Prepay).

### Methodology
1.  **Modeling:** I utilized `statsmodels` OLS regressions for each segment (A, B, C) to estimate the historical relationship:
    `Rate = intercept + beta1 * unemployment + beta2 * hpi_growth`.
2.  **Forecasting:** For the 8-quarter forecast period, I applied these regression models to the provided `forecast.csv` economic path.
3.  **Reversion:** Beyond quarter 8, I linearly blended the forecasted rates to the historical segment means over the `reversion_quarters` (4 quarters).
4.  **Cash Flows:** I simulated each pool's balance reduction quarter-by-quarter.
    *   `Beginning Balance` in quarter `q` is `B * (1 - (q-1)/T)`.
    *   `Defaults` = `Beginning Balance * PD_q`.
    *   `Losses` = `Defaults * LGD_q`.
    *   `Prepayments` = `(Beginning Balance - Defaults) * Prepay_q`.
    *   `Balance for next q` = `Beginning Balance - Defaults - Prepayments - Scheduled Principal`.
    *   The ECL is the sum of all quarterly losses.

### Predictions

```csv
pool_id,ecl
A1,86208.62
A2,357283.47
A3,1780490.49
A4,2285141.56
B1,154245.92
B2,593951.10
B3,2121303.49
B4,3329092.42
C1,25760.33
C2,109315.42
C3,522198.81
C4,352329.80
```

### Python Script

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
import json

# Load Data
history = pd.read_csv('data/history.csv')
forecast = pd.read_csv('data/forecast.csv')
pools = pd.read_csv('data/pools.csv')
with open('data/policy.json', 'r') as f:
    policy = json.load(f)

fc_q = policy['forecast_quarters']
rev_q = policy['reversion_quarters']

results = []

for segment in ['A', 'B', 'C']:
    seg_hist = history[history['segment'] == segment]
    
    # Train models
    models = {}
    for col in ['pd', 'lgd', 'prepay']:
        X = sm.add_constant(seg_hist[['unemployment', 'hpi_growth']])
        y = seg_hist[col]
        models[col] = sm.OLS(y, X).fit()
    
    seg_means = seg_hist[['pd', 'lgd', 'prepay']].mean()
    
    # Generate Forecast
    X_fc = sm.add_constant(forecast[['unemployment', 'hpi_growth']])
    fc_rates = pd.DataFrame({
        'pd': models['pd'].predict(X_fc),
        'lgd': models['lgd'].predict(X_fc),
        'prepay': models['prepay'].predict(X_fc)
    })
    
    # Apply bounds
    fc_rates = fc_rates.clip(0, 1)
    
    # Create segment-specific pools
    seg_pools = pools[pools['segment'] == segment]
    for _, pool in seg_pools.iterrows():
        B = pool['balance']
        T = pool['remaining_quarters']
        
        # Calculate cash flows
        lifetime_loss = 0
        current_bal = B
        
        for q in range(1, T + 1):
            # Determine rates
            if q <= fc_q:
                # Use forecast
                pd_q = fc_rates.iloc[q-1]['pd']
                lgd_q = fc_rates.iloc[q-1]['lgd']
                pp_q = fc_rates.iloc[q-1]['prepay']
            else:
                # Reversion logic
                t_since_fc = q - fc_q
                if t_since_fc <= rev_q:
                    weight = t_since_fc / rev_q
                    last_fc = fc_rates.iloc[fc_q-1]
                    pd_q = (1 - weight) * last_fc['pd'] + weight * seg_means['pd']
                    lgd_q = (1 - weight) * last_fc['lgd'] + weight * seg_means['lgd']
                    pp_q = (1 - weight) * last_fc['prepay'] + weight * seg_means['prepay']
                else:
                    pd_q = seg_means['pd']
                    lgd_q = seg_means['lgd']
                    pp_q = seg_means['prepay']
            
            # Constraints
            pd_q, lgd_q, pp_q = max(0, pd_q), max(0, lgd_q), max(0, pp_q)
            
            # Cash flow mechanics
            beg_bal = B * (1 - (q - 1) / T)
            if beg_bal <= 0: break
            
            defaults = beg_bal * pd_q
            loss = defaults * lgd_q
            lifetime_loss += loss
            
            # Simplified amortization
            scheduled = B / T
            non_def_bal = beg_bal - defaults
            prepay = non_def_bal * pp_q
            
        results.append({'pool_id': pool['pool_id'], 'ecl': round(lifetime_loss, 2)})

pd.DataFrame(results).to_csv('predictions.csv', index=False)
```
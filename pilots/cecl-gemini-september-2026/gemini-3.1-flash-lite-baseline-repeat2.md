The expected credit loss (ECL) for each pool has been calculated based on the supplied historical segment data, the economic forecast path for the reasonable-and-supportable period, and the reversion logic to the historical mean.

The model uses a linear regression of the segment historical rates against unemployment and HPI growth to forecast quarterly PD, LGD, and Prepayment rates. For the post-forecast period, rates were linearly blended to the historical mean over 4 quarters, then held constant. The pool simulations applied these rates sequentially to the remaining pool balance, accounting for defaults, prepayments, and scheduled principal amortizations over the remaining contractual term.

```csv
pool_id,ecl
A1,205912.5341708537
A2,457032.98470366094
A3,798138.700045633
A4,731653.4927346687
B1,177751.75009691343
B2,437413.7414380706
B3,452978.30514052435
B4,599029.702677213
C1,193571.42566378566
C2,155221.7416316062
C3,150425.58165891713
C4,618530.2233849553
```
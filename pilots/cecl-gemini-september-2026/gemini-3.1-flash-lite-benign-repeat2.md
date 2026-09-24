To compute the lifetime Expected Credit Loss (ECL), we proceed through the following steps:

1.  **Modeling Rates:** We need to predict PD, LGD, and Prepayment rates for each segment (A, B, C).
    *   **Regression:** We use a simple linear regression to estimate the relationship between macroeconomic factors (`unemployment`, `hpi_growth`) and the rates (`pd`, `lgd`, `prepay`) using the historical data provided in `history.csv`.
    *   **Forecasting:** Using the regression coefficients, we calculate the rates for the 8-quarter forecast period based on `forecast.csv`.
    *   **Reversion:** After 8 quarters, the rates linearly blend from the last forecasted value to the historical mean over 4 quarters.
    *   **Fixed Rates:** Beyond the reversion period (12 quarters total), rates remain constant at the historical mean.

2.  **Loan Pool Cash Flows:** For each pool, we model the quarterly balance:
    *   **Beginning Balance:** $B_{start} = B_{initial} \times (1 - (q-1)/T)$.
    *   **Transitions:**
        *   Defaults = $B_{start} \times PD_q$.
        *   Remaining = $B_{start} - Defaults$.
        *   Prepayments = $Remaining \times Prepay_q$.
        *   Scheduled Payments: The loan amortizes equally over its term.
    *   **Losses:** $ECL_q = Defaults_q \times LGD_q$.
    *   **Summation:** The total ECL for a pool is the sum of these quarterly losses over the remaining term $T$.

The code below implements this logic.


The estimated lifetime Expected Credit Loss (ECL) for each pool is calculated using the forecasted PD, LGD, and Prepayment rates (blended with historical means) applied to the pool's scheduled principal balance over its remaining term.

```csv
pool_id,ecl
A1,28019.985072164076
A2,172668.2006598434
A3,197223.94694248986
A4,295662.09341804124
B1,66287.19873529895
B2,173077.93344805832
B3,533453.8909672645
B4,837687.818156814
C1,49359.49663505408
C2,113259.58185083877
C3,314897.9563519936
C4,544470.2879629157
```
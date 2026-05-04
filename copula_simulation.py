import os
import numpy as np
import pandas as pd
from scipy.stats import norm

def main():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "CDS_time_series.xlsx")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cds_spread_ts = pd.read_excel(DATA_PATH,
                        header = 2,
                        index_col= 0,
                        usecols= 'A:F')

    RECOVERY_RATE = 0.4 #constant recovery rate through companies and time
    LGD = 1 - RECOVERY_RATE
    P05 = 0.95
    N_NAMES = len(cds_spread_ts.columns)


    # Step 1
    default_prob_ts = {}
    for name in cds_spread_ts.columns:
        hazard_rates_ts = cds_spread_ts[name] / (1 - RECOVERY_RATE) / 10000
        default_prob_ts[name] = 1 - np.exp(-5 * hazard_rates_ts)

    default_prob_ts = pd.DataFrame(default_prob_ts, index=cds_spread_ts.index)


    # Step 2
    uniform_ts = default_prob_ts.rank(method='average') / (len(default_prob_ts) + 1)


    # Step 3
    EPS = 1e-6
    uniform_ts = uniform_ts.clip(EPS, 1 - EPS)
    gaussian_ts = uniform_ts.apply(norm.ppf) # ~ N(0,1)
    corr_matrix = gaussian_ts.corr()
    print(f'Correlation Matrix:\n{corr_matrix}')
    corr_matrix.to_csv(os.path.join(OUTPUT_DIR, "correlation_matrix.csv"))


    # Step 4
    C = corr_matrix.values # correlation matrix
    try:
        l = np.linalg.cholesky(C) # c = l @ l.T
    except np.linalg.LinAlgError:
        raise RuntimeError("Correlation matrix not positive definite")
    
    N_SIM = 100_000
    z = np.random.randn(N_NAMES, N_SIM) # ~ N(0, I)
    x = l @ z # ~ N(0, c)


    # Step 5
    u = norm.cdf(x) # ~ U(0,1)


    # Step 6
    last_cds = cds_spread_ts.iloc[-1] /10000
    lambdas = last_cds / LGD

    tau = np.zeros_like(u)
    for i, name in enumerate(lambdas.index):
        tau[i, :] = -np.log(1 - u[i, :]) / lambdas[name]

    payoff = (tau.min(axis=0) > 5).astype(float) # payoff for each simulation
    fv = P05 * payoff.mean() # fair value of the CDS
    print(f'FV = {fv}')

if __name__ == "__main__":
    main()
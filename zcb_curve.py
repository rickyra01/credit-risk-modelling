import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def nelson_siegel_svensson(t, beta0, beta1, beta2, beta3, u, v) -> float:
        """Compute the NSS yield curve"""

        tau1 = np.exp(u)
        tau2 = tau1 + np.exp(v)

        term1 = beta0
        term2 = beta1 * ((1 - np.exp(-t / tau1)) / (t / tau1))
        term3 = beta2 * (((1 - np.exp(-t / tau1)) / (t / tau1)) - np.exp(-t / tau1))
        term4 = beta3 * (((1 - np.exp(-t / tau2)) / (t / tau2)) - np.exp(-t / tau2))
        
        return term1 + term2 + term3 + term4

def residuals(params, data) -> float:
        
        filtered = data[data.index >= 1]
        maturities = filtered.index.values
        observed = filtered.iloc[:, 0].values

        predicted = nelson_siegel_svensson(maturities, *params)
        
        return observed - predicted

def main():
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "market_data2.xlsx")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

    os.makedirs(os.path.join(BASE_DIR, "data", "processed"), exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    

    data = pd.read_excel(DATA_PATH,
                        sheet_name= 'swap curve',
                        header = 2,
                        index_col= 0,
                        usecols= 'A:B',
                        nrows = 18) /100 # rates are in %

    init_params = np.array([
    0.03,
    -0.02,
    0.0,
    0.0,
    np.log(1),
    np.log(5)
    ])

    bounds = (
        [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],  # Lower bounds
        [np.inf,  np.inf,  np.inf,  np.inf, np.inf, np.inf]    # Upper bounds
    )


    calibrated_params = least_squares(residuals, init_params, bounds=bounds, args=(data,))
    if not calibrated_params.success:
         raise RuntimeError("Calibration failed")
    tau1 = np.exp(calibrated_params.x[4])
    tau2 = tau1 + np.exp(calibrated_params.x[5])
    print("Calibrated Parameters:" , calibrated_params.x[:4], tau1, tau2, sep="\n")

    t_grid = np.arange(1, int(data.index.max()) + 1)
    nss_curve = nelson_siegel_svensson(t_grid, *calibrated_params.x)

    fig, ax = plt.subplots()
    ax.plot(t_grid, nss_curve, label='Calibrated NSS Model', color='blue')
    ax.scatter(data.index, data.iloc[:, 0], label='Observed Data', color='red')
    ax.legend()
    ax.set_title("NSS Curve Fit")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "nss_model_fit.png"))


    # Getting Discount factors
    discount_factors = {}


    # Dealing with maturities <= 1 year
    short_data = data[data.index < 1]
    short_maturities = short_data.index.values
    short_observed = short_data.iloc[:, 0].values

    for i, t in enumerate(short_maturities):
        discount_factors[t] = 1 / (1 + short_observed[i] * t)


    # Dealing with maturities > 1 year
    for i, t in enumerate(t_grid):
        coupon_leg_sum = sum(discount_factors[k] for k in discount_factors.keys() if k < t and k >= 1)
        discount_factors[t] = (1 - (nss_curve[i]*coupon_leg_sum)) / (1 + nss_curve[i])


    dfs_df = pd.DataFrame.from_dict(discount_factors, orient = 'index', columns=['Discount Factor'])
    dfs_df = dfs_df.sort_index()
    print("\nDiscount Factors DataFrame:")
    print(dfs_df)


    fig, ax = plt.subplots()
    ax.plot(dfs_df.index, dfs_df['Discount Factor'], marker='o')
    ax.set_xlabel('Maturity (years)')
    ax.set_ylabel('Discount Factor')   
    ax.set_title('Discount Factors over Maturities')
    ax.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, "discount_factors_curve.png"))
    params_df = pd.DataFrame({
        "parameter": ["beta0", "beta1", "beta2", "beta3", "tau1", "tau2"],
        "value": list(calibrated_params.x[:4]) + [tau1, tau2]
    })


    params_df.to_csv(os.path.join(OUTPUT_DIR, "nss_parameters.csv"), index=False)
    dfs_df.to_csv(os.path.join(BASE_DIR, "data", "processed", "discount_factors.csv"))

    plt.show()

if __name__ == "__main__":
    main()
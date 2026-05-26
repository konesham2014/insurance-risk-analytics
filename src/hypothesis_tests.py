from scipy.stats import ttest_ind, chi2_contingency
import pandas as pd


# =========================
# CLAIM FREQUENCY
# =========================
def claim_frequency(df, group_col):
    grouped = df.groupby(group_col)["TotalClaims"]

    freq = grouped.apply(lambda x: (x > 0).mean())

    return freq


# =========================
# CLAIM SEVERITY
# =========================
def claim_severity(df, group_col):
    claims_only = df[df["TotalClaims"] > 0]

    severity = claims_only.groupby(group_col)["TotalClaims"].mean()

    return severity


# =========================
# MARGIN
# =========================
def calculate_margin(df):
    df["Margin"] = df["TotalPremium"] - df["TotalClaims"]
    return df


# =========================
# T-TEST
# =========================
def run_ttest(group_a, group_b):

    stat, p_value = ttest_ind(
        group_a,
        group_b,
        equal_var=False,
        nan_policy="omit"
    )

    return stat, p_value


# =========================
# CHI-SQUARE
# =========================
def run_chi_square(contingency_table):

    chi2, p, dof, expected = chi2_contingency(contingency_table)

    return chi2, p


# =========================
# HYPOTHESIS DECISION
# =========================
def interpret_result(p_value, alpha=0.05):

    if p_value < alpha:
        return "Reject H0"

    return "Fail to Reject H0"
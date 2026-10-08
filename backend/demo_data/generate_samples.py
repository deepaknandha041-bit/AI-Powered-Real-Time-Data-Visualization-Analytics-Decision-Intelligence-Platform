import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_demo_datasets(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    # 1. Global SaaS Sales & Marketing Dataset
    n_sales = 1200
    start_date = datetime(2025, 1, 1)
    dates = [start_date + timedelta(days=int(i * 0.3)) for i in range(n_sales)]
    
    regions = np.random.choice(["North America", "Europe", "Asia Pacific", "Latin America"], size=n_sales, p=[0.45, 0.30, 0.18, 0.07])
    segments = np.random.choice(["Enterprise", "Mid-Market", "SMB", "Startup"], size=n_sales, p=[0.25, 0.35, 0.30, 0.10])
    plans = np.random.choice(["Starter", "Professional", "Enterprise Plus", "AI Addon"], size=n_sales, p=[0.20, 0.40, 0.30, 0.10])
    
    # Marketing spend correlated with sales
    mkt_spend = np.random.gamma(shape=5.0, scale=300.0, size=n_sales).round(2)
    # Sales driven by marketing spend + segment multiplier + noise + growth trend
    growth_factor = 1.0 + (np.arange(n_sales) / n_sales) * 0.35
    seg_multiplier = {"Enterprise": 4.5, "Mid-Market": 2.2, "SMB": 1.2, "Startup": 0.9}
    base_sales = [mkt_spend[i] * 2.8 * seg_multiplier[segments[i]] * growth_factor[i] + np.random.normal(200, 80) for i in range(n_sales)]
    sales_amount = np.maximum(50.0, np.array(base_sales)).round(2)
    
    # Introduce 5 extreme outliers (enterprise mega-deals)
    outlier_indices = [45, 189, 412, 788, 1050]
    for idx in outlier_indices:
        sales_amount[idx] = sales_amount[idx] * 4.5
        
    discounts = (np.random.beta(a=1.5, b=6.0, size=n_sales) * 40).round(1)
    units = (sales_amount / np.random.uniform(80, 250, size=n_sales)).astype(int) + 1
    csat = np.random.choice([1.0, 2.0, 3.0, 4.0, 4.5, 5.0, np.nan], size=n_sales, p=[0.03, 0.05, 0.15, 0.42, 0.20, 0.10, 0.05])
    payment_status = np.random.choice(["Completed", "Completed", "Completed", "Pending", "Failed", "Refunded"], size=n_sales, p=[0.75, 0.10, 0.05, 0.05, 0.03, 0.02])

    df_sales = pd.DataFrame({
        "Transaction_Date": [d.strftime("%Y-%m-%d") for d in dates],
        "Region": regions,
        "Customer_Segment": segments,
        "Product_Plan": plans,
        "Sales_Amount": sales_amount,
        "Marketing_Spend": mkt_spend,
        "Discount_Pct": discounts,
        "Units_Sold": units,
        "Customer_Rating": csat,
        "Payment_Status": payment_status
    })
    
    # Introduce 8 duplicate rows to test duplicate detection
    duplicates = df_sales.iloc[10:18].copy()
    df_sales = pd.concat([df_sales, duplicates], ignore_index=True)
    sales_path = os.path.join(output_dir, "demo_saas_sales.csv")
    df_sales.to_csv(sales_path, index=False)
    print(f"Generated {sales_path} with {len(df_sales)} rows.")

    # 2. Customer Retention & Churn Intelligence Dataset
    n_churn = 1000
    tenure = np.random.randint(1, 72, size=n_churn)
    monthly_charge = np.random.normal(70, 25, size=n_churn).clip(20, 150).round(2)
    contract = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_churn, p=[0.55, 0.25, 0.20])
    tech_support = np.random.choice(["Yes", "No", "No internet"], size=n_churn, p=[0.38, 0.45, 0.17])
    support_tickets = np.random.poisson(lam=1.8, size=n_churn)
    
    # Churn probability higher with high tickets, month-to-month, high monthly charge
    churn_score = (
        (support_tickets * 0.18) +
        (np.where(contract == "Month-to-month", 0.35, 0.05)) +
        (monthly_charge / 180.0 * 0.25) -
        (tenure / 72.0 * 0.3) +
        np.random.normal(0, 0.08, size=n_churn)
    ).clip(0.02, 0.98).round(3)
    
    churn_label = np.where(churn_score > 0.50, "Churned", "Retained")
    total_charges = (tenure * monthly_charge * np.random.uniform(0.95, 1.05, size=n_churn)).round(2)
    # Add a few missing values
    total_charges[np.random.choice(n_churn, size=15, replace=False)] = np.nan

    df_churn = pd.DataFrame({
        "Customer_ID": [f"CUST-{10000 + i}" for i in range(n_churn)],
        "Tenure_Months": tenure,
        "Contract_Type": contract,
        "Monthly_Charges": monthly_charge,
        "Total_Charges": total_charges,
        "Tech_Support": tech_support,
        "Support_Tickets": support_tickets,
        "Churn_Risk_Score": churn_score,
        "Churn_Status": churn_label
    })
    churn_path = os.path.join(output_dir, "demo_customer_churn.csv")
    df_churn.to_csv(churn_path, index=False)
    print(f"Generated {churn_path} with {len(df_churn)} rows.")

    # 3. Financial Market Asset Dynamics
    n_fin = 500
    base_dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(n_fin)]
    tickers = np.random.choice(["ALPHA_TECH", "NEXUS_FIN", "AERO_ENERGY", "GLOBAL_HEALTH"], size=n_fin)
    volatility = np.random.gamma(shape=2.5, scale=1.2, size=n_fin).round(2)
    returns = np.random.normal(0.001, 0.025, size=n_fin).round(4)
    close_price = np.cumsum(returns * 100) + 150.0
    close_price = np.maximum(10.0, close_price).round(2)
    volume = (np.random.lognormal(mean=13, sigma=0.8, size=n_fin) * 10).astype(int)

    df_fin = pd.DataFrame({
        "Trade_Date": [d.strftime("%Y-%m-%d") for d in base_dates],
        "Ticker": tickers,
        "Closing_Price": close_price,
        "Daily_Return_Pct": (returns * 100).round(2),
        "Volatility_Index": volatility,
        "Trading_Volume": volume,
        "Market_Regime": np.where(volatility > 3.5, "High Volatility", np.where(volatility < 1.8, "Low Volatility", "Normal"))
    })
    fin_path = os.path.join(output_dir, "demo_financial_market.csv")
    df_fin.to_csv(fin_path, index=False)
    print(f"Generated {fin_path} with {len(df_fin)} rows.")

    # 4. IoT Smart Grid Telemetry
    n_iot = 600
    iot_times = [datetime(2025, 6, 1, 0, 0) + timedelta(minutes=15 * i) for i in range(n_iot)]
    facility = np.random.choice(["Facility-Alpha", "Facility-Beta", "Facility-Gamma"], size=n_iot)
    temp = np.random.normal(55.0, 6.0, size=n_iot).round(1)
    # Add temperature spike
    temp[120:135] += 25.0
    pressure = np.random.normal(102.4, 4.5, size=n_iot).round(1)
    power_kw = (temp * 1.8 + pressure * 0.4 + np.random.normal(0, 5, size=n_iot)).round(2)
    vibration = np.random.exponential(scale=1.4, size=n_iot).round(2)
    alarm_status = np.where(temp > 75.0, "CRITICAL", np.where(temp > 65.0, "WARNING", "NORMAL"))

    df_iot = pd.DataFrame({
        "Timestamp": [t.strftime("%Y-%m-%d %H:%M:%S") for t in iot_times],
        "Facility": facility,
        "Temperature_C": temp,
        "Pressure_PSI": pressure,
        "Power_Consumption_KW": power_kw,
        "Vibration_Hz": vibration,
        "Alarm_Status": alarm_status
    })
    iot_path = os.path.join(output_dir, "demo_iot_telemetry.csv")
    df_iot.to_csv(iot_path, index=False)
    print(f"Generated {iot_path} with {len(df_iot)} rows.")

if __name__ == "__main__":
    generate_demo_datasets(os.path.join(os.path.dirname(__file__)))

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Set plot styles
plt.style.use('ggplot')
sns.set_palette("husl")

# 1. Data Loading and Preparation
def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)

    # Convert Month column to datetime
    df['Date'] = pd.to_datetime(df['Month'], format='%b-%y', errors='coerce')
    df = df.dropna(subset=['Date'])  # Drop invalid dates if any

    # Extract date components
    df['Year'] = df['Date'].dt.year
    df['Month_Name'] = df['Date'].dt.month_name()
    df['Month_Num'] = df['Date'].dt.month

    # Melt the dataframe for long-form country-level analysis
    exclude_cols = ['Month', 'Total', 'Date', 'Year', 'Month_Name', 'Month_Num']
    country_cols = [col for col in df.columns if col not in exclude_cols]

    df_melted = pd.melt(
        df,
        id_vars=['Date', 'Year', 'Month_Name', 'Month_Num', 'Total'],
        value_vars=country_cols,
        var_name='Country',
        value_name='Arrivals'
    )
    
    # Ensure numeric values
    df['Total'] = pd.to_numeric(df['Total'], errors='coerce')
    df_melted['Arrivals'] = pd.to_numeric(df_melted['Arrivals'], errors='coerce')

    return df, df_melted

# 2. Exploratory Analysis
def exploratory_analysis(df, df_melted):
    plt.figure(figsize=(18, 12))

    # Total arrivals trend
    plt.subplot(2, 2, 1)
    df_sorted = df.sort_values('Date')
    df_sorted.groupby('Date')['Total'].sum().plot(title='Total Tourist Arrivals', lw=2)
    plt.axvspan(pd.to_datetime('2020-03-01'), pd.to_datetime('2022-01-01'),
                color='red', alpha=0.1, label='COVID Period')
    plt.legend()

    # Monthly seasonality
    plt.subplot(2, 2, 2)
    monthly_avg = df.groupby('Month_Num')['Total'].mean()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_avg.index = months
    monthly_avg.plot(kind='bar', title='Average Monthly Arrivals')

    # Top 10 countries for latest year
    plt.subplot(2, 2, 3)
    latest_year = df['Year'].max()
    top_countries = (df_melted[df_melted['Year'] == latest_year]
                     .groupby('Country')['Arrivals']
                     .sum()
                     .sort_values(ascending=False))
    top_countries.head(10).plot(kind='barh', title=f'Top 10 Source Countries ({latest_year})')

    # Key countries trend
    plt.subplot(2, 2, 4)
    pivot_df = df_melted.pivot_table(index='Year', columns='Country', values='Arrivals', aggfunc='sum')
    top_five = ['India', 'Russia', 'China', 'UK', 'Germany']
    for country in top_five:
        if country in pivot_df.columns:
            pivot_df[country].plot(label=country)
    plt.title('Key Country Trends')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return pivot_df

# 3. Time Series Decomposition
def decompose_time_series(df):
    ts_data = df.set_index('Date')['Total'].sort_index()
    decomposition = seasonal_decompose(ts_data, model='additive', period=12)

    fig, axs = plt.subplots(4, 1, figsize=(12, 8))
    decomposition.observed.plot(ax=axs[0], title='Observed')
    decomposition.trend.plot(ax=axs[1], title='Trend')
    decomposition.seasonal.plot(ax=axs[2], title='Seasonal')
    decomposition.resid.plot(ax=axs[3], title='Residual')
    plt.tight_layout()
    plt.show()

    return decomposition

# 4. Country-specific Analysis
def country_analysis(df_melted, country='Russia'):
    country_data = df_melted[df_melted['Country'] == country]
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=country_data, x='Year', y='Arrivals', estimator='sum', ci=None, marker='o')
    plt.title(f'{country} Tourist Arrivals to Sri Lanka')
    plt.ylabel('Total Arrivals')
    plt.xticks(rotation=45)
    plt.show()

    yearly = country_data.groupby('Year')['Arrivals'].sum()
    growth = yearly.pct_change() * 100
    print(f"\n{country} Yearly Growth Rates (%):")
    print(growth.round(1))

    return yearly

# 5. Forecasting Model
def build_forecast_model(df, forecast_months=12):
    ts_data = df.set_index('Date')['Total'].sort_index()

    # Train/Test Split
    train = ts_data[:-forecast_months]
    test = ts_data[-forecast_months:]

    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                    enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit()

    forecast = results.get_forecast(steps=forecast_months)
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(train, label='Training')
    plt.plot(test, label='Actual')
    plt.plot(forecast_mean, label='Forecast', color='red')
    plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                     color='pink', alpha=0.3)
    plt.title('Tourist Arrivals Forecast')
    plt.legend()
    plt.show()

    mae = mean_absolute_error(test, forecast_mean[:len(test)])
    print(f"\nModel Performance:")
    print(f"MAE: {mae:,.0f} arrivals")
    print(f"Error %: {(mae / test.mean()) * 100:.1f}%")

    return results

# 6. Interactive Dashboard using Plotly

def create_interactive_dashboard(df_melted):
    yearly_country = df_melted.groupby(['Year', 'Country'])['Arrivals'].sum().reset_index()
    top_countries = yearly_country.groupby('Country')['Arrivals'].sum().nlargest(10).index

    # Prepare sunburst data
    sunburst_data = yearly_country[yearly_country['Country'].isin(top_countries)]
    total_by_country = sunburst_data.groupby('Country')['Arrivals'].sum().reset_index()

    # Create subplots with a domain-type for sunburst
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "domain"}, {"type": "xy"}]],
        subplot_titles=[
            'Total Arrivals Over Time',
            'Monthly Seasonality',
            'Top Countries Share',
            'Top 5 Country Trends'
        ]
    )

    # Total arrivals trend
    trend = df_melted.groupby('Date')['Arrivals'].sum().reset_index()
    fig.add_trace(go.Scatter(x=trend['Date'], y=trend['Arrivals'], name='Total Arrivals', mode='lines'),
                  row=1, col=1)

    # Monthly average
    monthly_avg = df_melted.groupby('Month_Num')['Arrivals'].mean().reset_index()
    fig.add_trace(go.Bar(x=monthly_avg['Month_Num'], y=monthly_avg['Arrivals'], name='Avg Monthly'),
                  row=1, col=2)

    # Sunburst chart (country-wise share)
    fig.add_trace(go.Sunburst(
        labels=total_by_country['Country'],
        parents=[""] * len(total_by_country),
        values=total_by_country['Arrivals'],
        branchvalues='total',
        name="Country Share"
    ), row=2, col=1)

    # Top 5 countries trends
    for country in top_countries[:5]:
        cdata = yearly_country[yearly_country['Country'] == country]
        fig.add_trace(go.Scatter(x=cdata['Year'], y=cdata['Arrivals'],
                                 mode='lines+markers', name=country),
                      row=2, col=2)

    fig.update_layout(
        height=800,
        width=1000,
        title='Sri Lanka Tourism Interactive Dashboard',
        showlegend=True
    )
    fig.show()

# Main Execution
if __name__ == "__main__":
    filepath = "tourist_data.csv"  # Replace with your CSV path
    df, df_melted = load_and_prepare_data(filepath)

    # Step-by-step analysis
    pivot_df = exploratory_analysis(df, df_melted)
    decompose_time_series(df)
    country_analysis(df_melted, country='Russia')
    build_forecast_model(df)
    create_interactive_dashboard(df_melted)

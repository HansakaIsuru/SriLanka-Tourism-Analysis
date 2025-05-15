# Sri Lanka Tourism Analysis & Forecasting

This project analyzes and forecasts tourist arrivals to Sri Lanka using historical data from key countries. It combines data science techniques, time series forecasting, and interactive visualizations to provide insights into tourism trends.

---

## 🚀 Features

- **Data Cleaning & Preparation:** Processes raw tourism data for analysis.
- **Exploratory Data Analysis (EDA):** Visualizes trends, seasonal patterns, and country-wise arrivals.
- **Time Series Forecasting:** Uses SARIMA models to predict future tourist arrivals.
- **Interactive Dashboards:** Built with Plotly and Dash for dynamic exploration of data.
- **Multi-country Analysis:** Focus on major tourist source countries including India, China, Russia, UK, Germany, and others.

---

## 📂 Project Structure
├── Country Data.csv # Tourism arrival dataset (CSV)
├── tourism_analysis.py # Main Python script for analysis and forecasting
├── README.md # Project documentation
|─  requirements.txt # Python dependencies


⚙️ **Installation & Setup**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/IsuruHansaka/SriLanka-Tourism-Analysis.git
   cd SriLanka-Tourism-Analysis
   
Create a virtual environment (recommended):
  python -m venv venv
  source venv/bin/activate     # On Windows: venv\Scripts\activate

Install required Python packages:
  pip install -r requirements.txt

🏃 **How to Run**
Run the main analysis and dashboard script:
  python tourism_analysis.py

This will:
Perform exploratory data analysis with static plots.
Build and evaluate SARIMA forecasting models.
Launch an interactive Plotly Dash dashboard in your web browser.

📈 Insights
Monthly tourist arrivals by country.
Seasonal trends and patterns.
Forecasts for the next 12 months.
Interactive visualizations for deeper exploration.

📋 Dependencies
  pandas
  numpy
  matplotlib
  seaborn
  statsmodels
  plotly
  dash

🤝 Contribution
Feel free to fork the project, create issues, or submit pull requests. Suggestions and improvements are always welcome!

📄 License
This project is open-source and available under the MIT License.

🙏 Acknowledgements
Special thanks to all open data sources and Python libraries that make this analysis possible.

Created by Isuru Hansaka

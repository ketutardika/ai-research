# ai-research

AI/Machine Learning research project focused on predictive analytics for e-commerce and digital marketing applications.

## Project Overview

This repository contains three independent machine learning models:

1. **LSTM Sales Forecasting** - Time-series prediction for daily e-commerce sales using deep learning
2. **Delivery Time Estimation** - Random Forest model to predict order delivery times
3. **Conversion Rate Prediction** - Random Forest model for website traffic conversion analysis

## Development Environment

This project is developed using **Spyder IDE**, a scientific Python development environment that's particularly well-suited for data science and machine learning projects.

## Running the Models

Each script can be executed independently:

```bash
# LSTM-based sales forecasting (generates GIF/MP4 animation)
python e-commerce-sales-data.py

# Random Forest delivery time prediction
python product_recomendation.py

# Random Forest conversion rate prediction
python web_traffic.py
```

## Dependencies

Install required libraries:

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn pillow
```

## Datasets

- **E-commerce-Dataset.csv**: 51,291 transactions from 2018-2019
- **website_data.csv**: 2,000 web traffic sessions with conversion data

## License

This project is licensed under the GNU General Public License v2.0 - see the LICENSE file for details.
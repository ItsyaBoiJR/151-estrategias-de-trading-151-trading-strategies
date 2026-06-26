# 151 Trading Strategies - Research Paper Implementation

This repository contains a Python/PyTorch implementation of selected strategies from the paper "[151 Estrategias de Trading (151 Trading Strategies)](https://arxiv.org/pdf/1912.04492v1)" by Zura Kakushadze and Juan Andrés Serur. The paper, written in Spanish, provides detailed descriptions and mathematical formulations for over 150 trading strategies across diverse asset classes and trading styles. This repository focuses on implementing and backtesting a subset of these strategies.

---

## Overview

### Core Concept
The paper aims to serve as a comprehensive guide to trading strategies for various asset classes, including stocks, options, fixed income, futures, ETFs, commodities, cryptocurrencies, and more. It combines theoretical descriptions with practical implementations, including machine learning-based strategies such as neural networks, Bayesian models, and k-nearest neighbors.

The strategies are backed by over 550 mathematical formulas and include pedagogical notes to help practitioners and researchers understand their applications. Additionally, the paper provides source code for out-of-sample backtesting, making it a valuable resource for quantitative finance and algorithmic trading enthusiasts.

### Key Features of the Paper
1. **Diversified Asset Classes**: Covers trading strategies for stocks, options, fixed income, commodities, cryptocurrencies, and several other asset types.
2. **Machine Learning Integration**: Includes strategies based on machine learning algorithms such as artificial neural networks, Bayesian inference, and k-nearest neighbors.
3. **Extensive Supporting Materials**: Provides over 550 mathematical formulations, notes for backtesting, 2,000 references, and 900 glossary definitions.
4. **Pedagogical Approach**: Designed to be descriptive and educational, making complex strategies accessible to a wider audience.

---

## What's in this Repository?

This repository implements a subset of the trading strategies described in the paper, focusing on providing practical examples and backtesting tools using Python and PyTorch. The implementation is modular and includes the following components:

### Code Structure
- **`data_preprocessing.py`**: Handles data preparation, including cleaning, normalization, and feature engineering for financial datasets.
- **`strategy_models.py`**: Contains implementations of selected trading strategies, including machine learning-based approaches.
- **`backtesting.py`**: Provides utilities for out-of-sample backtesting to evaluate the performance of the implemented strategies.
- **`config.py`**: Centralized configuration file to set parameters for the models, backtesting, and data handling.
- **`main.py`**: Main entry point to execute the strategies, invoking data preprocessing, model training, and backtesting.

### Features
1. **Implementation of Trading Strategies**:
   - Selected strategies from the paper, including traditional and machine learning-based ones.
   - Modular design for easy extension and experimentation.

2. **Backtesting Framework**:
   - Evaluate strategies on realistic financial data.
   - Includes tools for performance metrics like Sharpe ratio, maximum drawdown, and cumulative returns.

3. **Machine Learning Integration**:
   - Demonstrates the use of PyTorch for training and evaluating machine learning-based trading strategies.
   - Example algorithms: artificial neural networks, Bayesian models.

4. **Comprehensive Documentation**:
   - Comments and explanatory notes throughout the code to help users understand the implementation details.

---

## Installation

### Prerequisites
- Python 3.8 or later
- PyTorch 1.10 or later
- Required Python libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn`

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/151-trading-strategies.git
   cd 151-trading-strategies
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Data**:
   - Place your financial data files in the `data/` directory.
   - Update the paths in `config.py` accordingly.

4. **Run the Implementation**:
   ```bash
   python main.py
   ```

---

## Example Usage

The repository includes example scripts for running individual trading strategies. By modifying `config.py`, you can select the strategy to implement, adjust hyperparameters, and specify the data source.

```python
# config.py
STRATEGY = "neural_network"  # Choose the strategy to run
DATA_PATH = "data/financial_dataset.csv"  # Path to the financial dataset
BACKTEST_SPLIT = 0.8  # Train-test split ratio for backtesting
```

Run the selected strategy:
```bash
python main.py
```

---

## Results and Evaluation

The implementation outputs performance metrics for the selected strategies, including:
- **Sharpe Ratio**: Measures risk-adjusted returns.
- **Maximum Drawdown**: Indicates the largest peak-to-trough loss during the backtesting period.
- **Cumulative Returns**: Tracks the overall profitability of the strategy.

Visualizations such as equity curves and drawdown charts are generated to help users analyze the strategy's performance.

---

## Contributing

Contributions are welcome! If you have suggestions for improvements or want to add new trading strategies, feel free to open an issue or submit a pull request.

---

## References

- Zura Kakushadze, Juan Andrés Serur, "[151 Estrategias de Trading (151 Trading Strategies)](https://arxiv.org/pdf/1912.04492v1)", arXiv preprint, 2019.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
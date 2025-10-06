# XAUUSD-LSTM

A Python-based LSTM model for forecasting **XAU/USD (Gold vs USD)** prices using historical 15-minute data. This project allows you to train an LSTM neural network, evaluate its predictions, and generate future price forecasts.

---

## Features

- Train an LSTM model on historical gold prices.
- Automatic preprocessing and resampling of data.
- Customizable sequence length, batch size, epochs, and validation split.
- Save best model checkpoint and final trained model.
- Generate recursive multi-step forecasts.
- Plot validation predictions and historical + forecast trends.
- Support for CSV input and output for easy integration.

---

## Dataset

XAUUSD 15 minutes (2024 - 2025)

```
Date;Open;High;Low;Close;Volume
2004.06.11 07:15;384;384.3;383.8;384.3;12
2004.06.11 07:30;383.8;384.3;383.6;383.8;12
2004.06.11 07:45;383.3;383.8;383.3;383.8;20
...
```

https://www.kaggle.com/datasets/novandraanugrah/xauusd-gold-price-historical-data-2004-2024

---

## Running

```bash
python train_xau_lstm.py --data "XAU_15m_data.csv" --datetime_col "Date" --target "Close" --seq_len 96 --epochs 5 --batch_size 64 --resample 15T
```

---

## Requirements

- Python 3.9+
- TensorFlow 2.x
- NumPy
- pandas
- scikit-learn
- matplotlib

Install dependencies via:

```bash
pip install -r requirements.txt
````

*(You can create `requirements.txt` with the following:)*

```
tensorflow
numpy
pandas
scikit-learn
matplotlib
```

---

## Usage

### Command-Line Arguments

```bash
python train_xau_lstm.py --data "XAU_15m_data.csv" --datetime_col "Date" --target "Close" \
--seq_len 96 --epochs 30 --batch_size 64 --resample 15min --forecast_steps 16
```

**Arguments:**

| Argument           | Description                                                        | Default                |
| ------------------ | ------------------------------------------------------------------ | ---------------------- |
| `--data`           | Path to the CSV file containing historical data                    | `XAU_15m_data.csv`     |
| `--datetime_col`   | Name of the datetime column in CSV (auto-detected if not provided) | `None`                 |
| `--target`         | Target column for prediction                                       | `Close`                |
| `--resample`       | Resample frequency (`15min`, `H`, `D`, etc.)                       | `min`                  |
| `--seq_len`        | Sequence length for LSTM input                                     | `96`                   |
| `--val_split`      | Fraction of data for validation                                    | `0.2`                  |
| `--epochs`         | Number of training epochs                                          | `30`                   |
| `--batch_size`     | Training batch size                                                | `64`                   |
| `--patience`       | Early stopping patience (number of epochs)                         | `6`                    |
| `--dropout`        | Dropout rate in LSTM layers                                        | `0.2`                  |
| `--model_path`     | Path to save the best model checkpoint                             | `xau_lstm_best.keras`  |
| `--final_model`    | Path to save the final trained model                               | `xau_lstm_final.keras` |
| `--plot_path`      | Path to save validation prediction plot                            | `prediction_plot.png`  |
| `--forecast_steps` | Number of future steps to forecast                                 | `16`                   |
| `--forecast_out`   | CSV output path for forecast results                               | `forecast_next.csv`    |

---

## Output

1. **Model files:**

   * `xau_lstm_best.keras`: Best model checkpoint during training.
   * `xau_lstm_final.keras`: Final trained model.

2. **Forecast CSV:**
   Example `forecast_next.csv`:

```
,forecast
2025-09-30 19:45:00,3280.56
2025-09-30 20:00:00,3308.98
2025-09-30 20:15:00,3337.23
...
2025-09-30 23:30:00,3529.38
```

3. **Plots:**

   * `prediction_plot.png`: Validation actual vs predicted prices.
   * `historical_forecast_plot.png`: Historical prices + future forecast.

---

## Example Workflow

```bash
# Train and forecast
python train_xau_lstm.py --data "XAU_15m_data.csv" --datetime_col "Date" --target "Close" \
--seq_len 96 --epochs 5 --batch_size 32 --resample 15min --forecast_steps 16
```

* Model trains on historical XAU/USD prices.
* Validation predictions are plotted.
* Next 16 steps forecast is saved to `forecast_next.csv`.

---

## Notes

* Make sure your CSV uses `;` as a separator.
* `resample` should match your data frequency (15min, 1H, etc.).
* For faster testing, reduce `seq_len`, `epochs`, or use a subset of the data.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Author

**Max Base**

GitHub: [https://github.com/BaseMax/XAUUSD-LSTM](https://github.com/BaseMax/XAUUSD-LSTM)

Copyright 2025, Max Base

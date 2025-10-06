import os
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def load_data(path, datetime_col=None):
    df = pd.read_csv(path)
    if datetime_col is None:
        for name in ['Datetime','Date','timestamp','time','date']:
            if name in df.columns:
                datetime_col = name
                break
    if datetime_col is None:
        raise ValueError("Couldn't find datetime column. Provide --datetime_col if needed.")
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
    df = df.dropna(subset=[datetime_col])
    df = df.sort_values(datetime_col).reset_index(drop=True)
    df = df.set_index(datetime_col)
    return df

def preprocess(df, target_col='Close', feature_cols=None, resample=None):
    if feature_cols is None:
        candidates = ['Open','High','Low','Close','Volume']
        feature_cols = [c for c in candidates if c in df.columns]
        if len(feature_cols) == 0 and 'close' in [c.lower() for c in df.columns]:
            for c in df.columns:
                if c.lower() == 'close':
                    feature_cols = [c]
    data = df[feature_cols].copy()
    if resample:
        data = data.resample(resample).agg({
            col: 'last' if col=='Close' else 'mean' for col in data.columns
        }).dropna()
    return data, feature_cols

def create_sequences(data_array, seq_len):
    X, y = [], []
    for i in range(len(data_array) - seq_len):
        X.append(data_array[i:i+seq_len])
        y.append(data_array[i+seq_len, 0])
    return np.array(X), np.array(y)

def build_model(input_shape, dropout=0.2):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def main(args):
    print("Loading:", args.data)
    df = load_data(args.data, datetime_col=args.datetime_col)
    data, feature_cols = preprocess(df, target_col=args.target, resample=args.resample)
    print("Using features:", feature_cols)
    if args.target not in data.columns:
        raise ValueError(f"Target column {args.target} not found in data columns: {list(data.columns)}")
    cols = [args.target] + [c for c in data.columns if c != args.target]
    data = data[cols]
    data = data.fillna(method='ffill').dropna()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data.values)
    seq_len = args.seq_len
    X, y = create_sequences(scaled, seq_len)
    split = int(len(X) * (1 - args.val_split))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
    model = build_model((seq_len, X.shape[2]), dropout=args.dropout)
    model.summary()
    callbacks = []
    ckpt_path = args.model_path or "xau_lstm_best.h5"
    callbacks.append(ModelCheckpoint(ckpt_path, save_best_only=True, monitor='val_loss', verbose=1))
    callbacks.append(EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True, verbose=1))
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2
    )
    y_pred = model.predict(X_val).flatten()
    def inv_scale(y_scaled):
        dummy = np.zeros((len(y_scaled), scaled.shape[1]))
        dummy[:,0] = y_scaled
        inv = scaler.inverse_transform(dummy)[:,0]
        return inv
    y_val_inv = inv_scale(y_val)
    y_pred_inv = inv_scale(y_pred)
    rmse = mean_squared_error(y_val_inv, y_pred_inv, squared=False)
    mae = mean_absolute_error(y_val_inv, y_pred_inv)
    print(f"Validation RMSE: {rmse:.6f}, MAE: {mae:.6f}")
    model.save(args.final_model or "xau_lstm_final.h5")
    plt.figure(figsize=(12,6))
    plt.plot(y_val_inv, label='Actual (val)')
    plt.plot(y_pred_inv, label='Predicted (val)')
    plt.legend()
    plt.title('XAU/USD - Actual vs Predicted (Validation)')
    plt.xlabel('Sample')
    plt.ylabel('Price (USD)')
    plt.grid(True)
    plt.savefig(args.plot_path or "prediction_plot.png", dpi=150)
    print("Plot saved.")
    n_steps = args.forecast_steps
    last_window = scaled[-seq_len:]
    preds = []
    window = last_window.copy()
    for i in range(n_steps):
        input_w = window.reshape((1, seq_len, window.shape[1]))
        p = model.predict(input_w)[0,0]
        preds.append(p)
        window = np.vstack([window[1:], np.hstack([p, window[-1,1:]])])
    preds_inv = inv_scale(np.array(preds))
    print(f"Next {n_steps} step forecast (inverse-scaled):")
    print(preds_inv)
    future_index = pd.date_range(start=data.index[-1], periods=n_steps+1, freq=args.resample or '15T')[1:]
    df_fore = pd.DataFrame({'forecast': preds_inv}, index=future_index)
    df_fore.to_csv(args.forecast_out or "forecast_next.csv")
    print("Forecast saved to", args.forecast_out or "forecast_next.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to CSV file')
    parser.add_argument('--datetime_col', default=None, help='Name of datetime column (infer if not given)')
    parser.add_argument('--target', default='Close', help='Target column name (default: Close)')
    parser.add_argument('--resample', default=None, help="Optional resample rule like '15T','H','D'. If None, use original index.")
    parser.add_argument('--seq_len', type=int, default=96, help='Sequence length (default 96 => approx 24h for 15m data)')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split fraction')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--model_path', help='Checkpoint path')
    parser.add_argument('--final_model', help='Final saved model path')
    parser.add_argument('--plot_path', help='Plot file path')
    parser.add_argument('--forecast_steps', type=int, default=16, help='How many future steps to forecast recursively')
    parser.add_argument('--forecast_out', help='CSV output for forecast')
    args = parser.parse_args()
    main(args)

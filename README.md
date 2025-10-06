# XAUUSD-LSTM

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

## Running

```bash
python train_xau_lstm.py --data "XAU_15m_data.csv" --datetime_col "Date" --target "Close" --seq_len 96 --epochs 30 --batch_size 64
```

Copyright 2025, Max Base

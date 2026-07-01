# Bot Sinyal XAUUSD Telegram

Bot Telegram berbasis Python yang menghasilkan sinyal trading otomatis untuk pasangan XAUUSD (Emas/USD) menggunakan Machine Learning dan analisis teknikal multi-indikator.

## Fitur Utama
- **Data pasar real-time** dari Deriv API (WebSocket)
- **Indikator teknikal lengkap**: SMA, EMA, BB, MACD, RSI, Stochastic, CCI, Williams %R, ATR, OBV, MFI, Parabolic SAR
- **Deteksi pola candlestick**: Hammer, Engulfing, Doji, Morning Star, Evening Star, Shooting Star, Marubozu
- **Machine Learning**: Random Forest (scikit-learn) untuk meningkatkan akurasi sinyal
- **Auto-evaluasi**: Retrain model otomatis setiap 50 trade
- **Database SQLite**: Pencatatan semua trade dan hasil evaluasi

## Perintah Bot
| Perintah | Fungsi |
|----------|--------|
| `/start` | Daftar menerima sinyal |
| `/stop`  | Berhenti menerima sinyal |
| `/stats` | Lihat statistik (win rate, total trade) |
| `/ping`  | Cek status bot |

## Konfigurasi
Secrets yang diperlukan di Replit:
- `TELEGRAM_BOT_TOKEN` — Token dari @BotFather

## Struktur File
```
main.py            – Entry point
config.py          – Konfigurasi global
database.py        – Operasi SQLite (trades.db)
deriv_client.py    – WebSocket Deriv API
indicators.py      – Indikator teknikal (pandas-ta + manual CDL)
ml_model.py        – Model Random Forest
signal_generator.py– Logika sinyal & tracking TP/SL
bot.py             – Telegram bot handler
requirements.txt   – Dependensi Python
```

## Cara Kerja
1. Bot terhubung ke Deriv WebSocket dan mengunduh 500 candle historis (5 menit)
2. Indikator teknikal dihitung, model awal dilatih dari data historis
3. Setiap 5 detik: ensemble voting (10 indikator) + prediksi ML → sinyal BUY/SELL
4. Sinyal dikirim ke Telegram dengan Entry, TP, dan SL
5. Harga dipantau real-time; hasil WIN/LOSE dicatat di database
6. Setiap 50 trade selesai: model dilatih ulang otomatis

## User Preferences
- Gunakan bahasa Indonesia untuk komentar dan pesan bot

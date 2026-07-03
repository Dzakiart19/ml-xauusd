"""
Model Machine Learning (Random Forest) untuk prediksi sinyal.
Thread-safe: semua akses model dilindungi oleh model_lock.
"""

import json
import logging
import os
import threading
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from config import (
    RF_N_ESTIMATORS, RF_MAX_DEPTH,
    MODEL_FILE, SCALER_FILE,
    ATR_TP_MULTIPLIER, ATR_SL_MULTIPLIER,
    LABEL_LOOKAHEAD, SPREAD_ESTIMATE,
    FEATURE_IMPORTANCE_FILE,
)
from indicators import get_feature_names, extract_features

logger = logging.getLogger(__name__)


class XAUModel:
    def __init__(self):
        self.model  = None
        self.scaler = None
        # Lock melindungi model + scaler dari race condition predict vs retrain
        self.model_lock = threading.RLock()
        self._load_or_create()

    # ─── Simpan / Muat ───────────────────────────────────────────────────────

    def _load_or_create(self):
        if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
            try:
                with self.model_lock:
                    self.model  = joblib.load(MODEL_FILE)
                    self.scaler = joblib.load(SCALER_FILE)
                logger.info("Model dimuat dari disk.")
                return
            except Exception as e:
                logger.warning(f"Gagal memuat model: {e}. Buat model baru.")

        with self.model_lock:
            self.model  = RandomForestClassifier(
                n_estimators=RF_N_ESTIMATORS,
                max_depth=RF_MAX_DEPTH,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1,          # paralel semua core
            )
            self.scaler = StandardScaler()

    def save(self):
        try:
            with self.model_lock:
                joblib.dump(self.model,  MODEL_FILE)
                joblib.dump(self.scaler, SCALER_FILE)
            logger.info("Model disimpan ke disk.")
        except Exception as e:
            logger.error(f"Gagal menyimpan model: {e}")

    # ─── Label Simulasi ───────────────────────────────────────────────────────

    @staticmethod
    def simulate_labels(df: pd.DataFrame) -> pd.Series:
        """
        Simulasi apakah trade BUY menang (1) atau kalah (0).
        Melihat ke LABEL_LOOKAHEAD candle ke depan.
        """
        highs  = df['high'].values
        lows   = df['low'].values
        closes = df['close'].values
        atrs   = df['ATRr_14'].values if 'ATRr_14' in df.columns else np.full(len(df), 1.0)

        labels = np.full(len(df), np.nan)

        for i in range(len(df) - LABEL_LOOKAHEAD):
            atr = atrs[i]
            if np.isnan(atr) or atr <= 0:
                continue

            entry = closes[i]
            tp    = entry + ATR_TP_MULTIPLIER * atr
            sl    = entry - ATR_SL_MULTIPLIER * atr

            outcome = np.nan
            for j in range(i + 1, i + 1 + LABEL_LOOKAHEAD):
                # Fix 2: terapkan spread ke TP agar label lebih realistis
                # BUY entry sesungguhnya di ask (mid + spread), TP lebih sulit dicapai
                if highs[j] >= tp + SPREAD_ESTIMATE:
                    outcome = 1   # WIN
                    break
                if lows[j] <= sl:
                    outcome = 0   # LOSE
                    break
            labels[i] = outcome

        return pd.Series(labels, index=df.index)

    # ─── Training ─────────────────────────────────────────────────────────────

    def initial_train(self, df: pd.DataFrame) -> float:
        """Latih model pertama kali pada data historis."""
        logger.info("Memulai pelatihan awal model...")

        labels = self.simulate_labels(df)
        X      = extract_features(df)

        mask    = ~labels.isna()
        X_train = X[mask].values
        y_train = labels[mask].values

        if len(X_train) < 20:
            logger.warning("Data latih terlalu sedikit untuk pelatihan awal.")
            return 0.0

        try:
            with self.model_lock:
                X_scaled = self.scaler.fit_transform(X_train)
                self.model.fit(X_scaled, y_train)
                scores = cross_val_score(
                    self.model, X_scaled, y_train, cv=3, scoring='accuracy'
                )
                acc = scores.mean()

            logger.info(f"Pelatihan awal selesai. Akurasi CV: {acc:.2%}")
            self._save_feature_importance()
            self.save()
            return acc
        except Exception as e:
            logger.error(f"Pelatihan awal gagal: {e}")
            return 0.0

    def retrain(self, trades: list) -> float:
        """Latih ulang model dengan data trade nyata dari SQLite."""
        logger.info(f"Melatih ulang model dengan {len(trades)} trade...")

        feature_names = get_feature_names()
        records = []
        labels  = []

        for t in trades:
            col_map = {
                'RSI_14':           t.get('rsi',         np.nan),
                'MACDh_12_26_9':    t.get('macd_hist',   np.nan),
                'ATRr_14':          t.get('atr',         np.nan),
                'bb_pos':           t.get('bb_pos',      np.nan),
                'ema_cross':        t.get('ema_signal',  0),
                'STOCHk_14_3_3':    t.get('stoch_k',     np.nan),
                'STOCHd_14_3_3':    t.get('stoch_d',     np.nan),
                'CCI_20_0.015':     t.get('cci',         np.nan),
                'WILLR_14':         t.get('willr',       np.nan),
                'bullish_cdl':      t.get('bullish_cdl', 0),
                'bearish_cdl':      t.get('bearish_cdl', 0),
                'sar_bull':         t.get('sar_bull', 0),
                'trend_bull':       t.get('trend_bull',  0),
            }
            row = [col_map.get(feat, np.nan) for feat in feature_names]
            records.append(row)
            labels.append(1 if t['outcome'] == 'WIN' else 0)

        X = np.array(records, dtype=float)
        y = np.array(labels)

        # Isi NaN dengan median kolom
        col_medians = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            nan_mask = np.isnan(X[:, j])
            X[nan_mask, j] = col_medians[j] if not np.isnan(col_medians[j]) else 0

        if len(X) < 20:
            logger.warning("Data retrain terlalu sedikit.")
            return 0.0

        try:
            with self.model_lock:
                X_scaled = self.scaler.fit_transform(X)
                self.model.fit(X_scaled, y)
                win_rate = y.mean() * 100

            logger.info(f"Retrain selesai. Win rate data latih: {win_rate:.1f}%")
            self._save_feature_importance()
            self.save()
            return win_rate
        except Exception as e:
            logger.error(f"Retrain gagal: {e}")
            return 0.0

    # ─── Feature Importance (Fix 6) ───────────────────────────────────────────

    def _save_feature_importance(self):
        """
        Simpan feature importance ke JSON setelah setiap training/retrain.
        Membantu identifikasi fitur mana yang paling prediktif vs noise.
        """
        try:
            if not hasattr(self.model, 'feature_importances_'):
                return
            names = get_feature_names()
            importances = {
                name: round(float(imp), 4)
                for name, imp in zip(names, self.model.feature_importances_)
            }
            importances_sorted = dict(
                sorted(importances.items(), key=lambda x: x[1], reverse=True)
            )
            with open(FEATURE_IMPORTANCE_FILE, 'w') as f:
                json.dump(importances_sorted, f, indent=2)
            top3 = list(importances_sorted.items())[:3]
            logger.info(f"Feature importance disimpan. Top 3: {top3}")
        except Exception as e:
            logger.warning(f"Gagal simpan feature importance: {e}")

    # ─── Prediksi ─────────────────────────────────────────────────────────────

    def predict(self, feature_row: pd.Series) -> tuple:
        """
        Kembalikan (label, proba).
        label: 1 = BUY, 0 = SELL, None = model belum siap (jangan trade).
        proba: probabilitas untuk label tersebut.
        """
        with self.model_lock:
            if self.model is None or not hasattr(self.model, 'estimators_'):
                # Model belum dilatih — jangan trade (fail-safe netral)
                return None, 0.0

            try:
                X        = feature_row.values.reshape(1, -1)
                X_scaled = self.scaler.transform(X)
                label    = int(self.model.predict(X_scaled)[0])
                proba    = float(self.model.predict_proba(X_scaled)[0][label])
                return label, proba
            except Exception as e:
                logger.error(f"Prediksi gagal: {e}")
                return None, 0.0

from datetime import datetime
from pandas import DataFrame
import numpy as np
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy
from smartmoneyconcepts import smc


class OBlock_1h(IStrategy):
    """
    Order Block + Monday Range (ICT) stratejisi:
    - smc.ob() ile gerçek order block tespiti
    - smc.liquidity() ile likidite süpürme
    - Pazartesi range süpürmesi sonrası reversal
    - 4h timeframe (kurumsal seviyeler net)
    """
    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = True
    startup_candle_count = 200

    stoploss = -0.06
    minimal_roi = {"0": 0.08, "480": 0.05, "1440": 0.025, "2880": 0.01}
    process_only_new_candles = True
    use_exit_signal = True

    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.04
    trailing_only_offset_is_reached = True

    leverage_value = 3

    def leverage(self, pair, current_time, current_rate, proposed_leverage, max_leverage, side, **kwargs):
        return min(self.leverage_value, max_leverage)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()
        try:
            shl = smc.swing_highs_lows(df, swing_length=10)
            ob = smc.ob(df, shl, close_mitigation=False)
            df["ob_signal"] = ob["OB"].fillna(0).astype(float)
            df["ob_top"] = ob["Top"].ffill()
            df["ob_bottom"] = ob["Bottom"].ffill()
            liq = smc.liquidity(df, shl, range_percent=0.01)
            df["liq_signal"] = liq["Liquidity"].fillna(0).astype(float)
            df["liq_swept"] = liq["Swept"].fillna(0).astype(float)
        except Exception:
            df["ob_signal"] = 0
            df["ob_top"] = np.nan
            df["ob_bottom"] = np.nan
            df["liq_signal"] = 0
            df["liq_swept"] = 0

        # Bullish OB = +1, Bearish OB = -1; son N barda OB oluştu mu
        df["bull_ob"] = (df["ob_signal"] == 1).astype(int)
        df["bear_ob"] = (df["ob_signal"] == -1).astype(int)
        df["bull_ob_recent"] = df["bull_ob"].rolling(8).max().fillna(0)
        df["bear_ob_recent"] = df["bear_ob"].rolling(8).max().fillna(0)
        df["liq_sweep_recent"] = (df["liq_swept"] > 0).rolling(5).max().fillna(0).astype(int)

        # Monday Range (haftalık) — Pazartesi high/low, hafta boyunca taşınır
        df["dow"] = df["date"].dt.dayofweek
        # Yıl-hafta string key (NaN sorunu olmaz)
        df["weekkey"] = df["date"].dt.strftime("%G-%V")
        monday_mask = df["dow"] == 0
        df["monday_high"] = df["high"].where(monday_mask).groupby(df["weekkey"]).transform("max")
        df["monday_low"] = df["low"].where(monday_mask).groupby(df["weekkey"]).transform("min")
        df["monday_high"] = df["monday_high"].ffill()
        df["monday_low"] = df["monday_low"].ffill()
        df["swept_monday_high"] = (df["high"] > df["monday_high"]).astype(int)
        df["swept_monday_low"] = (df["low"] < df["monday_low"]).astype(int)

        df["ema200"] = ta.EMA(df, timeperiod=200)
        df["rsi"] = ta.RSI(df, timeperiod=14)
        df["volume_sma"] = ta.SMA(df["volume"], timeperiod=20)
        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe
        # LONG: Monday low süpürüldü + bullish OB + reversal + hacim
        long_cond = (
            (df["swept_monday_low"].shift(1) == 1)        # Monday low süpürüldü (önceki bar)
            & (df["close"] > df["monday_low"])            # geri içeri döndü
            & (df["bull_ob_recent"] == 1)                 # yakında bullish OB
            & (df["rsi"] < 50)
            & (df["volume"] > df["volume_sma"])
        )
        # SHORT: Monday high süpürüldü + bearish OB + reversal
        short_cond = (
            (df["swept_monday_high"].shift(1) == 1)
            & (df["close"] < df["monday_high"])
            & (df["bear_ob_recent"] == 1)
            & (df["rsi"] > 50)
            & (df["volume"] > df["volume_sma"])
        )
        df.loc[long_cond, ["enter_long", "enter_tag"]] = (1, "ob_monday_long")
        df.loc[short_cond, ["enter_short", "enter_tag"]] = (1, "ob_monday_short")
        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe
        df.loc[(df["rsi"] > 70) | (df["bear_ob"] == 1), "exit_long"] = 1
        df.loc[(df["rsi"] < 30) | (df["bull_ob"] == 1), "exit_short"] = 1
        return df

"""
╔══════════════════════════════════════════════════════════════════════╗
║   NIFTY GAMMA SCALPING ENGINE — STRADDLE + DELTA HEDGE               ║
║   Based on v4.0 — Hedging restored, afternoon balance fixed          ║
║                                                                      ║
║  FIXES vs previous pure-straddle version:                            ║
║  1. DTE_MIN=2, DTE_MAX=6  → Mon(3), Tue(2), Fri(6) all qualify      ║
║     Previously DTE 3-4 only let Monday in (was 65:7 morning:arvo)   ║
║  2. COOLDOWN reduced 64→30 min — 64 min blocked whole afternoons     ║
║  3. MAX_TRADES_PER_DAY: 5→8, separate morning/afternoon caps (4+4)   ║
║  4. FLAT_STOP added: exit if straddle flat (±2%) for >45 min         ║
║     Converts slow theta_bleed into quicker, smaller exits            ║
║  5. Delta hedge re-added (correctly):                                 ║
║     - Spot moves HEDGE_STEP_PTS from anchor → open hedge             ║
║     - UP move → buy PUT proxy (straddle/2 price, avoids ITM bug)     ║
║     - DOWN move → buy CALL proxy                                     ║
║     - Revert within HEDGE_REVERT_PTS → close for gamma P&L           ║
║     - SL if spot extends HEDGE_SL_PTS further against hedge          ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os, glob, re, warnings
from datetime import date
from typing import Optional, Dict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from scipy.stats import norm

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None


# ═════════════════════════════════════════════════════════════════════
#  PATHS
# ═════════════════════════════════════════════════════════════════════
DATA_1MIN = './Updated Data'
OUT_DIR   = './OUTPUT'


# ═════════════════════════════════════════════════════════════════════
#  TRADE TIMING WINDOWS
#  Each tuple: (start_time, end_time) in 'HH:MM'.
#  New entries are only opened inside these windows.
# ═════════════════════════════════════════════════════════════════════
TRADE_WINDOWS = [
    ("09:30", "12:30"),   # Morning window
    ("12:30", "15:15"),   # Afternoon window
]

# Per-window max trades cap (aligned with TRADE_WINDOWS order above).
# Set to a large number to effectively disable per-window cap and rely
# only on MAX_TRADES_PER_DAY.
WINDOW_MAX_TRADES = [4, 4]   # 4 morning + 4 afternoon = 8 total max


# ═════════════════════════════════════════════════════════════════════
#  STRATEGY PARAMETERS
# ═════════════════════════════════════════════════════════════════════
P = {
    # ── Entry filters ─────────────────────────────────────────────
    "IVR_MAX"           : 49.79,
    "IV_RV_MAX"         : 1.037,
    "RV_WINDOW"         : 59,
    "IVR_DAYS"          : 23,
    "ER_WINDOW"         : 38,

    # ── Session timing ────────────────────────────────────────────
    "SESSION_END"       : "15:10",

    # ── DTE filter — FIXED: was 3-4 (Monday ONLY). Now 2-6:
    #    Monday=3✅  Tuesday=2✅  Friday=6✅  Wed=1❌  Thu=7❌
    "DTE_MIN"           : 2,
    "DTE_MAX"           : 6,

    # ── Trade limits ──────────────────────────────────────────────
    "MAX_TRADES_PER_DAY": 8,     # raised from 5; window caps (4+4) enforce balance
    "COOLDOWN_LOSS_MIN" : 30,    # reduced from 64; 64 min blocked entire afternoons

    # ── Straddle exit thresholds ──────────────────────────────────
    "STRADDLE_TP"       : 0.1554,
    "STRADDLE_SL"       : 0.1380,
    "IV_EXPAND_EXIT"    : 0.1008,
    "THETA_MAX_MIN"     : 75,    # reduced from 88; faster exit on stale trades

    # ── Flat-stop — NEW: exit if straddle going nowhere ───────────
    # Fires when: |straddle_ret| < FLAT_STOP_PCT  AND  elapsed > FLAT_STOP_MINS
    # Converts slow theta_bleed into smaller, quicker exits
    "FLAT_STOP_MINS"    : 45,
    "FLAT_STOP_PCT"     : 0.02,

    # ── Delta hedge parameters — RESTORED ─────────────────────────
    "HEDGE_STEP_PTS"    : 51,    # open hedge when spot moves this much from anchor
    "HEDGE_REVERT_PTS"  : 41,    # close hedge when spot reverts this close to anchor
    "HEDGE_SL_PTS"      : 171,   # close hedge if spot extends this far against it
    "HEDGE_COST"        : 60,    # Rs per hedge round-trip (brokerage + impact)

    # ── Costs ─────────────────────────────────────────────────────
    "STRADDLE_COST"     : 120,

    # ── Position sizing ───────────────────────────────────────────
    "LOT_SIZE"          : 65,
    "LOTS"              : 1,

    # ── Constants ─────────────────────────────────────────────────
    "RISK_FREE"         : 0.065,
    "MINS_PER_DAY"      : 375,
}


# ═════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═════════════════════════════════════════════════════════════════════

def _clean_date(s):
    return str(s).replace('="','').replace('"','').replace('=','').strip()

def _offset_num(s):
    s = str(s).strip().upper()
    if s == 'ATM': return 0
    m = re.search(r'ATM([+-]\d+)', s)
    return int(m.group(1)) if m else None

def load_and_prepare(folder: str, label: str = '1MIN', keep: int = 2) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(folder, '*.csv')))
    if not files:
        raise FileNotFoundError(f"No CSVs in {folder}")

    print(f"\n{'─'*58}")
    print(f"  {label}: {len(files)} files from {folder}")
    frames = []
    for f in files:
        df = pd.read_csv(f, dtype={'date': str, 'time': str}, low_memory=False)
        df.columns = [c.strip().lower() for c in df.columns]
        frames.append(df)
        print(f"  ✓ {os.path.basename(f):30s} {len(df):>10,} rows")

    raw = pd.concat(frames, ignore_index=True)
    raw['date_c']   = raw['date'].apply(_clean_date)
    raw['datetime'] = pd.to_datetime(
        raw['date_c'] + ' ' + raw['time'].astype(str).str.strip(),
        dayfirst=True, errors='coerce')
    raw.dropna(subset=['datetime'], inplace=True)

    for col in ['open','high','low','close','iv','spot','volume','oi']:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors='coerce')

    raw['option_type']   = raw['option_type'].str.upper().str.strip()
    raw['strike_offset'] = raw['strike_offset'].str.upper().str.strip()
    raw['_off']          = raw['strike_offset'].apply(_offset_num)
    raw.dropna(subset=['_off'], inplace=True)
    raw['_off'] = raw['_off'].astype(int)
    raw = raw[raw['_off'].abs() <= keep]
    raw.sort_values('datetime', inplace=True)
    raw.reset_index(drop=True, inplace=True)
    print(f"  Total after filter: {len(raw):,} | "
          f"{raw['datetime'].min().date()} → {raw['datetime'].max().date()}")
    return raw


def make_straddle(raw: pd.DataFrame) -> pd.DataFrame:
    atm   = raw[raw['_off'] == 0]
    calls = atm[atm['option_type'] == 'CALL']
    puts  = atm[atm['option_type'] == 'PUT']

    def rename(df, prefix):
        return df[['datetime','open','high','low','close','iv','spot']].rename(
            columns={c: f'{prefix}_{c}' if c != 'datetime' else c
                     for c in ['open','high','low','close','iv','spot']})

    c = rename(calls, 'c')
    p = rename(puts,  'p')

    if c.empty and p.empty:
        raise ValueError("No ATM data found")
    elif c.empty:
        merged = p.copy(); merged['c_close'] = merged['p_close']
        merged['c_iv'] = merged['p_iv']; merged['c_spot'] = merged['p_spot']
    elif p.empty:
        merged = c.copy(); merged['p_close'] = merged['c_close']
        merged['p_iv'] = merged['c_iv']
    else:
        merged = pd.merge(c, p, on='datetime', how='inner')

    merged['spot']     = merged['c_spot']
    merged['straddle'] = merged['c_close'] + merged['p_close']
    merged['mid_iv']   = (merged['c_iv'] + merged['p_iv']) / 2.0
    merged['date']     = merged['datetime'].dt.date
    merged['hm']       = merged['datetime'].dt.strftime('%H:%M')
    merged.sort_values('datetime', inplace=True)
    merged.reset_index(drop=True, inplace=True)
    print(f"  ATM straddle rows: {len(merged):,} | avg: ₹{merged['straddle'].mean():.1f}")
    return merged


# ═════════════════════════════════════════════════════════════════════
#  INDICATORS
# ═════════════════════════════════════════════════════════════════════

def add_indicators(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    df = df.copy()
    annual = p['MINS_PER_DAY'] * 252

    df['iv_pct'] = df['mid_iv']
    df['iv_dec'] = df['mid_iv'] / 100.0

    lr       = np.log(df['spot'] / df['spot'].shift(1))
    df['rv'] = lr.rolling(p['RV_WINDOW']).std() * np.sqrt(annual)

    lb = p['IVR_DAYS'] * p['MINS_PER_DAY']
    lo = df['iv_dec'].rolling(lb, min_periods=50).min()
    hi = df['iv_dec'].rolling(lb, min_periods=50).max()
    df['ivr'] = (df['iv_dec'] - lo) / (hi - lo + 1e-10) * 100

    df['iv_rv'] = df['iv_dec'] / df['rv'].clip(lower=0.005)

    w     = p['ER_WINDOW']
    net   = (df['spot'] - df['spot'].shift(w)).abs()
    total = df['spot'].diff().abs().rolling(w).sum()
    df['er'] = (net / total.clip(lower=1e-10)).clip(0, 1)

    def _dte(dt):
        days = (3 - dt.weekday()) % 7
        return days if days > 0 else 7
    df['dte'] = df['datetime'].apply(_dte)

    hi_s = df['spot'] * 1.001; lo_s = df['spot'] * 0.999
    tr = pd.concat([hi_s - lo_s,
                    (hi_s - df['spot'].shift()).abs(),
                    (lo_s - df['spot'].shift()).abs()], axis=1).max(axis=1)
    df['spot_atr'] = tr.ewm(span=14, adjust=False).mean()
    return df


def _window_idx(hm: str, windows: list) -> int:
    """Return the index of the window this hm falls in, or -1."""
    for i, (s, e) in enumerate(windows):
        if s <= hm <= e:
            return i
    return -1


def add_signals(df: pd.DataFrame, p: dict, trade_windows: list) -> pd.DataFrame:
    df = df.copy()
    weekday_ok = df['datetime'].dt.weekday.isin([0,1,2,3,4])
    time_ok    = df['hm'].apply(lambda hm: _window_idx(hm, trade_windows) >= 0)
    ivr_ok     = df['ivr'] < p['IVR_MAX']
    ivrv_ok    = df['iv_rv'] < p['IV_RV_MAX']
    dte_ok     = df['dte'].between(p['DTE_MIN'], p['DTE_MAX'])
    valid      = df['rv'].notna() & (df['straddle'] > 0) & (df['iv_dec'] > 0)

    df['signal'] = weekday_ok & time_ok & ivr_ok & ivrv_ok & dte_ok & valid

    print(f"\n  Trade windows configured:")
    for i, (s, e) in enumerate(trade_windows):
        print(f"    Window {i+1}: {s} – {e}  (max {WINDOW_MAX_TRADES[i]} trades)")

    print(f"\n  Signals: {df['signal'].sum():,}")
    print(f"    weekday ok : {weekday_ok.sum():,}")
    print(f"    time ok    : {time_ok.sum():,}")
    print(f"    IVR<{p['IVR_MAX']} : {ivr_ok.sum():,}")
    print(f"    IV/RV<{p['IV_RV_MAX']} : {ivrv_ok.sum():,}")
    print(f"    DTE {p['DTE_MIN']}-{p['DTE_MAX']} ok   : {dte_ok.sum():,}")
    return df


# ═════════════════════════════════════════════════════════════════════
#  HEDGE DATACLASS
# ═════════════════════════════════════════════════════════════════════

@dataclass
class Hedge:
    """One delta-hedge leg."""
    flag        : str            # 'put' or 'call'
    entry_dt    : pd.Timestamp
    anchor_spot : float          # spot level at hedge open (revert target)
    entry_price : float          # option price proxy at entry
    sl_spot     : float          # spot SL level
    status      : str = 'open'
    exit_price  : float = 0.0
    exit_reason : str = ''
    net_pnl     : float = 0.0


# ═════════════════════════════════════════════════════════════════════
#  BACKTEST — STRADDLE + DELTA HEDGE
# ═════════════════════════════════════════════════════════════════════

def run_backtest(sig: pd.DataFrame, p: dict,
                 trade_windows: list, window_max: list) -> pd.DataFrame:
    """
    Main backtest loop — straddle + delta hedge.

    Hedge logic (correctly implemented):
    ─────────────────────────────────────
    When |spot - last_hedge_spot| >= HEDGE_STEP_PTS:
      • Spot moved UP  → straddle delta turned positive  → buy PUT proxy
      • Spot moved DOWN→ straddle delta turned negative  → buy CALL proxy
    Option price proxy = current_straddle / 2
      (ATM call ≈ ATM put ≈ straddle/2 by put-call parity)
      This avoids the ITM/OTM mismatch that caused -₹70k in v4.
    Close hedge when:
      • Spot reverts within HEDGE_REVERT_PTS of anchor  → spot_revert (profit)
      • Spot extends HEDGE_SL_PTS further against hedge  → sl (loss)
      • Straddle exits for any reason                   → force_close

    Flat-stop (new):
    ────────────────
    If elapsed > FLAT_STOP_MINS AND |straddle_ret| < FLAT_STOP_PCT:
      Exit immediately — position going nowhere, wasting theta.

    Afternoon cap (new):
    ────────────────────
    Per-window daily counters enforce WINDOW_MAX_TRADES independently,
    so morning losses can't exhaust the afternoon budget.
    """
    df   = sig.reset_index(drop=True)
    ls   = p['LOT_SIZE']
    lots = p['LOTS']

    open_trade   = None
    trades       = []

    # Separate per-window counters keyed by (date, window_idx)
    window_count : Dict[tuple, int] = {}
    daily_total  : Dict[date, int]  = {}
    last_loss_dt : Optional[pd.Timestamp] = None

    for i in range(len(df)):
        row   = df.iloc[i]
        dt    = row['datetime']
        spot  = row['spot']
        iv    = row['iv_dec']
        hm    = row['hm']
        strad = row['straddle']

        if pd.isna(spot) or pd.isna(iv) or spot <= 0 or iv <= 0:
            continue
        if dt.weekday() > 4:
            continue

        # ── MANAGE OPEN TRADE ─────────────────────────────────────
        if open_trade is not None:
            t       = open_trade
            elapsed = (dt - t['entry_dt']).total_seconds() / 60.0
            strad_ret = (strad / t['entry_straddle']) - 1.0
            iv_ratio  = iv / t['entry_iv'] if t['entry_iv'] > 0 else 1.0
            opt_proxy = strad / 2.0   # ATM option price ≈ straddle / 2

            # ── CHECK HEDGE EXITS ──────────────────────────────────
            for h in t['hedges']:
                if h.status != 'open':
                    continue
                reverted = abs(spot - h.anchor_spot) <= p['HEDGE_REVERT_PTS']
                sl_hit   = (h.flag == 'put'  and spot >= h.sl_spot) or \
                           (h.flag == 'call' and spot <= h.sl_spot)
                if reverted or sl_hit:
                    h.exit_price  = opt_proxy
                    h.status      = 'closed'
                    h.exit_reason = 'spot_revert' if reverted else 'sl'
                    h.net_pnl     = ((h.exit_price - h.entry_price)
                                     * lots * ls - p['HEDGE_COST'])
                    t['hedge_pnl'] += h.net_pnl

            # ── CHECK STRADDLE EXITS ───────────────────────────────
            exit_reason = None
            if strad_ret >= p['STRADDLE_TP']:
                exit_reason = 'gamma_win'
            elif strad_ret <= -p['STRADDLE_SL']:
                exit_reason = 'max_drawdown'
            elif iv_ratio >= (1 + p['IV_EXPAND_EXIT']):
                exit_reason = 'vega_win'
            elif (elapsed >= p['FLAT_STOP_MINS'] and
                  abs(strad_ret) < p['FLAT_STOP_PCT']):
                exit_reason = 'flat_stop'
            elif elapsed >= p['THETA_MAX_MIN'] and strad < t['entry_straddle']:
                exit_reason = 'theta_bleed'
            elif hm >= p['SESSION_END']:
                exit_reason = 'eod'

            if exit_reason:
                # Force-close all open hedges
                for h in t['hedges']:
                    if h.status == 'open':
                        h.exit_price  = opt_proxy
                        h.status      = 'closed'
                        h.exit_reason = 'force_close'
                        h.net_pnl     = ((h.exit_price - h.entry_price)
                                         * lots * ls - p['HEDGE_COST'])
                        t['hedge_pnl'] += h.net_pnl

                strad_pnl = (strad - t['entry_straddle']) * lots * ls
                total_pnl = strad_pnl + t['hedge_pnl'] - p['STRADDLE_COST']

                trades.append({
                    'entry_dt'       : t['entry_dt'],
                    'exit_dt'        : dt,
                    'entry_spot'     : t['entry_spot'],
                    'exit_spot'      : spot,
                    'strike'         : t['strike'],
                    'dte_entry'      : t['dte'],
                    'entry_iv_pct'   : t['entry_iv'] * 100,
                    'entry_rv_pct'   : t['entry_rv'] * 100,
                    'entry_ivr'      : t['entry_ivr'],
                    'entry_straddle' : t['entry_straddle'],
                    'exit_straddle'  : strad,
                    'straddle_pnl'   : strad_pnl,
                    'hedge_pnl'      : t['hedge_pnl'],
                    'costs'          : p['STRADDLE_COST'],
                    'realized_pnl'   : total_pnl,
                    'exit_reason'    : exit_reason,
                    'duration_mins'  : elapsed,
                    'hedge_count'    : len(t['hedges']),
                    'hedge_revert'   : sum(1 for h in t['hedges']
                                           if h.exit_reason == 'spot_revert'),
                    'hedge_sl'       : sum(1 for h in t['hedges']
                                           if h.exit_reason == 'sl'),
                    'er_entry'       : t['er'],
                    'trade_window'   : t['trade_window'],
                    'window_idx'     : t['window_idx'],
                })

                if total_pnl < 0:
                    last_loss_dt = dt
                open_trade = None
                continue

            # ── MAYBE OPEN A NEW HEDGE ─────────────────────────────
            spot_move = spot - t['last_hedge_spot']
            if abs(spot_move) >= p['HEDGE_STEP_PTS']:
                # Spot moved UP → delta positive → buy PUT to neutralise
                # Spot moved DOWN → delta negative → buy CALL
                flag    = 'put'  if spot_move > 0 else 'call'
                sl_spot = (spot + p['HEDGE_SL_PTS'] if spot_move > 0
                           else spot - p['HEDGE_SL_PTS'])
                h = Hedge(
                    flag        = flag,
                    entry_dt    = dt,
                    anchor_spot = t['last_hedge_spot'],
                    entry_price = opt_proxy,
                    sl_spot     = sl_spot,
                )
                t['hedges'].append(h)
                t['last_hedge_spot'] = spot
            continue

        # ── LOOK FOR NEW ENTRY ────────────────────────────────────
        if not row['signal']:
            continue

        w_idx = _window_idx(hm, trade_windows)
        if w_idx < 0:
            continue

        d  = dt.date()
        wk = (d, w_idx)

        # Daily total cap
        if daily_total.get(d, 0) >= p['MAX_TRADES_PER_DAY']:
            continue

        # Per-window cap
        if window_count.get(wk, 0) >= window_max[w_idx]:
            continue

        # Cooldown after any loss
        if last_loss_dt is not None:
            if dt < last_loss_dt + pd.Timedelta(minutes=p['COOLDOWN_LOSS_MIN']):
                continue

        if strad <= 0:
            continue

        K = round(spot / 50) * 50
        open_trade = {
            'entry_dt'       : dt,
            'entry_spot'     : spot,
            'entry_iv'       : iv,
            'entry_rv'       : float(row.get('rv', iv)),
            'entry_ivr'      : float(row.get('ivr', 50.0)),
            'entry_straddle' : strad,
            'strike'         : K,
            'dte'            : int(row.get('dte', 3)),
            'er'             : float(row.get('er', 0.5)),
            'hedges'         : [],
            'hedge_pnl'      : 0.0,
            'last_hedge_spot': spot,
            'trade_window'   : f"{trade_windows[w_idx][0]}-{trade_windows[w_idx][1]}",
            'window_idx'     : w_idx,
        }
        daily_total[d]   = daily_total.get(d, 0) + 1
        window_count[wk] = window_count.get(wk, 0) + 1

    # Close any open trade at end of data
    if open_trade is not None:
        last = df.iloc[-1]
        opt_proxy = last['straddle'] / 2.0
        for h in open_trade['hedges']:
            if h.status == 'open':
                h.exit_price  = opt_proxy
                h.status      = 'closed'
                h.exit_reason = 'end_of_data'
                h.net_pnl     = ((h.exit_price - h.entry_price)
                                 * lots * ls - p['HEDGE_COST'])
                open_trade['hedge_pnl'] += h.net_pnl
        sp    = (last['straddle'] - open_trade['entry_straddle']) * lots * ls
        total = sp + open_trade['hedge_pnl'] - p['STRADDLE_COST']
        trades.append({
            'entry_dt': open_trade['entry_dt'], 'exit_dt': last['datetime'],
            'entry_spot': open_trade['entry_spot'], 'exit_spot': last['spot'],
            'strike': open_trade['strike'], 'dte_entry': open_trade['dte'],
            'entry_iv_pct': open_trade['entry_iv']*100,
            'entry_rv_pct': open_trade['entry_rv']*100,
            'entry_ivr': open_trade['entry_ivr'],
            'entry_straddle': open_trade['entry_straddle'],
            'exit_straddle': last['straddle'],
            'straddle_pnl': sp, 'hedge_pnl': open_trade['hedge_pnl'],
            'costs': p['STRADDLE_COST'], 'realized_pnl': total,
            'exit_reason': 'end_of_data', 'er_entry': open_trade['er'],
            'duration_mins': (last['datetime']-open_trade['entry_dt']).total_seconds()/60,
            'hedge_count': len(open_trade['hedges']),
            'hedge_revert': sum(1 for h in open_trade['hedges'] if h.exit_reason=='spot_revert'),
            'hedge_sl': sum(1 for h in open_trade['hedges'] if h.exit_reason=='sl'),
            'trade_window': open_trade['trade_window'],
            'window_idx': open_trade['window_idx'],
        })

    out = pd.DataFrame(trades)
    if not out.empty:
        out['cumulative_pnl'] = out['realized_pnl'].cumsum()
    return out


# ═════════════════════════════════════════════════════════════════════
#  PERFORMANCE METRICS
# ═════════════════════════════════════════════════════════════════════

def print_metrics(df: pd.DataFrame):
    if df.empty:
        print("No trades."); return
    pnl  = df['realized_pnl']
    wins = pnl[pnl > 0]; loss = pnl[pnl <= 0]
    cum  = pnl.cumsum(); mdd  = (cum - cum.cummax()).min()
    df2  = df.copy()
    df2['d'] = pd.to_datetime(df2['entry_dt']).dt.date
    daily  = df2.groupby('d')['realized_pnl'].sum()
    sharpe = daily.mean()/daily.std()*np.sqrt(252) if len(daily)>1 and daily.std()>0 else 0
    pf     = -wins.sum()/loss.sum() if loss.sum()<0 and wins.sum()>0 else 0

    print(f"\n{'═'*60}")
    print(f"  PERFORMANCE SUMMARY — STRADDLE + DELTA HEDGE")
    print(f"{'═'*60}")
    print(f"  Trades            : {len(df)}")
    print(f"  Win Rate          : {len(wins)/len(pnl)*100:.1f}%")
    print(f"  Total P&L         : ₹{pnl.sum():,.0f}")
    print(f"  Profit Factor     : {pf:.2f}")
    print(f"  Sharpe (ann.)     : {sharpe:.2f}")
    print(f"  Max Drawdown      : ₹{mdd:,.0f}")
    print(f"  Avg Win           : ₹{wins.mean() if len(wins) else 0:,.0f}")
    print(f"  Avg Loss          : ₹{loss.mean() if len(loss) else 0:,.0f}")
    print(f"  Straddle P&L      : ₹{df['straddle_pnl'].sum():,.0f}")
    print(f"  Hedge P&L         : ₹{df['hedge_pnl'].sum():,.0f}")
    print(f"  Total Costs       : ₹{df['costs'].sum():,.0f}")
    print(f"  Avg Duration      : {df['duration_mins'].mean():.0f} min")
    print(f"  Total Hedges      : {df['hedge_count'].sum()}")
    print(f"  Avg Hedges/Trade  : {df['hedge_count'].mean():.1f}")

    print(f"\n  Exit breakdown:")
    for r, cnt in df['exit_reason'].value_counts().items():
        sub = df[df['exit_reason']==r]
        print(f"    {r:15s}: {cnt:3d}  avg ₹{sub['realized_pnl'].mean():,.0f}"
              f"  avg_dur {sub['duration_mins'].mean():.0f}min")

    print(f"\n  Per-window breakdown:")
    print(f"  {'Window':<16s} {'Trades':>7s} {'WinRate':>8s} "
          f"{'TotalPnL':>10s} {'AvgPnL':>9s} {'AvgDur':>8s} {'Hedges':>8s}")
    print(f"  {'─'*70}")
    for window, grp in df.groupby('trade_window'):
        wp  = grp['realized_pnl']
        wr  = (wp > 0).mean() * 100
        hc  = grp['hedge_count'].sum()
        dur = grp['duration_mins'].mean()
        print(f"  {window:<16s} {len(grp):>7d} {wr:>7.1f}% "
              f"{wp.sum():>10,.0f} {wp.mean():>9,.0f} "
              f"{dur:>7.0f}m {hc:>8d}")
    print(f"{'═'*60}")


# ═════════════════════════════════════════════════════════════════════
#  CHARTS
# ═════════════════════════════════════════════════════════════════════

DARK='#0d1117'; PANEL='#161b22'; GREEN='#39d353'; RED='#f85149'
BLUE='#58a6ff'; ORANGE='#e3b341'; PURPLE='#bc8cff'; GRAY='#8b949e'; TEXT='#c9d1d9'

def _ax(ax, title=''):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=GRAY, labelsize=8)
    for s in ax.spines.values(): s.set_edgecolor('#30363d')
    if title: ax.set_title(title, color=TEXT, fontsize=9, fontweight='bold', pad=5)
    ax.grid(alpha=0.15, color=GRAY, lw=0.4)


def plot_results(df: pd.DataFrame, sig: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    pnl  = df['realized_pnl']
    cum  = pnl.cumsum(); dd = cum - cum.cummax()
    wins = pnl[pnl > 0]; loss = pnl[pnl <= 0]

    df2 = df.copy()
    df2['d'] = pd.to_datetime(df2['entry_dt']).dt.date
    daily   = df2.groupby('d')['realized_pnl'].sum()
    sharpe  = daily.mean()/daily.std()*np.sqrt(252) if daily.std()>0 else 0
    pf      = -wins.sum()/loss.sum() if loss.sum()<0 and wins.sum()>0 else 0
    wr      = len(wins)/len(pnl)*100
    total   = pnl.sum()
    pc      = GREEN if total >= 0 else RED

    fig = plt.figure(figsize=(24, 28), facecolor=DARK)
    gs  = gridspec.GridSpec(5, 3, figure=fig, hspace=0.45, wspace=0.30,
                            top=0.93, bottom=0.04, left=0.06, right=0.97)

    fig.text(0.5, 0.964,
             f'NIFTY STRADDLE + DELTA HEDGE  |  P&L: ₹{total:,.0f}  |  '
             f'Sharpe: {sharpe:.2f}  |  Win: {wr:.1f}%  |  PF: {pf:.2f}  |  '
             f'Trades: {len(df)}  |  Hedges: {df["hedge_count"].sum()}',
             ha='center', color=pc, fontsize=12, fontweight='bold',
             fontfamily='monospace')

    # 1. Equity curve
    ax = fig.add_subplot(gs[0, :2])
    ax.fill_between(range(len(cum)), cum, where=cum>=0, color=GREEN, alpha=0.25)
    ax.fill_between(range(len(cum)), cum, where=cum< 0, color=RED,   alpha=0.25)
    ax.plot(cum.values, color=pc, lw=2)
    ax.axhline(0, color=GRAY, lw=0.6, ls='--', alpha=0.5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f'₹{v:,.0f}'))
    ax.set_xlabel('Trade #', color=GRAY)
    _ax(ax, '📈 Cumulative P&L')

    # 2. Drawdown
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.fill_between(range(len(dd)), dd, color=RED, alpha=0.55)
    ax2.plot(dd.values, color=RED, lw=0.8)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f'₹{v:,.0f}'))
    _ax(ax2, '📉 Drawdown')

    # 3. P&L distribution
    ax3 = fig.add_subplot(gs[1, 0])
    pv = pnl.values
    b  = np.linspace(pv.min()*1.1, pv.max()*1.1, 35)
    ax3.hist(pv[pv>=0], bins=b, color=GREEN, alpha=0.8, label=f'Win {len(wins)}')
    ax3.hist(pv[pv< 0], bins=b, color=RED,   alpha=0.8, label=f'Loss {len(loss)}')
    ax3.axvline(pv.mean(), color=ORANGE, lw=1.5, ls=':', label=f'Avg ₹{pv.mean():.0f}')
    ax3.axvline(0, color=GRAY, lw=0.8, ls='--')
    ax3.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT)
    _ax(ax3, '📊 P&L Distribution')

    # 4. Straddle P&L vs Hedge P&L scatter
    ax4 = fig.add_subplot(gs[1, 1])
    sc = ax4.scatter(df['straddle_pnl'], df['hedge_pnl'],
                     c=df['realized_pnl'], cmap='RdYlGn', s=35, alpha=0.8)
    ax4.axhline(0, color=GRAY, lw=0.5, ls='--')
    ax4.axvline(0, color=GRAY, lw=0.5, ls='--')
    plt.colorbar(sc, ax=ax4, label='Total P&L (₹)')
    ax4.set_xlabel('Straddle P&L (₹)', color=GRAY)
    ax4.set_ylabel('Hedge P&L (₹)', color=GRAY)
    _ax(ax4, '🔀 Straddle vs Hedge P&L')

    # 5. P&L by Trade Window (bar)
    ax5 = fig.add_subplot(gs[1, 2])
    if 'trade_window' in df.columns:
        wp = df.groupby('trade_window')['realized_pnl'].sum()
        colors = [GREEN if v >= 0 else RED for v in wp.values]
        ax5.barh(range(len(wp)), wp.values, color=colors, alpha=0.85)
        ax5.set_yticks(range(len(wp)))
        ax5.set_yticklabels(wp.index, fontsize=8)
        ax5.axvline(0, color=GRAY, lw=0.5, ls='--')
        ax5.xaxis.set_major_formatter(FuncFormatter(lambda v,_: f'₹{v:,.0f}'))
    _ax(ax5, '⏰ P&L by Trade Window')

    # 6. Exit reasons pie
    ax6 = fig.add_subplot(gs[2, 0])
    ec  = df['exit_reason'].value_counts()
    pal = [GREEN, ORANGE, RED, BLUE, PURPLE, GRAY, '#ff6b9d']
    ax6.pie(ec.values, labels=ec.index, colors=pal[:len(ec)],
            autopct='%1.0f%%', pctdistance=0.72, startangle=140,
            textprops={'color': TEXT, 'fontsize': 7})
    ax6.set_facecolor(PANEL)
    ax6.set_title('Exit Reasons', color=TEXT, fontsize=9, fontweight='bold')

    # 7. Monthly P&L bar
    ax7 = fig.add_subplot(gs[2, 1:])
    df2['month'] = pd.to_datetime(df2['entry_dt']).dt.to_period('M')
    monthly = df2.groupby('month')['realized_pnl'].sum()
    cm = [GREEN if v>=0 else RED for v in monthly.values]
    ax7.bar(range(len(monthly)), monthly.values, color=cm, alpha=0.88)
    ax7.set_xticks(range(len(monthly)))
    ax7.set_xticklabels([str(m) for m in monthly.index], rotation=45, fontsize=7)
    ax7.axhline(0, color=GRAY, lw=0.5)
    ax7.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f'₹{v/1e3:.0f}k'))
    _ax(ax7, '📅 Monthly P&L')

    # 8. Hedge count vs P&L
    ax8 = fig.add_subplot(gs[3, 0])
    ax8.scatter(df['hedge_count'], df['realized_pnl'],
                c=df['realized_pnl'], cmap='RdYlGn', s=30, alpha=0.8)
    ax8.axhline(0, color=GRAY, lw=0.5, ls='--')
    ax8.set_xlabel('Hedge Count', color=GRAY); ax8.set_ylabel('P&L (₹)', color=GRAY)
    _ax(ax8, '🔄 Hedge Count vs P&L')

    # 9. Duration distribution
    ax9 = fig.add_subplot(gs[3, 1])
    ax9.hist(df['duration_mins'], bins=30, color=BLUE, alpha=0.8)
    ax9.axvline(df['duration_mins'].mean(), color=ORANGE, lw=1.5, ls=':',
                label=f'Avg {df["duration_mins"].mean():.0f} min')
    ax9.axvline(P['THETA_MAX_MIN'], color=RED, lw=1, ls='--',
                label=f'Theta stop {P["THETA_MAX_MIN"]}m')
    ax9.axvline(P['FLAT_STOP_MINS'], color=GRAY, lw=1, ls='-.',
                label=f'Flat stop {P["FLAT_STOP_MINS"]}m')
    ax9.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT)
    ax9.set_xlabel('Duration (min)', color=GRAY)
    _ax(ax9, '⏱ Trade Duration Distribution')

    # 10. IV vs RV over time
    ax10 = fig.add_subplot(gs[3, 2])
    if 'iv_dec' in sig.columns and 'rv' in sig.columns:
        iv_ts = sig.set_index('datetime')['iv_dec'].dropna() * 100
        rv_ts = sig.set_index('datetime')['rv'].dropna() * 100
        ax10.plot(iv_ts.index[::10], iv_ts.values[::10], color=ORANGE, lw=0.5, alpha=0.9, label='IV%')
        ax10.plot(rv_ts.index[::10], rv_ts.values[::10], color=GREEN,  lw=0.5, alpha=0.7, label='RV%')
        entries = sig[sig['signal']]
        ax10.scatter(entries['datetime'], entries['iv_pct'],
                     color=BLUE, s=4, alpha=0.7, zorder=3, label='Entry')
        ax10.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT)
        ax10.xaxis.set_major_formatter(mdates.DateFormatter('%b%y'))
    _ax(ax10, '🌡 IV% vs RV%')

    # 11. Key metrics table
    ax11 = fig.add_subplot(gs[4, :])
    ax11.axis('off'); ax11.set_facecolor(PANEL)
    ax11.text(0.02, 0.95, '📋 KEY METRICS', transform=ax11.transAxes,
              color=TEXT, fontsize=9, fontweight='bold')

    # Two-column metrics layout
    rows_l = [
        ('Trades',         str(len(df))),
        ('Win Rate',       f'{wr:.1f}%'),
        ('Total P&L',      f'₹{total:,.0f}'),
        ('Profit Factor',  f'{pf:.2f}'),
        ('Sharpe',         f'{sharpe:.2f}'),
        ('Max Drawdown',   f'₹{dd.min():,.0f}'),
    ]
    rows_r = [
        ('Avg Win',        f'₹{wins.mean() if len(wins) else 0:,.0f}'),
        ('Avg Loss',       f'₹{loss.mean() if len(loss) else 0:,.0f}'),
        ('Straddle P&L',   f'₹{df["straddle_pnl"].sum():,.0f}'),
        ('Hedge P&L',      f'₹{df["hedge_pnl"].sum():,.0f}'),
        ('Avg Duration',   f'{df["duration_mins"].mean():.0f} min'),
        ('Avg Hedges',     f'{df["hedge_count"].mean():.1f}'),
    ]
    y = 0.78
    for (k1,v1),(k2,v2) in zip(rows_l, rows_r):
        ax11.text(0.02, y, k1, transform=ax11.transAxes, color=GRAY, fontsize=8)
        ax11.text(0.16, y, v1, transform=ax11.transAxes,
                  color=GREEN if '₹' in v1 and '-' not in v1 and 'Loss' not in k1 and 'DD' not in k1 else TEXT,
                  fontsize=8, fontweight='bold')
        ax11.text(0.52, y, k2, transform=ax11.transAxes, color=GRAY, fontsize=8)
        ax11.text(0.66, y, v2, transform=ax11.transAxes,
                  color=GREEN if '₹' in v2 and '-' not in v2 and 'Loss' not in k2 else TEXT,
                  fontsize=8, fontweight='bold')
        y -= 0.13

    path = os.path.join(out_dir, 'backtest_straddle_hedge.png')
    plt.savefig(path, dpi=145, bbox_inches='tight', facecolor=DARK)
    print(f"  Chart → {path}")
    plt.close()


# ═════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════

def run(data_1min=DATA_1MIN, out_dir=OUT_DIR, params=None,
        trade_windows=None, window_max=None):
    p   = params       or P
    tw  = trade_windows or TRADE_WINDOWS
    wm  = window_max    or WINDOW_MAX_TRADES
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  NIFTY GAMMA SCALPING — STRADDLE + DELTA HEDGE")
    print(f"{'═'*60}")
    print(f"\n  ⏰ Trade Windows:")
    for i, (s, e) in enumerate(tw):
        print(f"     [{i+1}] {s} – {e}  (cap: {wm[i]} trades)")
    print(f"\n  DTE range: {p['DTE_MIN']}–{p['DTE_MAX']}")
    print(f"  Eligible weekdays:")
    for wd, name in enumerate(['Monday','Tuesday','Wednesday','Thursday','Friday']):
        d = (3 - wd) % 7 or 7
        ok = '✅' if p['DTE_MIN'] <= d <= p['DTE_MAX'] else '❌'
        print(f"    {ok} {name} (DTE={d})")

    raw  = load_and_prepare(data_1min, '1MIN', keep=2)

    print(f"\n{'─'*58}\n  Building ATM straddle\n{'─'*58}")
    strad = make_straddle(raw)

    print(f"\n{'─'*58}\n  Computing indicators & signals\n{'─'*58}")
    sig   = add_indicators(strad, p)
    sig   = add_signals(sig, p, tw)

    print(f"\n{'═'*58}\n  Running backtest (STRADDLE + HEDGE)\n{'═'*58}")
    trades = run_backtest(sig, p, tw, wm)

    if trades.empty:
        print("  No trades generated.")
        return trades

    csv_path = os.path.join(out_dir, 'trades_straddle_hedge.csv')
    trades.to_csv(csv_path, index=False)
    print(f"  Trades CSV → {csv_path}  ({len(trades)} trades)")

    print_metrics(trades)
    plot_results(trades, sig, out_dir)

    return trades


if __name__ == '__main__':
    trades = run()
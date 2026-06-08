"""Shared helpers: rupee formatting + dataset loading."""
import pandas as pd
import config as C


def inr(x, paise=False):
    """Indian-grouped rupee string. e.g. -452300 -> 'Rs -4,52,300'."""
    val = float(x)
    neg = val < 0
    whole = int(abs(val))
    s = str(whole)
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:]); head = head[:-2]
        parts.insert(0, head)
        s = ",".join(parts) + "," + tail
    out = f"Rs {'-' if neg else ''}{s}"
    if paise:
        out += f".{int(round((abs(val)-whole)*100)):02d}"
    return out


def load():
    port = pd.read_csv(C.PORT_RET_CSV, index_col=0, parse_dates=True)["portfolio_return"]
    assets = pd.read_csv(C.RETURNS_CSV, index_col=0, parse_dates=True)
    wdf = pd.read_csv(C.WEIGHTS_CSV, index_col=0)
    return port, assets, wdf

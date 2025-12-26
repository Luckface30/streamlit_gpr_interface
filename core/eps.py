# core/eps.py
import numpy as np
import pandas as pd
from typing import Tuple, Optional

# ============================================================
# Config de quantification
# ============================================================
STEP = 0.2  # pas de variation désiré (0.2)

def _quantize(x: float, step: float = STEP) -> float:
    """
    Quantifie x au pas 'step' et formate à 1 décimale.
    Exemple: step=0.2 -> ... 4.2, 4.4, 4.6 ...
    """
    return round(round(x / step) * step, 1)

# Générateur aléatoire (non déterministe à chaque appel)
_rng = np.random.default_rng()

# ============================================================
# Tables intégrées (issues des Excel d’origine)
# ============================================================

# EPS1 (BS)
# Colonnes : observation, humidite, nature_colmatage, min, max
_EPS1_RULES = pd.DataFrame([
    ("glaiseuse/argileuse/remontee dans bs", "-",    "-",    8.0, 10.0),
    ("pollution diffuse dans bs",            "-",    "-",    4.5,  6.0),
    ("observation ne mentionne pas bs",      "sec",  "-",    3.5,  4.5),
    ("observation ne mentionne pas bs",      "hum",  "-",    4.5,  6.0),
    ("observation ne mentionne pas bs",      "sat",  "-",   11.0, 13.0),
])

# EPS2 (BC)
# Colonnes : observation, humidite, nature_colmatage, min, max
_EPS2_RULES = pd.DataFrame([
    ("glaiseuse/argileuse/remontee", "-",    "-",   11.0, 13.0),
    ("pollution diffuse dans bc",    "sec",  "n",    6.0,  7.5),
    ("pollution diffuse dans bc",    "hum",  "n",    8.0, 10.0),
    ("pollution diffuse dans bc",    "hum",  "n",    8.0, 10.0),  # doublon volontairement maintenu
    ("pollution diffuse dans bc",    "sat",  "n",   11.0, 13.0),
    ("pollution diffuse dans bc",    "hum",  "gl",  10.0, 14.0),
    ("autres",                       "-",    "-",    7.0,  9.0),
])

# ============================================================
# Matching générique
# ============================================================

def _match(df: pd.DataFrame, obs: str, hum: str, colm: str = "-") -> Tuple[float, float]:
    """
    Cherche dans la table la plage correspondant à (obs, hum, colm).
    """
    obs = (obs or "").lower()
    hum = (hum or "-").lower()
    colm = (colm or "-").lower()

    # Sélection par observation (contains)
    sub = df[df[0].apply(lambda s: s in obs)]
    if sub.empty:
        sub = df[df[0] == "autres"]
    if sub.empty:
        sub = df

    # Filtrage humidité + colmatage, avec jokers
    r = sub[(sub[1] == hum) & (sub[2] == colm)]
    if r.empty: r = sub[(sub[1] == hum) & (sub[2] == "-")]
    if r.empty: r = sub[(sub[1] == "-")   & (sub[2] == colm)]
    if r.empty: r = sub[(sub[1] == "-")   & (sub[2] == "-")]

    if r.empty:
        return (np.nan, np.nan)

    row = r.iloc[0]
    return float(row[3]), float(row[4])

# ============================================================
# API publique
# ============================================================

def get_eps1_interval(humidite_bs: str, obs: str) -> Tuple[Optional[float], Optional[float]]:
    """Retourne (min, max) de la plage ε1 pour la combinaison donnée."""
    a, b = _match(_EPS1_RULES, obs, humidite_bs)
    return (None if np.isnan(a) else a, None if np.isnan(b) else b)

def get_eps2_interval(humidite_bc: str, colmatage: str, obs: str) -> Tuple[Optional[float], Optional[float]]:
    """Retourne (min, max) de la plage ε2 pour la combinaison donnée."""
    a, b = _match(_EPS2_RULES, obs, humidite_bc, colmatage)
    return (None if np.isnan(a) else a, None if np.isnan(b) else b)

def get_eps1(humidite_bs: str, obs: str, jitter: float = 0.0) -> float:
    """
    Tirage **aléatoire** uniforme dans [min, max], puis quantification au pas 0.2.
    `jitter` conservé pour compatibilité mais ignoré.
    """
    a, b = _match(_EPS1_RULES, obs, humidite_bs)
    if np.isnan(a) or np.isnan(b) or b < a:
        return np.nan
    # tirage uniformément dans [a, b]
    val = a + _rng.random() * (b - a)
    # quantifie puis clamp dans [a, b]
    q = _quantize(val, STEP)
    return float(min(max(q, a), b))

def get_eps2(humidite_bc: str, colmatage: str, obs: str, jitter: float = 0.0) -> float:
    """
    Tirage **aléatoire** uniforme dans [min, max], puis quantification au pas 0.2.
    `jitter` conservé pour compatibilité mais ignoré.
    """
    a, b = _match(_EPS2_RULES, obs, humidite_bc, colmatage)
    if np.isnan(a) or np.isnan(b) or b < a:
        return np.nan
    val = a + _rng.random() * (b - a)
    q = _quantize(val, STEP)
    return float(min(max(q, a), b))
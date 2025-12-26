import math
import numpy as np
import pandas as pd
import unicodedata
from typing import Optional, Tuple

# ===============================
# Constantes
# ===============================
STEP = 0.2  # pas d'affichage pour EPS (mettre 0 pour aucun arrondi)
_rng = np.random.default_rng()

def _quantize(x: float, step: float) -> float:
    """Quantifie x au pas `step` avec un arrondi half-up (les .5 montent)."""
    if step <= 0:
        return float(x)
    u = x / step
    q_units = math.floor(u + 0.5) if u >= 0 else math.ceil(u - 0.5)
    q = q_units * step
    return round(q, 1 if step < 1 else 0)

# ===============================
# Normalisation des entrées
# ===============================
def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))

def _norm_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = _strip_accents(s)
    return " ".join(s.split())

def _norm_hum(s: Optional[str]) -> str:
    s = _norm_text(s)
    if s in ("", "-", "na", "nan", "none"): return "-"
    if s in ("h", "hum", "humide") or s.startswith("hum"): return "hum"
    if s in ("s", "sec", "seche", "sechette") or s.startswith("sec"): return "sec"
    if s.startswith("sat"): return "sat"
    return s

def _norm_colm(s: Optional[str]) -> str:
    s = _norm_text(s)
    if s in ("", "-", "na", "nan", "none"): return "-"
    if s.startswith("gl") or "argil" in s or "glais" in s: return "gl"
    NEG = {
        "n", "non", "aucun", "aucune", "aucuns", "aucunes",
        "sans", "0", "zero", "absent", "absence", "neant", "rien", "no", "none", "np"
    }
    if s in NEG: return "n"
    return s

# ===============================
# Normalisations d'observation (priorités)
# ===============================
def _norm_obs_bs(s: Optional[str]) -> str:
    """Clé BS avec priorité: clay>pollution, sinon 'ne mentionne pas'."""
    s = _norm_text(s)
    if s == "":
        return "observation ne mentionne pas bs"

    has_bs = "bs" in s
    has_pd = "pollution diffuse" in s
    has_clay = ("glaiseuse" in s) or ("argileuse" in s) or ("remontee" in s)

    if has_clay and has_bs:
        return "glaiseuse/argileuse/remontee dans bs"
    if has_pd and has_bs:
        return "pollution diffuse dans bs"
    if ("bc" in s) and not has_bs:
        return "observation ne mentionne pas bs"
    return "observation ne mentionne pas bs"

def _norm_obs_bc(s: Optional[str]) -> str:
    """Clé BC avec priorité: clay>pollution, sinon 'ne mentionne pas'."""
    s = _norm_text(s)
    if s == "":
        return "observation ne mentionne pas bc"

    has_bc = "bc" in s
    has_pd = "pollution diffuse" in s
    has_clay = ("glaiseuse" in s) or ("argileuse" in s) or ("remontee" in s)

    if has_clay and has_bc:
        return "glaiseuse/argileuse/remontee dans bc"
    if has_pd and has_bc:
        return "pollution diffuse dans bc"
    if ("bs" in s) and not has_bc:
        return "observation ne mentionne pas bc"
    return "observation ne mentionne pas bc"

# ===============================
# Tables de règles
# ===============================
_EPS1 = pd.DataFrame([
    ("remontee dans bs", "-",  "-",   8.0, 10.0, 22, 45),
    ("pollution diffuse dans bs",            "sec","-",   4.0,  5.5,  4, 10),
    ("pollution diffuse dans bs",            "hum","-",   4.5,  6.0,  4, 10),
    ("pollution diffuse dans bs",            "sat","-",  10.0, 14.0,  4, 10),
    ("pollution diffuse dans bs",            "-",  "-",   4.5,  6.0,  4, 10),
    ("observation ne mentionne pas bs",      "sec","-",   3.5,  4.5,  0,  2),
    ("observation ne mentionne pas bs",      "hum","-",   4.5,  6.0,  0,  2),
    ("observation ne mentionne pas bs",      "sat","-",  10.0, 14.0,  2,  4),
], columns=["obs","hum","colm","eps_min","eps_max","ind_min","ind_max"])

_EPS2 = pd.DataFrame([
    ("glaiseuse/argileuse/remontee dans bc", "-",   "-", 10.0, 14.0, 55, 70),
    ("glaiseuse/argileuse/remontee dans bc", "-",   "gl",16.0, 20.0, 55, 70),
    ("glaiseuse/argileuse/remontee dans bc", "hum", "gl",16.0, 20.0, 55, 70),

    ("pollution diffuse dans bc", "sec", "-", 6.0,  7.5, 12, 20),
    ("pollution diffuse dans bc", "hum", "-", 8,  10, 12, 20),
    ("pollution diffuse dans bc", "sat", "-", 10.0, 14.0, 22, 45),
    ("pollution diffuse dans bc", "-",   "gl",10.0, 14.0, 55, 70),
    ("pollution diffuse dans bc", "sec", "n", 6.0,  7.5, 12, 20),
    ("pollution diffuse dans bc", "hum", "n", 8.0, 10.0, 12, 20),
    ("pollution diffuse dans bc", "sat", "n", 12.0, 16.0, 22, 45),

    ("observation ne mentionne pas bc", "hum", "gl",14.0, 18.0, 55, 70),
    ("observation ne mentionne pas bc", "sat", "gl",16.0, 20.0, 55, 70),
    ("observation ne mentionne pas bc", "-",   "n", 6.0,  7.5, 12, 20),
    ("observation ne mentionne pas bc", "hum", "n", 8.0,  10, 12, 20),
    ("observation ne mentionne pas bc", "-",   "-", 6.0,  7.5, 12, 20),  # fallback
], columns=["obs","hum","colm","eps_min","eps_max","ind_min","ind_max"])

# ===============================
# Matching
# ===============================
def _find_rule(
    df: pd.DataFrame,
    obs: str,
    hum: str,
    colm: str,
    norm_obs_fn=lambda x: x
) -> Optional[pd.Series]:
    obs, hum, colm = norm_obs_fn(obs), _norm_hum(hum), _norm_colm(colm)
    sub = df[df["obs"] == obs]
    if sub.empty:
        return None
    subset = sub[(sub["hum"] == hum) & (sub["colm"] == colm)]
    if subset.empty:
        subset = sub[(sub["hum"] == hum) & (sub["colm"] == "-")]
    if subset.empty:
        subset = sub[(sub["hum"] == "-") & (sub["colm"] == colm)]
    if subset.empty:
        subset = sub[(sub["hum"] == "-") & (sub["colm"] == "-")]
    if subset.empty:
        return None
    return subset.iloc[0]

# ===============================
# Interpolation ENTIER (proportionnelle)
# ===============================
def _map_eps_to_ind_linear_int(eps_value: float, lo: float, hi: float,
                               ind_lo: float, ind_hi: float) -> int:
    """EPS∈[lo,hi] -> ENTIER arrondi dans [ind_lo,ind_hi] par interpolation linéaire."""
    if hi <= lo:
        return int(round(ind_lo))
    t = (eps_value - lo) / (hi - lo)
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    ind_float = ind_lo + t * (ind_hi - ind_lo)
    ind = int(np.floor(ind_float + 0.5))
    lo_i, hi_i = int(round(min(ind_lo, ind_hi))), int(round(max(ind_lo, ind_hi)))
    return max(lo_i, min(hi_i, ind))

# ===============================
# Tirage couplé (IND = f(EPS))
# ===============================
def _sample_eps_and_ind(row: pd.Series, rng=_rng) -> Tuple[float, int]:
    """Tire EPS ~ U[eps_min,eps_max], quantifie (STEP), puis calcule IND depuis CET EPS."""
    lo, hi = float(row.eps_min), float(row.eps_max)
    ind_lo, ind_hi = float(row.ind_min), float(row.ind_max)

    eps_raw = lo if hi <= lo else lo + rng.random() * (hi - lo)  # seul aléa : EPS
    eps = _quantize(eps_raw, STEP) if hi > 0 else _quantize(lo, STEP)
    eps = min(max(eps, lo), hi)

    ind = _map_eps_to_ind_linear_int(eps, lo, hi, ind_lo, ind_hi)
    return eps, ind

# ===============================
# API publique — tirage couplé
# ===============================
def sample_eps1_with_indicator(hum: Optional[str], obs: Optional[str], rng=_rng) -> Tuple[float, int]:
    """EPS1 (BS) : (eps, ind). Fallback si aucune règle, selon humidité (avec cas '-' => sec)."""
    row = _find_rule(_EPS1, obs, hum, "-", norm_obs_fn=_norm_obs_bs)
    if row is None:
        # Normalisations
        h = _norm_hum(hum)              # 'sec' | 'hum' | 'sat' | '-'
        obs_bs = _norm_obs_bs(obs)      # 'observation ne mentionne pas bs' si vide

        # Métier: hum='-' & obs vide -> traiter comme 'sec'
        if h == "-" and obs_bs == "observation ne mentionne pas bs":
            h = "sec"

        defaults = {
            "sec": (3.5, 4.5, 0, 2),
            "hum": (4.5, 6.0, 0, 2),
            "sat": (10.0, 14.0, 2, 4),
        }
        eps_lo, eps_hi, ind_lo, ind_hi = defaults.get(h, (3.5, 4.5, 0, 2))  # défaut = sec
        row = pd.Series({"eps_min": eps_lo, "eps_max": eps_hi, "ind_min": ind_lo, "ind_max": ind_hi})
    return _sample_eps_and_ind(row, rng=rng)

def sample_eps2_with_indicator(hum: Optional[str], colm: Optional[str], obs: Optional[str], rng=_rng) -> Tuple[float, int]:
    """EPS2 (BC) : (eps, ind). Fallback 6.0–7.5 ; 12–20 si aucune règle ne matche."""
    row = _find_rule(_EPS2, obs, hum, colm, norm_obs_fn=_norm_obs_bc)
    if row is None:
        row = pd.Series({"eps_min": 6.0, "eps_max": 7.5, "ind_min": 12, "ind_max": 20})
    return _sample_eps_and_ind(row, rng=rng)

# ===============================
# IND pour un EPS donné (déterministe)
# ===============================
def indicator_from_given_eps(eps_value: float,
                             hum: Optional[str],
                             colm: Optional[str],
                             obs: Optional[str],
                             which: int = 2) -> Optional[int]:
    """Calcule l'IND ENTIER à partir d'un EPS fourni (valeur affichée)."""
    if which == 1:
        row = _find_rule(_EPS1, obs, hum, "-", norm_obs_fn=_norm_obs_bs)
        if row is None:
            h = _norm_hum(hum)
            obs_bs = _norm_obs_bs(obs)
            if h == "-" and obs_bs == "observation ne mentionne pas bs":
                h = "sec"
            defaults = {
                "sec": (3.5, 4.5, 0, 2),
                "hum": (4.5, 6.0, 0, 2),
                "sat": (10.0, 14.0, 2, 4),
            }
            eps_lo, eps_hi, ind_lo, ind_hi = defaults.get(h, (3.5, 4.5, 0, 2))
            row = pd.Series({"eps_min": eps_lo, "eps_max": eps_hi, "ind_min": ind_lo, "ind_max": ind_hi})
    else:
        row = _find_rule(_EPS2, obs, hum, colm, norm_obs_fn=_norm_obs_bc)
        if row is None:
            row = pd.Series({"eps_min": 6.0, "eps_max": 7.5, "ind_min": 12, "ind_max": 20})

    lo, hi = float(row.eps_min), float(row.eps_max)
    ind_lo, ind_hi = float(row.ind_min), float(row.ind_max)
    eps = min(max(float(eps_value), lo), hi)
    return _map_eps_to_ind_linear_int(eps, lo, hi, ind_lo, ind_hi)

# ===============================
# Utilitaires DataFrame
# ===============================
def fill_eps_ind_for_df(df: pd.DataFrame,
                        hum_col="Humidité Ballast sain",
                        colm_col="Nature du colmatage",
                        obs_col="Observations",
                        rng_seed: Optional[int] = None) -> pd.DataFrame:
    """Remplit Eps_1/Ind_1 et Eps_2/Ind_2 pour chaque ligne (IND=f(EPS))."""
    rng = np.random.default_rng(rng_seed) if rng_seed is not None else _rng

    def draw_row(r):
        hum, colm, obs = r[hum_col], r[colm_col], r[obs_col]
        eps1, ind1 = sample_eps1_with_indicator(hum, obs, rng=rng)
        eps2, ind2 = sample_eps2_with_indicator(hum, colm, obs, rng=rng)
        return pd.Series({"Eps_1": eps1, "Ind_1": ind1, "Eps_2": eps2, "Ind_2": ind2})

    df[["Eps_1","Ind_1","Eps_2","Ind_2"]] = df.apply(draw_row, axis=1)
    # df["Ind_1"] = pd.to_numeric(df["Ind_1"], errors="coerce").astype("Int64")
    # df["Ind_2"] = pd.to_numeric(df["Ind_2"], errors="coerce").astype("Int64")
    return df

def recompute_ind_from_eps(df: pd.DataFrame,
                           hum_col="Humidité Ballast sain",
                           colm_col="Nature du colmatage",
                           obs_col="Observations") -> pd.DataFrame:
    """Répare Ind_1/Ind_2 à partir de Eps_1/Eps_2 existants (déterministe, même mapping)."""
    def fix_row(r):
        hum, colm, obs = r[hum_col], r[colm_col], r[obs_col]
        i1 = indicator_from_given_eps(r["Eps_1"], hum, None, obs, which=1)
        i2 = indicator_from_given_eps(r["Eps_2"], hum, colm, obs, which=2)
        return pd.Series({"Ind_1": i1, "Ind_2": i2})

    df[["Ind_1","Ind_2"]] = df.apply(fix_row, axis=1)
    # df["Ind_1"] = pd.to_numeric(df["Ind_1"], errors="coerce").astype("Int64")
    # df["Ind_2"] = pd.to_numeric(df["Ind_2"], errors="coerce").astype("Int64")
    return df

# ===============================
# Wrappers pour compatibilité avec core/interp.py
# ===============================
def get_eps1(hum: Optional[str], obs: Optional[str], rng_seed: Optional[int] = None) -> float:
    """Retourne uniquement l'EPS1 (float)."""
    rng = np.random.default_rng(rng_seed) if rng_seed is not None else _rng
    eps, _ = sample_eps1_with_indicator(hum, obs, rng=rng)
    return float(eps)

def get_eps2(hum: Optional[str], colm: Optional[str], obs: Optional[str], rng_seed: Optional[int] = None) -> float:
    """Retourne uniquement l'EPS2 (float)."""
    rng = np.random.default_rng(rng_seed) if rng_seed is not None else _rng
    eps, _ = sample_eps2_with_indicator(hum, colm, obs, rng=rng)
    return float(eps)

# ===============================
# API exportée
# ===============================
__all__ = [
    "sample_eps1_with_indicator", "sample_eps2_with_indicator",
    "indicator_from_given_eps", "fill_eps_ind_for_df", "recompute_ind_from_eps",
    "get_eps1", "get_eps2",
]

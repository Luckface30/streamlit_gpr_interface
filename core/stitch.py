# core/stitch.py
"""
Raboutage spatial (stitching) des interfaces sur toute la ligne.

Objectif
--------
À partir d'un DataFrame agrégé « par image » (colonnes `x_cm` +
`interface_*` issues des exports Mask R‑CNN), reconstruire des
« pistes » d'interface globales cohérentes spatialement, afin que
les mêmes couches conservent un identifiant stable sur tout le
tronçon.

Fonctions principales
---------------------
- stitch_interfaces_across_line(df, ...)
    Retourne (df_out, tracks_meta) où df_out = x_cm + interface_k (globaux).

Hypothèses & entrées
--------------------
- df contient au minimum `x_cm` et >=1 colonnes qui commencent par
  `interface_` (px ou ns, peu importe). Les valeurs manquantes sont NaN.
- `x_cm` est dans l'ordre croissant (sinon, on trie).

Stratégie
---------
1) Pour chaque abscisse, on récupère l'ensemble des profondeurs
   présentes (en ignorant le nom de la colonne locale). On ordonne
   ces profondeurs croissantes.
2) On fait évoluer un ensemble de « pistes actives ». À chaque pas,
   on apparie les points courants aux pistes par **proximité
   verticale** sous contrainte d'ordre (on interdit les croisements),
   avec un saut vertical maximum `max_jump_px`. Les points non appariés
   démarrent de nouvelles pistes ; les pistes non appariées survivent
   jusqu'à `max_gap_pts` pas, puis s'éteignent.
3) À la fin, on renumérote les pistes par profondeur médiane globale
   pour produire `interface_0, interface_1, ...` stables.

Remarque :
- L'algo est déterministe et O(N * K^2) (K ~ nb d'interfaces < 6),
  ce qui est négligeable au regard de la taille de ligne.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass
class Track:
    id: int
    xs_idx: List[int] = field(default_factory=list)  # indices (lignes du df trié)
    ys: List[float] = field(default_factory=list)
    last_idx: Optional[int] = None
    last_y: Optional[float] = None
    alive_gap: int = 0  # nombre de pas depuis le dernier appariement

    def alive(self) -> bool:
        return self.last_idx is not None

    def can_take(self, idx: int, y: float, max_jump: float, max_gap_pts: int) -> bool:
        if self.last_y is None:
            return True
        gap_ok = (idx - (self.last_idx or idx)) <= max_gap_pts
        return gap_ok and (abs(y - self.last_y) <= max_jump)

    def push(self, idx: int, y: float):
        self.xs_idx.append(idx)
        self.ys.append(float(y))
        self.last_idx = idx
        self.last_y = float(y)
        self.alive_gap = 0


def _sorted_interfaces_per_row(df: pd.DataFrame, prefix: str) -> Tuple[np.ndarray, List[List[float]]]:
    x = df["x_cm"].to_numpy()
    interf_cols = [c for c in df.columns if c.startswith(prefix)]
    mats: List[List[float]] = []
    for _, row in df[interf_cols].iterrows():
        vals = [float(v) for v in row.to_list() if pd.notna(v)]
        vals.sort()
        mats.append(vals)
    return x, mats


def _order_preserving_greedy(tracks: List[Track], points: List[float], idx: int,
                             max_jump: float, max_gap_pts: int) -> Tuple[List[Tuple[int, int]], List[int]]:
    """Affectation gloutonne qui préserve l'ordre vertical.

    - `tracks` : pistes actives triées par leur `last_y` croissant
    - `points` : profondeurs courantes triées croissantes
    Retourne :
      - matches : liste de paires (i_track, j_point)
      - new_points_idx : indices de points non appariés (naissances)
    """
    matches: List[Tuple[int, int]] = []
    used_tracks: set = set()
    used_points: set = set()

    # On parcourt les points du haut vers le bas, et on leur associe la
    # piste la plus proche qui ne violera pas l'ordre global.
    for j, y in enumerate(points):
        best_i, best_cost = None, None
        last_taken_y = tracks[matches[-1][0]].last_y if matches else -np.inf
        for i, tr in enumerate(tracks):
            if i in used_tracks:
                continue
            if tr.last_y is None:
                # une nouvelle piste « vierge » peut prendre le point, mais on préfère
                # d'abord les pistes réelles pour la stabilité ; on considère cost inf.
                continue
            if tr.last_y < last_taken_y:
                # préserver l'ordre (pas de croisement avec l'affectation précédente)
                continue
            if not tr.can_take(idx, y, max_jump=max_jump, max_gap_pts=max_gap_pts):
                continue
            cost = abs(y - tr.last_y)
            if (best_cost is None) or (cost < best_cost):
                best_cost = cost
                best_i = i
        if best_i is not None:
            matches.append((best_i, j))
            used_tracks.add(best_i)
            used_points.add(j)

    new_points_idx = [j for j in range(len(points)) if j not in used_points]
    return matches, new_points_idx


def stitch_interfaces_across_line(
    df: pd.DataFrame,
    interface_prefix: str = "interface_",
    max_jump_px: float = 12.0,
    max_gap_pts: int = 2,
    min_track_len_pts: int = 10,
    smoothing_window: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruit des pistes d'interface globales sur toute la ligne.

    Paramètres
    ---------
    df : DataFrame (colonnes `x_cm` + `interface_*`).
    max_jump_px : écart vertical max entre deux pas consécutifs pour la même piste.
    max_gap_pts : nb max de pas sans observation avant extinction de la piste.
    min_track_len_pts : longueur minimale (en points) pour garder une piste.
    smoothing_window : fenêtre (impair) pour un lissage médian intrasegment.

    Sorties
    -------
    df_out : DataFrame avec `x_cm` + `interface_k` globaux
    tracks_meta : DataFrame méta par piste (longueur, coût moyen, bornes, médiane y)
    """
    if df is None or df.empty or "x_cm" not in df.columns:
        return df, pd.DataFrame()

    # 1) Prépare / trie
    work = df.copy()
    work = work.sort_values("x_cm").reset_index(drop=True)
    x, per_row_points = _sorted_interfaces_per_row(work, interface_prefix)

    # 2) Boucle d'appariement
    tracks: List[Track] = []
    next_id = 0

    # initialiser avec 0 pistes actives
    for idx, points in enumerate(per_row_points):
        # trier pistes actives par last_y croissant
        active = [t for t in tracks if (t.last_idx is not None and (idx - t.last_idx) <= max_gap_pts)]
        active.sort(key=lambda t: (float("inf") if t.last_y is None else t.last_y))

        matches: List[Tuple[int, int]] = []
        births: List[int] = list(range(len(points)))  # défaut : toutes naissent
        if active and points:
            matches, births = _order_preserving_greedy(active, points, idx, max_jump=max_jump_px, max_gap_pts=max_gap_pts)

        # 2.a) Appliquer les matches (sur la vue active -> retrouver indices globaux)
        used_global_tracks: set = set()
        for i_active, j_point in matches:
            tr = active[i_active]
            tr.push(idx, points[j_point])
            used_global_tracks.add(tr.id)

        # 2.b) Naissances : créer des pistes pour points non appariés
        for j in births:
            y = points[j]
            t = Track(id=next_id)
            t.push(idx, y)
            tracks.append(t)
            next_id += 1

        # 2.c) Incrémenter les gaps pour les pistes non utilisées
        for tr in tracks:
            if tr.last_idx is None:
                continue
            if tr.last_idx != idx:
                tr.alive_gap += 1
                # extinction implicite quand trop de gaps ; on n'efface pas l'historique

    # 3) Filtrer les pistes trop courtes
    tracks = [t for t in tracks if len(t.xs_idx) >= max(1, int(min_track_len_pts))]
    if not tracks:
        # plus simple : renvoyer une copie de l'entrée
        return work[["x_cm"]].copy(), pd.DataFrame()

    # 4) Renumérotation par profondeur médiane globale
    order = np.argsort([np.nanmedian(t.ys) for t in tracks])
    tracks = [tracks[i] for i in order]

    # 5) Construire le DataFrame de sortie
    out = pd.DataFrame({"x_cm": x})

    def _smooth_segment(yv: np.ndarray, seg_mask: np.ndarray) -> np.ndarray:
        if smoothing_window is None or smoothing_window < 3 or smoothing_window % 2 == 0:
            return yv
        # appliquer médiane glissante seulement sur segments continus (gap=1)
        w = smoothing_window
        y = yv.copy()
        idxs = np.where(seg_mask)[0]
        if idxs.size == 0:
            return y
        # on segmente par ruptures >1
        cuts = np.where(np.diff(idxs) > 1)[0]
        starts = np.r_[0, cuts + 1]
        ends = np.r_[cuts, idxs.size - 1]
        for s, e in zip(starts, ends):
            seg = idxs[s:e + 1]
            if seg.size < 3:
                continue
            ys = y[seg].copy()
            # médiane glissante simple
            half = w // 2
            for k in range(seg.size):
                i0 = max(0, k - half)
                i1 = min(seg.size, k + half + 1)
                y[seg[k]] = float(np.median(ys[i0:i1]))
        return y

    meta_rows: List[Dict] = []
    for k, tr in enumerate(tracks):
        col = f"interface_{k}"
        yv = np.full(shape=len(x), fill_value=np.nan, dtype=float)
        yv[tr.xs_idx] = tr.ys

        # masque de continuité : indices consécutifs
        seg_mask = np.zeros_like(yv, dtype=bool)
        if tr.xs_idx:
            seg_mask[tr.xs_idx] = True
        yv = _smooth_segment(yv, seg_mask)
        out[col] = yv

        # méta
        meta_rows.append({
            "track_id": k,
            "orig_id": tr.id,
            "len_pts": len(tr.xs_idx),
            "x_cm_start": float(x[min(tr.xs_idx)]),
            "x_cm_end": float(x[max(tr.xs_idx)]),
            "y_median": float(np.nanmedian(tr.ys)),
        })

    tracks_meta = pd.DataFrame(meta_rows).sort_values("track_id").reset_index(drop=True)
    return out, tracks_meta


# Utilitaire convivial pour l'UI :

def stitch_and_merge_for_ui(
    df_raw: pd.DataFrame,
    interface_prefix: str = "interface_",
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prend un df « agrégé » (concat Excels) et renvoie un df prêt à tracer.

    - Conserve `x_cm` et remplace les colonnes locales par des `interface_k` globaux.
    - Les colonnes non-interfaces sont préservées si elles ne chevauchent pas.
    """
    if df_raw is None or df_raw.empty:
        return df_raw, pd.DataFrame()

    base_cols = [c for c in df_raw.columns if not c.startswith(interface_prefix)]
    stitched, meta = stitch_interfaces_across_line(df_raw, interface_prefix=interface_prefix, **kwargs)

    # fusionner x_cm + interfaces globaux ; on laisse l'UI faire renaming unité, filtres, etc.
    merged = pd.merge(df_raw[base_cols], stitched, on="x_cm", how="right")
    return merged, meta


import io
import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Outil: Tâches & Traitements Excel", layout="wide")

# ---------------------------
# Utils
# ---------------------------

def find_col(df, candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        for c in df.columns:
            if cand.lower() == c.lower():
                return c
        for low, orig in cols_lower.items():
            if cand.lower() in low:
                return orig
    return None

def make_unique(names):
    seen = {}
    out = []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}__{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    return out

def _clean_header_token(x):
    if x is None:
        return ""
    s = str(x).strip()
    if re.match(r"(?i)^unnamed.*", s) or s.lower() in {"nan", "none"}:
        return ""
    return s

def fuse_top_two_rows_as_header(df_raw):
    if df_raw.shape[0] < 2:
        return df_raw
    top = df_raw.iloc[0].tolist()
    sec = df_raw.iloc[1].tolist()
    headers = []
    for a, b in zip(top, sec):
        a_clean = _clean_header_token(a)
        b_clean = _clean_header_token(b)
        if a_clean and b_clean:
            h = f"{a_clean} {b_clean}"
        elif a_clean:
            h = a_clean
        elif b_clean:
            h = b_clean
        else:
            h = "col"
        headers.append(h.strip())
    import re as _re
    headers = [_re.sub(r"\s+", " ", h) for h in headers]
    headers = make_unique(headers)
    df = df_raw.iloc[2:].copy()
    df.columns = headers
    return df

def clean_numeric(df, cols):
    out = df.copy()
    if not cols:
        return out
    out[cols] = (
        out[cols]
        .replace('-', np.nan)
        .astype(str)
        .replace(r'\*', '', regex=True)
        .replace('', np.nan)
        .apply(pd.to_numeric, errors='coerce')
    )
    return out

def km_to_cm(series):
    return pd.to_numeric(series, errors='coerce') * 100_000.0

def ensure_sorted_unique_by_km(df, km_col):
    out = df.sort_values(km_col, kind="mergesort")
    out = out.drop_duplicates(subset=[km_col], keep="first")
    return out

# ε rules
import hashlib
def _stable_random_01(*values) -> float:
    def s(x):
        try:
            return "" if pd.isna(x) else str(x)
        except Exception:
            return str(x)
    h = hashlib.sha256(("|".join(s(v) for v in values)).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF

def get_eps1(humidite_bs, obs, jitter: float = 0.2) -> float:
    obs_s = str(obs).lower().strip()
    hum = str(humidite_bs).lower().strip()
    r = _stable_random_01(humidite_bs, obs)
    if any(kw in obs_s for kw in ["glaiseuse", "argileuse", "remontées argileuses"]) and "bs" in obs_s:
        base = 9.0
        return round(base + (r - 0.5) * jitter, 1)
    if "pollution diffuse" in obs_s and "bs" in obs_s:
        base = 4.7
        return round(base + (r - 0.5) * jitter, 1)
    if "bs" not in obs_s:
        if hum == "hum":
            base = 4.7
        elif hum == "sat":
            base = 12.0
        else:
            base = 4.0
        return round(base + (r - 0.5) * jitter, 1)
    return round(5.0 + (r - 0.5) * jitter, 1)

def get_eps2(humidite_bc, colmatage, obs, jitter: float = 0.2) -> float:
    obs_s = str(obs).lower().strip()
    hum = str(humidite_bc).lower().strip()
    colm = str(colmatage).lower().strip()
    r = _stable_random_01(humidite_bc, colmatage, obs)
    if any(kw in obs_s for kw in ["glaiseuse", "argileuse", "remontées argileuses"]) and (
        "bc" in obs_s or "bs" in obs_s or ("bc" not in obs_s and "bs" not in obs_s)
    ):
        base = 12.0
        return round(base + (r - 0.5) * jitter, 1)
    if "pollution diffuse" in obs_s and "bc" in obs_s and not any(kw in obs_s for kw in ["glaiseuse", "argileuse", "remontées argileuses"]):
        if hum == "sec":
            base = 6.7
        elif hum == "hum":
            base = 8.3
        elif hum == "sat":
            base = 12.0
        else:
            base = 6.7
        return round(base + (r - 0.5) * jitter, 1)
    if colm == "gl":
        base = 12.0
    elif colm == "n" and hum == "sec":
        base = 6.7
    elif colm == "n" and hum == "hum":
        base = 9.0
    else:
        base = 8.0
    return round(base + (r - 0.5) * jitter, 1)

def clean_and_interpolate_local(df, km_col, numeric_cols, position_filter_value, position_col, step_cm=10, half_window_cm=200):
    if position_col and position_filter_value is not None and position_filter_value != "— Aucun filtre —":
        df = df[df[position_col].astype(str) == str(position_filter_value)].copy()
    df = clean_numeric(df, [c for c in numeric_cols if c in df.columns])
    df = ensure_sorted_unique_by_km(df, km_col)
    x_cm = km_to_cm(df[km_col])
    xmin, xmax = np.nanmin(x_cm.values), np.nanmax(x_cm.values)
    valid = x_cm.notna().values
    x_base = x_cm.values[valid]
    if x_base.size < 2:
        st.warning("Pas assez de points valides pour interpoler (>= 2 requis).")
        return None
    sort_idx = np.argsort(x_base); x_base = x_base[sort_idx]
    interp_bases = {}
    for col in numeric_cols:
        if col not in df.columns:
            continue
        y = pd.to_numeric(df[col], errors="coerce").values
        yb = y[valid][sort_idx]
        interp_bases[col] = (x_base, yb)
    frames = []
    for i, xi in enumerate(x_cm.values):
        if np.isnan(xi): continue
        start = max(xmin, xi - half_window_cm); end = min(xmax, xi + half_window_cm)
        if end < start: continue
        x_win = np.arange(start, end + 0.1, step_cm)
        if x_win.size == 0: continue
        df_win = pd.DataFrame({km_col: x_win / 100_000.0, "anchor_idx": i, "anchor_Km": xi / 100_000.0})
        for col in numeric_cols:
            if col in interp_bases:
                xv, yv = interp_bases[col]
                df_win[col] = np.interp(x_win, xv, yv) if xv.size >= 2 else np.nan
        others = [c for c in df.columns if c not in numeric_cols + [km_col]]
        for col in others:
            df_win[col] = df.iloc[i][col]
        frames.append(df_win)
    if not frames: return None
    return pd.concat(frames, ignore_index=True)

# ---------------------------
# UI
# ---------------------------

st.title("📦 Mini outil (local) — Import • Fusion entêtes • Interpolation • ε₁/ε₂ • Export")

if "raw_df" not in st.session_state: st.session_state.raw_df = None
if "processed_df" not in st.session_state: st.session_state.processed_df = None

with st.sidebar:
    st.header("⚙️ Paramètres")
    step_cm = st.number_input("Pas d'interpolation (cm)", min_value=1, max_value=1000, value=10, step=1)
    half_window_cm = st.number_input("Fenêtre ± (cm)", min_value=10, max_value=10_000, value=200, step=10)
    st.caption("Coche 'Fusionner entêtes' si ton Excel a 2 lignes d'en-têtes.")

tab1, tab2, tab3, tab4 = st.tabs(["1) Import", "2) Interpolation", "3) ε₁/ε₂", "4) Export"])

with tab1:
    st.subheader("Importer un fichier")
    up = st.file_uploader("Excel ou CSV", type=["xlsx", "xls", "csv"])
    colA, colB = st.columns(2)
    with colA:
        merge_headers = st.checkbox("Fusionner les 2 premières lignes en en-têtes", value=True)
    with colB:
        skip_blank_top = st.checkbox("Ignorer lignes vides au-dessus", value=True)

    if up is not None:
        try:
            if up.name.lower().endswith(".csv"):
                raw = pd.read_csv(up, header=None) if merge_headers else pd.read_csv(up)
            else:
                raw = pd.read_excel(up, header=None) if merge_headers else pd.read_excel(up)
            if merge_headers and skip_blank_top:
                while len(raw) and raw.iloc[0].isna().all():
                    raw = raw.iloc[1:].reset_index(drop=True)
            df = fuse_top_two_rows_as_header(raw) if merge_headers else raw.copy()
            st.session_state.raw_df = df.copy(); st.session_state.processed_df = None
            st.success(f"Fichier chargé: {up.name} • {df.shape[0]} lignes, {df.shape[1]} colonnes")
            st.dataframe(df.head(50), use_container_width=True)
            km_guess = find_col(df, ["Km", "KM", "kilometre", "kilometer", "pk", "position_km"])
            pos_col_guess = find_col(df, ["Position", "position"])
            numeric_guess = [c for c in df.columns if c != km_guess and pd.api.types.is_numeric_dtype(df[c])]
            st.markdown("### Colonnes")
            c1, c2 = st.columns(2)
            with c1:
                km_col = st.selectbox("Colonne Km", options=list(df.columns), index=(list(df.columns).index(km_guess) if (km_guess in list(df.columns)) else 0))
            with c2:
                position_col = st.selectbox("Colonne Position (optionnel)", options=[None] + list(df.columns), index=(0 if pos_col_guess is None else (1 + list(df.columns).index(pos_col_guess))))
            if position_col:
                pos_values = ["— Aucun filtre —"] + sorted(df[position_col].dropna().astype(str).unique().tolist())
                position_filter_value = st.selectbox("Filtre Position", options=pos_values, index=(pos_values.index("O") if "O" in pos_values else 0))
            else:
                position_filter_value = None
            numeric_cols = st.multiselect("Colonnes numériques à interpoler", options=list(df.columns), default=numeric_guess)
            st.session_state._params = {"km_col": km_col, "position_col": position_col, "position_filter_value": position_filter_value, "numeric_cols": numeric_cols}
        except Exception as e:
            st.error(f"Erreur de lecture: {e}")

with tab2:
    st.subheader("Interpolation locale (±2m par défaut)")
    if st.session_state.raw_df is None:
        st.info("Charge d'abord un fichier.")
    else:
        p = st.session_state.get("_params", {})
        if not p.get("km_col"):
            st.warning("Sélectionne la colonne Km.")
        else:
            if st.button("🚀 Nettoyer + Interpoler"):
                df_proc = clean_and_interpolate_local(
                    df=st.session_state.raw_df.copy(),
                    km_col=p["km_col"],
                    numeric_cols=[c for c in p["numeric_cols"] if c != p["km_col"]],
                    position_filter_value=p["position_filter_value"],
                    position_col=p["position_col"],
                    step_cm=step_cm,
                    half_window_cm=half_window_cm,
                )
                if df_proc is None or df_proc.empty:
                    st.error("Aucun résultat généré.")
                else:
                    st.session_state.processed_df = df_proc
                    st.success(f"Nouveau tableau: {len(df_proc)} lignes.")
                    st.dataframe(df_proc.head(100), use_container_width=True)
            if st.session_state.processed_df is not None:
                try:
                    import plotly.express as px
                    km_col = p["km_col"]
                    num_cols = [c for c in st.session_state.processed_df.columns if c != km_col and pd.api.types.is_numeric_dtype(st.session_state.processed_df[c])]
                    if num_cols:
                        st.markdown("### Aperçu graphique")
                        fig = px.line(st.session_state.processed_df.head(5000), x=km_col, y=num_cols[:5])
                        fig.update_yaxes(autorange="reversed", title_text="Cote")
                        fig.update_xaxes(title_text="Position (km)")
                        st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass

with tab3:
    st.subheader("Assigner ε₁ / ε₂")
    if st.session_state.processed_df is None:
        st.info("Génère d'abord un tableau interpolé.")
    else:
        df = st.session_state.processed_df.copy()
        c_obs = st.selectbox("Colonne Observations", options=[None] + list(df.columns))
        c_hbs = st.selectbox("Colonne Humidité BS", options=[None] + list(df.columns))
        c_hbc = st.selectbox("Colonne Humidité BC", options=[None] + list(df.columns))
        c_colm = st.selectbox("Colonne Colmatage", options=[None] + list(df.columns))
        if st.button("🧪 Calculer ε₁/ε₂"):
            missing = [name for name, col in [("Observations", c_obs), ("Humidité BS", c_hbs), ("Humidité BC", c_hbc), ("Colmatage", c_colm)] if col is None]
            if missing:
                st.error(f"Colonnes manquantes: {', '.join(missing)}")
            else:
                df["Eps 1"] = df.apply(lambda r: get_eps1(r[c_hbs], r[c_obs]), axis=1)
                df["Eps 2"] = df.apply(lambda r: get_eps2(r[c_hbc], r[c_colm], r[c_obs]), axis=1)
                st.session_state.processed_df = df
                st.success("ε₁/ε₂ ajoutés.")
                st.dataframe(df.head(100), use_container_width=True)

with tab4:
    st.subheader("Export")
    if st.session_state.processed_df is None and st.session_state.raw_df is None:
        st.info("Rien à exporter.")
    else:
        choice = st.radio("Que veux-tu exporter ?", ["Tableau traité (si présent)", "Tableau brut (tel que chargé)"])
        if st.button("📥 Générer le fichier Excel"):
            df_to_save = st.session_state.processed_df if (choice == "Tableau traité (si présent)" and st.session_state.processed_df is not None) else st.session_state.raw_df
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df_to_save.to_excel(writer, index=False, sheet_name="data")
            buf.seek(0)
            st.download_button("Télécharger le fichier Excel", data=buf, file_name="resultat.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

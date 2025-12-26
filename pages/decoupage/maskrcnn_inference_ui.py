import os
import time
import streamlit as st

from core.mrcnn_infer import (
    list_image_files, classify_files_by_pk,
    run_inference_for_category, has_existing_inference
)

def inference_block():
    # --------------------------
    # Sidebar: chemins
    # --------------------------
    with st.sidebar:
        st.header("📁 Chemins")
        image_folder = st.text_input("Dossier d'images de test", value="test_simple")
        model_path   = st.text_input("Modèle Mask R-CNN (.pt)", value="models/maskrcnn.pt")
        output_root  = st.text_input("Dossier de sortie", value=os.path.join(image_folder, "predictions_maskrcnn"))
        st.caption("Les sorties seront séparées dans `ASC/`, `DESC/` et `FUSION/`.")

    # --------------------------
    # 1) Scan & classification
    # --------------------------
    st.markdown("### 1) Scanner & classer les images (PK croissants vs décroissants)")
    if st.button("🔎 Scanner le dossier"):
        if not os.path.isdir(image_folder):
            st.error(f"Dossier introuvable: {image_folder}")
            st.stop()
        files = list_image_files(image_folder)
        if not files:
            st.error("Aucune image .png/.jpg/.tif trouvée.")
            st.stop()

        asc_files, desc_files, unknown = classify_files_by_pk(files)
        st.session_state._scan = {
            "asc": asc_files,
            "desc": desc_files,
            "unknown": unknown,
            "image_folder": image_folder,
            "model_path": model_path,
            "output_root": output_root
        }
        st.cache_data.clear()
        st.session_state.df_asc_raw = None
        st.session_state.df_desc_raw = None

        st.success(f"Total: {len(files)} | ASC: {len(asc_files)} | DESC: {len(desc_files)} | Inconnus: {len(unknown)}")
        if unknown:
            st.warning("Noms non conformes (‘..._PK<start>_<end>’) : " + ", ".join(unknown[:10]) + ("..." if len(unknown) > 10 else ""))

    st.markdown("---")

    # --------------------------
    # 2) Paramètres d’inférence
    # --------------------------
    st.markdown("### 2) Paramètres d’inférence & de fusion")
    colA, colB, colC = st.columns(3)
    with colA:
        score_thr = st.number_input("Seuil score instance", min_value=0.0, max_value=1.0, value=0.40, step=0.05)
        pixel_thr = st.number_input("Seuil pixel masque",   min_value=0.0, max_value=1.0, value=0.40, step=0.05)
    with colB:
        fuse_dilate_iter = st.number_input("Dilations de fusion", min_value=0, max_value=10, value=2, step=1)
        kx = st.number_input("Noyau fusion (kx)", min_value=1, max_value=99, value=9, step=1)
        ky = st.number_input("Noyau fusion (ky)", min_value=1, max_value=99, value=3, step=1)
    with colC:
        hboost = st.checkbox("Boost horizontal (fermeture)", value=True)
        hline_len = st.number_input("Longueur horizontale", min_value=1, max_value=999, value=30, step=1)
        hline_thick = st.number_input("Épaisseur fermeture", min_value=1, max_value=99, value=3, step=1)
        hclose_iters = st.number_input("Itérations fermeture", min_value=0, max_value=10, value=1, step=1)

    params = {
        "score_thr": float(score_thr),
        "pixel_thr": float(pixel_thr),
        "pre_split_dilate_px": 0,
        "fuse_dilate_iter": int(fuse_dilate_iter),
        "fuse_kernel_size": (int(kx), int(ky)),
        "fuse_horizontal_boost": bool(hboost),
        "hline_len": int(hline_len),
        "hline_thick": int(hline_thick),
        "hline_close_iters": int(hclose_iters),
        "cm_step": 10,
        "expected_points": 40,
    }
    st.caption("Export d’interfaces: pas 10 cm, 40 points par image (≈ 4 m).")

    st.markdown("---")

    # --------------------------
    # 3) Inférence ASC / DESC
    # --------------------------
    st.markdown("### 3) Inférence par catégorie (ASC / DESC)")

    def _maybe_skip_or_force(category: str) -> bool:
        scan = st.session_state.get("_scan")
        if not scan:
            st.error("Commence par ‘Scanner le dossier’.")
            st.stop()
        cat_root = os.path.join(scan["output_root"], category)
        info = has_existing_inference(cat_root)
        if info["exists"]:
            st.warning(f"Des résultats existent déjà pour {category} ({info['num_excels']} excels).")
            return st.checkbox(f"✅ Recalculer {category} (écrase les sorties existantes) ?", key=f"force_{category}")
        return True

    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶️ Inférence — PK croissants (ASC)"):
            scan = st.session_state.get("_scan")
            if not scan:
                st.error("Commence par ‘Scanner le dossier’."); st.stop()
            force_run = _maybe_skip_or_force("ASC")
            t0 = time.time()
            out = run_inference_for_category(
                image_folder=scan["image_folder"],
                files=scan["asc"],
                model_path=scan["model_path"],
                output_root=scan["output_root"],
                category="ASC",
                params=params,
                force=force_run
            )
            if out.get("skipped_due_to_existing"):
                st.info("⏩ Inférence non relancée (déjà existante).")
            else:
                st.success(f"ASC: {out['num_images']} image(s) traitée(s) • {time.time()-t0:.1f}s")
                st.cache_data.clear()
            st.session_state.df_asc_raw = None

    with c2:
        if st.button("▶️ Inférence — PK décroissants (DESC)"):
            scan = st.session_state.get("_scan")
            if not scan:
                st.error("Commence par ‘Scanner le dossier’."); st.stop()
            force_run = _maybe_skip_or_force("DESC")
            t0 = time.time()
            out = run_inference_for_category(
                image_folder=scan["image_folder"],
                files=scan["desc"],
                model_path=scan["model_path"],
                output_root=scan["output_root"],
                category="DESC",
                params=params,
                force=force_run
            )
            if out.get("skipped_due_to_existing"):
                st.info("⏩ Inférence non relancée (déjà existante).")
            else:
                st.success(f"DESC: {out['num_images']} image(s) traitée(s) • {time.time()-t0:.1f}s")
                st.cache_data.clear()
            st.session_state.df_desc_raw = None

    if st.button("🔄 Rafraîchir la stratigraphie (vider le cache)"):
        st.cache_data.clear()
        st.session_state.df_asc_raw = None
        st.session_state.df_desc_raw = None
        st.success("Cache vidé. Les tableaux seront rechargés.")

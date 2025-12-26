import os
import re
import csv
import json
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# ---------------------------------------------------
# Fichiers & parsing
# ---------------------------------------------------
def list_image_files(folder):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted([f for f in os.listdir(folder) if os.path.splitext(f.lower())[1] in exts])

def parse_image_name(fname):
    """
    Exemples acceptés :
      calais_PK285766_285770.png
      calais_region1_PK285770_285766.jpg
      chantier-X_pk285770-285766.tif
    Retourne: (chantier, pk_start_cm, pk_end_cm) ou (None, None, None).
    """
    base = os.path.basename(fname)
    m = re.match(r'^(?P<chantier>.+?)_P[Kk](?P<start>\d+)[_-](?P<end>\d+)(?:\.[^.]+)?$', base)
    if not m:
        return None, None, None
    chantier = m.group("chantier")
    pk_start_cm = int(m.group("start")) * 100
    pk_end_cm   = int(m.group("end"))   * 100
    return chantier, pk_start_cm, pk_end_cm

def classify_files_by_pk(files):
    asc, desc, unknown = [], [], []
    for f in files:
        _, s, e = parse_image_name(f)
        if s is None or e is None or s == e:
            unknown.append(f)
        elif e > s:
            asc.append(f)
        else:
            desc.append(f)
    return asc, desc, unknown

# ---------------------------------------------------
# Morphologie & fusion
# ---------------------------------------------------
def _to_uint8_mask(m):
    return (m > 0).astype(np.uint8) * 255

def fuse_close_masks(masks,
                     dilate_iter=2,
                     kernel_size=(9, 3),
                     horiz_boost=True,
                     hlen=30,
                     hthick=3,
                     hclose_iter=1):
    """Fusionne des morceaux proches (PRÉDICTIONS) avec accent horizontal."""
    if not masks:
        return []
    H, W = masks[0].shape[:2]
    combined = np.zeros((H, W), dtype=np.uint8)
    for m in masks:
        combined = cv2.bitwise_or(combined, _to_uint8_mask(m))

    work = combined.copy()
    if horiz_boost and hlen > 1 and hthick > 0 and hclose_iter > 0:
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(hlen), int(hthick)))
        work = cv2.dilate(work, h_kernel, iterations=int(hclose_iter))
        work = cv2.erode(work,  h_kernel, iterations=int(hclose_iter))

    kx, ky = kernel_size
    iso_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1,int(kx)), max(1,int(ky))))
    if dilate_iter > 0:
        work = cv2.dilate(work, iso_kernel, iterations=int(dilate_iter))

    contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fused = []
    for cnt in contours:
        cnt = np.ascontiguousarray(cnt, dtype=np.int32)
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        fused.append(mask)
    return fused

# ---------------------------------------------------
# Export visu & données
# ---------------------------------------------------
def _ensure_dirs(root):
    dirs = {
        "root": root,
        "overlays": os.path.join(root, "overlays"),
        "pred_masks": os.path.join(root, "pred_masks"),
        "pred_colormap": os.path.join(root, "pred_colormap"),
        "positions": os.path.join(root, "positions"),
        "excels": os.path.join(root, "excels"),
    }
    for d in dirs.values():
        if d: os.makedirs(d, exist_ok=True)
    return dirs

def _save_visuals(image_rgb, fname, dirs, pred_instances):
    H, W = image_rgb.shape[:2]
    overlay = image_rgb.copy()
    for m in pred_instances:
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (255, 0, 0), 2)
    cv2.imwrite(os.path.join(dirs["overlays"], fname), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    pred_combined = np.zeros((H, W), dtype=np.uint8)
    for m in pred_instances:
        pred_combined = cv2.bitwise_or(pred_combined, (m > 0).astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(dirs["pred_masks"], fname), pred_combined)

    colored = cv2.applyColorMap(pred_combined, cv2.COLORMAP_JET)
    blend = cv2.addWeighted(image_rgb, 0.7, colored, 0.3, 0)
    cv2.imwrite(os.path.join(dirs["pred_colormap"], fname), cv2.cvtColor(blend, cv2.COLOR_RGB2BGR))

def _write_positions_json(fname, chantier, pk_start_cm, pk_end_cm, fused_masks, out_dir):
    if len(fused_masks) == 0:
        pixels = []
    else:
        H, W = fused_masks[0].shape[:2]
        combined = np.zeros((H, W), dtype=np.uint8)
        for m in fused_masks:
            combined = cv2.bitwise_or(combined, (m > 0).astype(np.uint8) * 255)
        yx = np.argwhere(combined > 0)  # rows, cols -> (y, x)
        pixels = [[int(x), int(y)] for (y, x) in yx]
    payload = {
        "image": fname,
        "chantier": chantier,
        "pk_start_cm": None if pk_start_cm is None else int(pk_start_cm),
        "pk_end_cm": None if pk_end_cm is None else int(pk_end_cm),
        "pixels": pixels
    }
    out_path = os.path.join(out_dir, os.path.splitext(fname)[0] + ".json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    index_path = os.path.join(out_dir, "_index.json")
    idx = {"files": []}
    if os.path.exists(index_path):
        try:
            with open(index_path, "r") as f:
                idx = json.load(f)
        except Exception:
            idx = {"files": []}
    if "files" not in idx or not isinstance(idx["files"], list):
        idx["files"] = []
    rel = os.path.basename(out_path)
    if rel not in idx["files"]:
        idx["files"].append(rel)
    with open(index_path, "w") as f:
        json.dump(idx, f, indent=2)
    return out_path

def _build_xcm_grid(pk_start_cm, pk_end_cm, step_cm=10, expected_points=40):
    if pk_start_cm is None or pk_end_cm is None or pk_start_cm == pk_end_cm:
        return np.array([], dtype=np.int64)
    sign = 1 if pk_end_cm > pk_start_cm else -1
    step = sign * abs(step_cm)
    x_cm = np.arange(pk_start_cm, pk_end_cm, step, dtype=np.int64)
    if expected_points:
        if x_cm.size > expected_points:
            x_cm = x_cm[:expected_points]
        elif x_cm.size < expected_points:
            x_cm = np.linspace(pk_start_cm, pk_end_cm, expected_points, endpoint=False, dtype=np.int64)
    return x_cm

def _xcm_to_xpx(x_cm_vals, pk_start_cm, pk_end_cm, width_px):
    denom = (pk_end_cm - pk_start_cm)
    if denom == 0:
        denom = 1
    x_rel = (x_cm_vals - pk_start_cm) / denom
    x_px = np.rint(x_rel * (width_px - 1)).astype(int)
    return np.clip(x_px, 0, width_px - 1)

def _y_profile_from_mask(mask_uint8, x_px_grid):
    ys = np.full(len(x_px_grid), np.nan, dtype=float)
    for i, x in enumerate(x_px_grid):
        col = mask_uint8[:, x] > 0
        if np.any(col):
            ys[i] = float(np.mean(np.where(col)[0]))
    return ys

def _order_top_down(fused_masks):
    scored = []
    for m in fused_masks:
        ys, xs = np.where(m > 0)
        y_min = int(np.min(ys)) if ys.size else 10**9
        scored.append((y_min, m))
    scored.sort(key=lambda t: t[0])  # haut -> bas
    return [m for _, m in scored]

def _write_interfaces_excel(fname, chantier, pk_start_cm, pk_end_cm, fused_masks, image_shape, out_dir,
                            cm_step=10, expected_points=40):
    """
    Écrit un Excel par image, feuille 'interfaces' :
      - colonnes: image, height_px, x_cm, interface_0, interface_1, ...
      - y = indices de pixels (pas de conversion ici)
    On ajoute 'image' et 'height_px' pour permettre plus tard la conversion en ns.
    """
    if chantier is None or pk_start_cm is None or pk_end_cm is None or pk_start_cm == pk_end_cm:
        return None
    H, W = image_shape[:2]
    x_cm_grid = _build_xcm_grid(pk_start_cm, pk_end_cm, cm_step, expected_points)
    if x_cm_grid.size == 0:
        return None
    x_px_grid = _xcm_to_xpx(x_cm_grid, pk_start_cm, pk_end_cm, W)
    masks_sorted = _order_top_down(fused_masks)

    data = {
        "image": [fname] * len(x_cm_grid),
        "height_px": [H] * len(x_cm_grid),
        "x_cm": x_cm_grid.astype(int),
    }
    if len(masks_sorted) == 0:
        data["interface_0"] = [np.nan] * len(x_cm_grid)
    else:
        for i, m in enumerate(masks_sorted):
            y_profile = _y_profile_from_mask(m, x_px_grid)
            data[f"interface_{i}"] = y_profile  # NaN si absent

    df = pd.DataFrame(data)
    excel_path = os.path.join(out_dir, os.path.splitext(fname)[0] + ".xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="interfaces", index=False)
    return excel_path

# ---------------------------------------------------
# Vérif d'existence de sorties
# ---------------------------------------------------
def has_existing_inference(cat_root: str) -> dict:
    """
    Vérifie si une inférence existe déjà dans cat_root.
    Retourne dict: {exists: bool, num_excels: int, csv: path|None}
    """
    excels_dir = os.path.join(cat_root, "excels")
    n_xls = 0
    if os.path.isdir(excels_dir):
        n_xls = len([f for f in os.listdir(excels_dir) if f.lower().endswith(".xlsx")])
    csv_path = os.path.join(cat_root, "per_image_predictions.csv")
    exists = (n_xls > 0) and os.path.isfile(csv_path)
    return {"exists": exists, "num_excels": n_xls, "csv": csv_path if os.path.isfile(csv_path) else None}

# ---------------------------------------------------
# Modèle & inférence (lazy-import torch/torchvision)
# ---------------------------------------------------
def _load_model(model_path, num_classes=2, device_str=None):
    import torch
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    device = torch.device(device_str) if device_str else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, device

def run_inference_for_category(image_folder, files, model_path, output_root, category, params, force: bool = False):
    """
    Traite la liste d'images ‘files’ dans image_folder,
    écrit les sorties dans {output_root}/{category}/...
    Si des résultats existent déjà et que force=False -> SKIP (pas de recalcul).
    """
    cat_root = os.path.join(output_root, category)
    dirs = _ensure_dirs(cat_root)

    # Vérif d'existence
    info = has_existing_inference(cat_root)
    if info["exists"] and not force:
        # On ne relance pas, on renvoie juste un résumé
        return {
            "num_images": info["num_excels"],
            "sample_paths": [],
            "csv": info["csv"],
            "out_dir": cat_root,
            "skipped_due_to_existing": True
        }

    # log paramètres
    with open(os.path.join(cat_root, "run_meta.json"), "w") as f:
        json.dump({"category": category, "params": params}, f, indent=2)

    # lazy import torch
    import torch

    model, device = _load_model(model_path)
    csv_path = os.path.join(cat_root, "per_image_predictions.csv")
    samples = []

    with open(csv_path, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["image", "num_pred_instances", "scores_mean", "scores_max", "num_interfaces_excel"])

        for fname in files or []:
            path = os.path.join(image_folder, fname)
            try:
                rgb = np.array(Image.open(path).convert("RGB"))
            except Exception:
                # image illisible
                continue

            tens = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            tens = tens.unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(tens)[0]

            chosen, score_list = [], []
            scores = out.get("scores")
            masks  = out.get("masks")
            if scores is not None and masks is not None:
                scores = scores.detach().cpu().numpy()
                probs  = masks.detach().cpu().numpy()[:, 0, :, :]
                for s, prob in zip(scores, probs):
                    if s >= params["score_thr"]:
                        binm = (prob > params["pixel_thr"]).astype(np.uint8) * 255
                        # Split composantes connexes
                        cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for c in cnts:
                            m = np.zeros_like(binm, dtype=np.uint8)
                            cv2.drawContours(m, [c], -1, 255, -1)
                            chosen.append(m)
                            score_list.append(float(s))

            preds = fuse_close_masks(
                chosen,
                dilate_iter=params["fuse_dilate_iter"],
                kernel_size=params["fuse_kernel_size"],
                horiz_boost=params["fuse_horizontal_boost"],
                hlen=params["hline_len"],
                hthick=params["hline_thick"],
                hclose_iter=params["hline_close_iters"]
            )

            _save_visuals(rgb, fname, dirs, preds)
            chantier, pk_start_cm, pk_end_cm = parse_image_name(fname)

            _write_positions_json(
                fname=fname,
                chantier=chantier,
                pk_start_cm=pk_start_cm,
                pk_end_cm=pk_end_cm,
                fused_masks=preds,
                out_dir=dirs["positions"]
            )

            excel_path = _write_interfaces_excel(
                fname=fname,
                chantier=chantier,
                pk_start_cm=pk_start_cm,
                pk_end_cm=pk_end_cm,
                fused_masks=preds,
                image_shape=rgb.shape,
                out_dir=dirs["excels"],
                cm_step=params["cm_step"],
                expected_points=params["expected_points"]
            )

            num_pred = len(preds)
            scores_mean = float(np.mean(score_list)) if score_list else 0.0
            scores_max  = float(np.max(score_list))  if score_list else 0.0
            num_interfaces_excel = len(preds) if excel_path else 0
            w.writerow([fname, num_pred, f"{scores_mean:.6f}", f"{scores_max:.6f}", num_interfaces_excel])

            if len(samples) < 5 and excel_path:
                samples.append({
                    "overlay": os.path.join(dirs["overlays"], fname),
                    "mask": os.path.join(dirs["pred_masks"], fname),
                    "excel": excel_path
                })

    # Récap
    excels_dir = os.path.join(cat_root, "excels")
    n_xls = len([f for f in os.listdir(excels_dir) if f.lower().endswith(".xlsx")]) if os.path.isdir(excels_dir) else 0
    return {"num_images": n_xls, "sample_paths": samples, "csv": csv_path, "out_dir": cat_root, "skipped_due_to_existing": False}

# ---------------------------------------------------
# Agrégation (stratigraphie)
# ---------------------------------------------------
def aggregate_excels_to_stratigraphy(excels_dir, sort_order="ASC"):
    """
    Concatène tous les Excels (x_cm + interface_i) d’un dossier et
    renvoie un DataFrame unique trié (ASC/DESC) par x_cm.
    Conserve aussi les colonnes 'image' et 'height_px' utiles au mapping px→ns.
    """
    if not os.path.isdir(excels_dir):
        return None
    excels = [os.path.join(excels_dir, f) for f in os.listdir(excels_dir) if f.lower().endswith(".xlsx")]
    if not excels:
        return None

    dfs = []
    for xls in sorted(excels):
        try:
            df = pd.read_excel(xls, sheet_name="interfaces", engine="openpyxl")
            dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return None

    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df.dropna(subset=["x_cm"], how="any")
    all_df["x_cm"] = all_df["x_cm"].astype(int)

    if sort_order.upper() == "ASC":
        all_df = all_df.sort_values("x_cm", ascending=True, kind="mergesort")
    else:
        all_df = all_df.sort_values("x_cm", ascending=False, kind="mergesort")

    all_df = all_df.reset_index(drop=True)
    return all_df

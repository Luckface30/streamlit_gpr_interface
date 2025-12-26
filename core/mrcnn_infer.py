# core/mrcnn_infer.py
"""
Module d'inférence Mask R-CNN pour la détection d'interfaces GPR.

- Scan de dossier (list_image_files) + classification par PK (..._PKstart_end)
- Charge deux modèles (ASC/DESC) et fusionne leurs masques
- Filtrage géométrique (largeur min)
- RÈGLES MÉTIER: zone interface_0/forbidden en RATIOS de la hauteur (0.15*H / 0.22*H)
- Profils y(x) échantillonnés sur EXPECTED_POINTS abscisses x_cm
- Exports : overlays PNG, masques PNG, colormap PNG, Excel (interfaces + meta), JSON positions
- Agrégation multi-images en stratigraphie triée par x_cm

Expose pour l’UI :
- DEFAULT_PARAMS (toutes les clés lues par la page Streamlit)
- list_image_files, classify_files_by_pk, has_existing_inference
- run_inference_for_category, aggregate_excels_to_stratigraphy, check_excels_exist
"""

import os
import re
import json
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# ===================== Chemins modèles par défaut ===================== #
MODEL_PATH1 = "models/mask_rcnn_trained_3.pth"
MODEL_PATH2 = "models/mask_rcnn_trained2.pth"

# ===================== Seuils / filtres ===================== #
THR_SCORE = 0.40
BINARY_THR = 0.15
MIN_WIDTH_PX = 100
ASPECT_RATIO_THR = 50.0
MIN_HEIGHT_PX = 2

# ===================== Bornes dynamiques (RATIOS de H) ===================== #
RATIO_INTERFACE_MIN = 0.0
RATIO_FORBIDDEN_MAX = 0.08

# ===================== Échantillonnage / pas ===================== #
EXPECTED_POINTS = 50
PX_STEP = 10
CM_STEP = 10

# ===================== Morphologie / fusion ===================== #
CLOSE_KSIZE = 3
THICKEN_H = 0
FUSE_HORIZONTAL_BOOST = True
FUSE_DILATE_ITER = 5
FUSE_KERNEL_SIZE = (9, 3)
PRE_SPLIT_DILATE_PX = 0

# ===================== Dessin / UI ===================== #
HLINE_LEN = 30
HLINE_THICK = 3
HLINE_CLOSE_ITERS = 1

# ===================== Dossiers de sortie ===================== #
OVERLAYS_DIRNAME  = "overlays"
MASKS_DIRNAME     = "pred_masks"
COLOR_DIRNAME     = "pred_colormap"
POSITIONS_DIRNAME = "positions"
EXCELS_DIRNAME    = "excels"

# ===================== Paramètres exposés à l’UI ===================== #
DEFAULT_PARAMS: Dict[str, Any] = {
    "model_path1": MODEL_PATH1,
    "model_path2": MODEL_PATH2,

    "score_thr": THR_SCORE,
    "binary_thr": BINARY_THR,
    "pixel_thr": BINARY_THR,
    "min_width": MIN_WIDTH_PX,
    "min_width_px": MIN_WIDTH_PX,
    "aspect_ratio_thr": ASPECT_RATIO_THR,
    "min_height_px": MIN_HEIGHT_PX,

    "ratio_interface_min": RATIO_INTERFACE_MIN,
    "ratio_forbidden_max": RATIO_FORBIDDEN_MAX,

    "expected_points": EXPECTED_POINTS,
    "px_step": PX_STEP,
    "cm_step": CM_STEP,

    "close_ksize": CLOSE_KSIZE,
    "thicken_h": THICKEN_H,
    "fuse_horizontal_boost": FUSE_HORIZONTAL_BOOST,
    "fuse_dilate_iter": FUSE_DILATE_ITER,
    "fuse_kernel_size": FUSE_KERNEL_SIZE,
    "pre_split_dilate_px": PRE_SPLIT_DILATE_PX,

    "hline_len": HLINE_LEN,
    "hline_thick": HLINE_THICK,
    "hline_close_iters": HLINE_CLOSE_ITERS,

    "overlays_dirname": OVERLAYS_DIRNAME,
    "masks_dirname": MASKS_DIRNAME,
    "color_dirname": COLOR_DIRNAME,
    "positions_dirname": POSITIONS_DIRNAME,
    "excels_dirname": EXCELS_DIRNAME,
}

# ===================== Utils fichiers/IO ===================== #
def list_image_files(image_folder: str, exts=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")) -> List[str]:
    if not image_folder or not os.path.isdir(image_folder):
        return []
    files = [f for f in os.listdir(image_folder) if f.lower().endswith(tuple(e.lower() for e in exts))]
    files.sort()
    return files

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

_PK_RE = re.compile(r"_PK(\d+)[_-](\d+)", re.IGNORECASE)

def _regex_pk(fname: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Extrait les bornes PK (en centimètres réels) à partir du nom du fichier.
    Exemple : '_PK280318-280378' → (28031800, 28037800)
    """
    m = _PK_RE.search(fname)
    if not m:
        return None, None
    try:
        start_cm = int(m.group(1)) * 100
        end_cm   = int(m.group(2)) * 100
        return start_cm, end_cm
    except Exception:
        return None, None

def classify_files_by_pk(files: List[str]) -> Tuple[List[str], List[str], Dict[str, Dict[str, int]]]:
    with_pk = []
    mapping: Dict[str, Dict[str, int]] = {}
    for f in files:
        s, e = _regex_pk(f)
        if s is not None and e is not None:
            with_pk.append((f, s, e))
            mapping[f] = {"pk_start_cm": s, "pk_end_cm": e}
    with_pk.sort(key=lambda t: t[1])
    files_with_pk_sorted = [t[0] for t in with_pk]
    files_without_pk = [f for f in files if f not in set(files_with_pk_sorted)]
    return files_with_pk_sorted, files_without_pk, mapping

def has_existing_inference(output_root: str, category: str) -> Dict[str, object]:
    base = os.path.join(output_root, category)
    exists_any = False
    for sub in [OVERLAYS_DIRNAME, MASKS_DIRNAME, COLOR_DIRNAME, EXCELS_DIRNAME]:
        d = os.path.join(base, sub)
        if os.path.isdir(d) and any(os.scandir(d)):
            exists_any = True

    num_excels = 0
    dir_excels = os.path.join(base, DEFAULT_PARAMS.get("excels_dirname", EXCELS_DIRNAME))
    if os.path.isdir(dir_excels):
        num_excels = len([f for f in os.listdir(dir_excels) if f.lower().endswith(".xlsx")])

    return {"exists": bool(exists_any), "num_excels": int(num_excels)}

# ===================== Modèle & prédiction ===================== #
def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _get_model_instance_segmentation(num_classes: int = 2) -> nn.Module:
    model = maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model

def _binary_mask_from_logits(logits: np.ndarray, thr: float) -> np.ndarray:
    return (logits >= thr).astype(np.uint8) * 255

def _pre_split_dilate(mask_uint8: np.ndarray, px: int) -> np.ndarray:
    if not px or int(px) <= 0:
        return mask_uint8
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(px), 1))
    return cv2.dilate(mask_uint8, k, iterations=1)

def _predict_single_model(model: nn.Module, device: torch.device, img_bgr: np.ndarray) -> List[np.ndarray]:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    x = x.to(device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        out = model(x)[0]
    masks = out.get("masks")
    scores = out.get("scores", None)
    out_masks: List[np.ndarray] = []
    if masks is not None and len(masks) > 0:
        m = masks[:, 0, :, :].detach().cpu().numpy()
        s = scores.detach().cpu().numpy().tolist() if scores is not None else [1.0] * m.shape[0]
        thr_score = float(DEFAULT_PARAMS.get("score_thr", THR_SCORE))
        pix_thr   = float(DEFAULT_PARAMS.get("pixel_thr", BINARY_THR))
        prepx     = int(DEFAULT_PARAMS.get("pre_split_dilate_px", PRE_SPLIT_DILATE_PX))
        for mi, si in zip(m, s):
            if si < thr_score:
                continue
            mask_bin = _binary_mask_from_logits(mi, pix_thr)
            mask_bin = _pre_split_dilate(mask_bin, prepx)
            out_masks.append(mask_bin)
    return out_masks

# ===================== Masques : overlay / tri / fusion ===================== #
def _overlay_mask(img: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    col = np.zeros_like(img)
    col[:, :, 1] = 255
    return cv2.addWeighted(img, 1.0, cv2.bitwise_and(col, col, mask=mask), alpha, 0)

def _order_interfaces_top_down(masks: List[np.ndarray]) -> List[np.ndarray]:
    if not masks:
        return []
    meds = []
    for m in masks:
        ys = np.where(m > 0)[0]
        meds.append(float(np.median(ys)) if ys.size else float("inf"))
    order = np.argsort(meds)
    return [masks[i] for i in order]

def _fuse_two_models_masks(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    m = np.logical_or(mask1 > 0, mask2 > 0).astype(np.uint8) * 255
    kx, ky = DEFAULT_PARAMS.get("fuse_kernel_size", FUSE_KERNEL_SIZE)
    if not kx: kx = DEFAULT_PARAMS.get("close_ksize", CLOSE_KSIZE)
    if not ky: ky = DEFAULT_PARAMS.get("close_ksize", CLOSE_KSIZE)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(kx), int(ky)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    if DEFAULT_PARAMS.get("fuse_horizontal_boost", FUSE_HORIZONTAL_BOOST):
        th = int(DEFAULT_PARAMS.get("thicken_h", THICKEN_H))
        it = int(DEFAULT_PARAMS.get("fuse_dilate_iter", FUSE_DILATE_ITER))
        if th > 0 and it > 0:
            k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (th, 1))
            m = cv2.dilate(m, k2, iterations=it)
    return m

def _morph_close_then_dilate(mask_uint8: np.ndarray) -> np.ndarray:
    kx, ky = DEFAULT_PARAMS.get("fuse_kernel_size", FUSE_KERNEL_SIZE)
    if not kx: kx = DEFAULT_PARAMS.get("close_ksize", CLOSE_KSIZE)
    if not ky: ky = DEFAULT_PARAMS.get("close_ksize", CLOSE_KSIZE)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(kx), int(ky)))
    out = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, k, iterations=1)
    if DEFAULT_PARAMS.get("fuse_horizontal_boost", FUSE_HORIZONTAL_BOOST):
        th = int(DEFAULT_PARAMS.get("thicken_h", THICKEN_H))
        it = int(DEFAULT_PARAMS.get("fuse_dilate_iter", FUSE_DILATE_ITER))
        if th > 0 and it > 0:
            k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (th, 1))
            out = cv2.dilate(out, k2, iterations=it)
    return out

def _fuse_close_masks_global(masks: List[np.ndarray], shape_hw: Tuple[int, int]) -> np.ndarray:
    H, W = shape_hw
    if not masks:
        return np.zeros((H, W), np.uint8)
    union = np.zeros((H, W), np.uint8)
    for m in masks:
        if m is None:
            continue
        mm = m
        if mm.shape[:2] != (H, W):
            mm = cv2.resize(mm, (W, H), interpolation=cv2.INTER_NEAREST)
        union = cv2.bitwise_or(union, (mm > 0).astype(np.uint8) * 255)
    smooth = _morph_close_then_dilate(union)
    contours, _ = cv2.findContours((smooth > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros((H, W), np.uint8)
    if contours:
        cv2.drawContours(clean, contours, -1, 255, thickness=cv2.FILLED)
    return clean

def _split_components(mask_uint8: np.ndarray) -> List[np.ndarray]:
    comp: List[np.ndarray] = []
    contours, _ = cv2.findContours((mask_uint8 > 0).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        m = np.zeros_like(mask_uint8)
        cv2.drawContours(m, [c], -1, 255, thickness=cv2.FILLED)
        comp.append(m)
    return _order_interfaces_top_down(comp)

def _filter_masks_geometry(masks: List[np.ndarray]) -> List[np.ndarray]:
    min_w = int(DEFAULT_PARAMS.get("min_width", MIN_WIDTH_PX))
    out = []
    for m in masks:
        if m is None:
            continue
        ys, xs = np.where(m > 0)
        if xs.size == 0:
            continue
        width = int(xs.max() - xs.min())
        if width < min_w:
            continue
        out.append(m)
    return out

def _apply_interface_thresholds(masks: List[np.ndarray]) -> List[np.ndarray]:
    """
    Supprime ENTIEREMENT les objets dont y_min tombe dans la zone interdite (0.15H, 0.22H).
    Conserve ceux avec y_min <= 0.15H (candidats interface_0) et y_min >= 0.22H, puis tri top->down.
    """
    if not masks:
        return []
    H = masks[0].shape[0]
    thr_min = float(DEFAULT_PARAMS.get("ratio_interface_min", RATIO_INTERFACE_MIN)) * float(H)
    thr_max = float(DEFAULT_PARAMS.get("ratio_forbidden_max", RATIO_FORBIDDEN_MAX)) * float(H)
    filtered = []
    for m in masks:
        ys = np.where(m > 0)[0]
        if ys.size == 0:
            continue
        y_min = float(np.min(ys))
        if thr_min < y_min < thr_max:
            continue
        filtered.append(m)
    return _order_interfaces_top_down(filtered)

# ===================== Profils ===================== #
def _y_profile_from_mask(mask_uint8: np.ndarray, x_px_grid: np.ndarray) -> np.ndarray:
    """Profil y(x) : FRONTIÈRE SUPÉRIEURE (min y) des pixels non-nuls par colonne x."""
    y = np.full_like(x_px_grid, fill_value=np.nan, dtype=float)
    for i, x in enumerate(x_px_grid):
        col = mask_uint8[:, int(x)]
        ys = np.where(col > 0)[0]
        y[i] = float(np.min(ys)) if ys.size else np.nan
    return y

# ===================== Exports ===================== #
def _write_pred_images(out_dir_overlays: str, out_dir_pred_masks: str, out_dir_pred_color: str,
                       fname: str, img: np.ndarray, combined_mask: np.ndarray) -> Tuple[str, str, str]:
    _ensure_dir(out_dir_overlays)
    _ensure_dir(out_dir_pred_masks)
    _ensure_dir(out_dir_pred_color)

    H, W = img.shape[:2]
    m = combined_mask
    if m.shape[:2] != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

    # overlay contours
    overlay = img.copy()
    contours, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), thickness=2)

    p_overlay = os.path.join(out_dir_overlays, fname)
    cv2.imwrite(p_overlay, overlay)

    # masque combiné unique
    p_mask = os.path.join(out_dir_pred_masks, f"{os.path.splitext(fname)[0]}_mask.png")
    cv2.imwrite(p_mask, m)

    # colormap JET
    m_norm = (m > 0).astype(np.uint8) * 255
    cm = cv2.applyColorMap(m_norm, cv2.COLORMAP_JET)
    color_blend = cv2.addWeighted(img, 0.7, cm, 0.3, 0.0)
    p_color = os.path.join(out_dir_pred_color, fname)
    cv2.imwrite(p_color, color_blend)

    return p_overlay, p_mask, p_color

def _save_json_positions(out_dir_positions: str, fname: str, x_list: List[int], interfaces: Dict[str, List[float]]):
    payload = {"file": fname, "x_cm": x_list, "interfaces": interfaces}
    _ensure_dir(out_dir_positions)
    out_path = os.path.join(out_dir_positions, os.path.splitext(fname)[0] + ".json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path

def _write_positions_json(out_dir_positions: str, fname: str,
                          x_cm_list: List[int], interfaces_df: pd.DataFrame) -> str:
    interfaces = {c: interfaces_df[c].astype(float).where(pd.notna(interfaces_df[c]), np.nan).tolist()
                  for c in interfaces_df.columns if str(c).startswith("interface_")}
    return _save_json_positions(out_dir_positions, fname, list(map(int, x_cm_list)), interfaces)

def _write_interfaces_excel(out_dir_excels: str, fname: str, chantier: Optional[str],
                            pk_start_cm: Optional[int], pk_end_cm: Optional[int],
                            fused_masks: List[np.ndarray], image_shape: Tuple[int, int, int]) -> Optional[str]:
    if pk_start_cm is None or pk_end_cm is None or pk_start_cm == pk_end_cm:
        return None

    H, W = image_shape[:2]
    x_cm_grid = np.linspace(pk_start_cm, pk_end_cm, int(DEFAULT_PARAMS.get("expected_points", EXPECTED_POINTS)),
                            endpoint=False, dtype=np.int64)
    if x_cm_grid.size == 0:
        return None
    x_px_grid = np.rint((x_cm_grid - pk_start_cm) / (pk_end_cm - pk_start_cm) * (W - 1)).astype(int)
    x_px_grid = np.clip(x_px_grid, 0, W - 1)

    masks_sorted = _order_interfaces_top_down(fused_masks)
    data: Dict[str, Any] = {
        "image_name": [os.path.basename(fname)] * len(x_cm_grid),
        "img_xmin_cm": [int(pk_start_cm)] * len(x_cm_grid),
        "img_xmax_cm": [int(pk_end_cm)] * len(x_cm_grid),
        "x_cm": x_cm_grid.astype(int)
    }

    if len(masks_sorted) == 0:
        data["interface_0"] = [np.nan] * len(x_cm_grid)
    else:
        profiles: List[np.ndarray] = []
        medians: List[float] = []
        for m in masks_sorted:
            prof = _y_profile_from_mask(m, x_px_grid)
            profiles.append(prof)
            med = np.nanmedian(prof)
            medians.append(float(med) if np.isfinite(med) else np.nan)

        thr_min = float(DEFAULT_PARAMS.get("ratio_interface_min", RATIO_INTERFACE_MIN)) * float(H)
        thr_max = float(DEFAULT_PARAMS.get("ratio_forbidden_max", RATIO_FORBIDDEN_MAX)) * float(H)
        group0_idx    = [i for i, med in enumerate(medians) if np.isfinite(med) and med <= thr_min]
        forbidden_idx = [i for i, med in enumerate(medians) if np.isfinite(med) and thr_min < med < thr_max]
        remaining_idx = [i for i in range(len(profiles)) if i not in group0_idx and i not in forbidden_idx]

        start_num = 1
        if group0_idx:
            stack = np.vstack([profiles[i] for i in group0_idx])
            data["interface_0"] = np.nanmean(stack, axis=0)

        for out_i, idx in enumerate(remaining_idx, start=start_num):
            data[f"interface_{out_i}"] = profiles[idx]

    df = pd.DataFrame(data)
    _ensure_dir(out_dir_excels)
    excel_path = os.path.join(out_dir_excels, os.path.splitext(fname)[0] + ".xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="interfaces", index=False)
        meta = {
            "file": os.path.basename(fname),
            "pk_start_cm": int(pk_start_cm),
            "pk_end_cm": int(pk_end_cm),
            "height_px": int(H),
            "width_px": int(W),
            "ratio_interface_min": float(DEFAULT_PARAMS.get("ratio_interface_min", RATIO_INTERFACE_MIN)),
            "ratio_forbidden_max": float(DEFAULT_PARAMS.get("ratio_forbidden_max", RATIO_FORBIDDEN_MAX)),
            "expected_points": int(DEFAULT_PARAMS.get("expected_points", EXPECTED_POINTS)),
            "score_thr": float(DEFAULT_PARAMS.get("score_thr", THR_SCORE)),
            "min_width": int(DEFAULT_PARAMS.get("min_width", MIN_WIDTH_PX)),
            "binary_thr": float(DEFAULT_PARAMS.get("binary_thr", BINARY_THR)),
            "px_step": int(DEFAULT_PARAMS.get("px_step", PX_STEP)),
            "cm_step": int(DEFAULT_PARAMS.get("cm_step", CM_STEP)),
            "close_ksize": int(DEFAULT_PARAMS.get("close_ksize", CLOSE_KSIZE)),
            "thicken_h": int(DEFAULT_PARAMS.get("thicken_h", THICKEN_H)),
            "fuse_horizontal_boost": bool(DEFAULT_PARAMS.get("fuse_horizontal_boost", FUSE_HORIZONTAL_BOOST)),
            "fuse_dilate_iter": int(DEFAULT_PARAMS.get("fuse_dilate_iter", FUSE_DILATE_ITER)),
            "fuse_kernel_size": list(DEFAULT_PARAMS.get("fuse_kernel_size", FUSE_KERNEL_SIZE)),
            "pre_split_dilate_px": int(DEFAULT_PARAMS.get("pre_split_dilate_px", PRE_SPLIT_DILATE_PX)),
            "hline_len": int(DEFAULT_PARAMS.get("hline_len", HLINE_LEN)),
            "hline_thick": int(DEFAULT_PARAMS.get("hline_thick", HLINE_THICK)),
            "hline_close_iters": int(DEFAULT_PARAMS.get("hline_close_iters", HLINE_CLOSE_ITERS)),
        }
        pd.DataFrame([meta]).to_excel(writer, sheet_name="meta", index=False)

    return excel_path

# ===================== Agrégation ===================== #
def aggregate_excels_to_stratigraphy(excels_dir: str, sort_order: str = "ASC") -> Optional[pd.DataFrame]:
    """Concatène toutes les feuilles 'interfaces' d'un dossier en un seul DataFrame trié par x_cm (ASC/DESC respecté)."""
    if not os.path.isdir(excels_dir):
        return None
    frames: List[pd.DataFrame] = []
    for f in os.listdir(excels_dir):
        if f.lower().endswith(".xlsx"):
            p = os.path.join(excels_dir, f)
            try:
                df = pd.read_excel(p, sheet_name="interfaces")
                if "x_cm" in df.columns:
                    df["x_cm"] = pd.to_numeric(df["x_cm"], errors="coerce")
                keep = ["x_cm"] + [c for c in df.columns if str(c).startswith("interface_")]
                for extra in ["image_name", "img_xmin_cm", "img_xmax_cm"]:
                    if extra in df.columns:
                        keep.insert(0, extra)
                frames.append(df[keep])
            except Exception:
                continue
    if not frames:
        return None
    out = pd.concat(frames, axis=0, ignore_index=True)
    if "x_cm" in out.columns:
        out = out.dropna(subset=["x_cm"])
        out["x_cm"] = pd.to_numeric(out["x_cm"], errors="coerce")
        asc = (str(sort_order).upper() == "ASC")
        out = out.sort_values("x_cm", ascending=asc).reset_index(drop=True)
    return out

# ===================== Inférence principale ===================== #
def run_inference_for_category(
    image_folder: str,
    files: List[str],
    model_path: str,
    output_root: str,
    category: str,
    chantier: Optional[str] = None,
    force: bool = True
) -> Dict[str, object]:
    info = has_existing_inference(output_root, category)
    if not force and info.get("exists"):
        return {
            "ok": True,
            "output_root": output_root,
            "category": category,
            "skipped_due_to_existing": True,
            "num_images": 0,
            "existing_num_excels": info.get("num_excels", 0),
        }

    device = _get_device()

    cat_root = os.path.join(output_root, category)
    dir_overlays   = os.path.join(cat_root, DEFAULT_PARAMS.get("overlays_dirname", OVERLAYS_DIRNAME))
    dir_pred_masks = os.path.join(cat_root, DEFAULT_PARAMS.get("masks_dirname", MASKS_DIRNAME))
    dir_pred_color = os.path.join(cat_root, DEFAULT_PARAMS.get("color_dirname", COLOR_DIRNAME))
    dir_positions  = os.path.join(cat_root, DEFAULT_PARAMS.get("positions_dirname", POSITIONS_DIRNAME))
    dir_excels     = os.path.join(cat_root, DEFAULT_PARAMS.get("excels_dirname", EXCELS_DIRNAME))
    for d in [dir_overlays, dir_pred_masks, dir_pred_color, dir_positions, dir_excels]:
        _ensure_dir(d)

    model1 = _get_model_instance_segmentation(2).to(device)
    model2 = _get_model_instance_segmentation(2).to(device)
    mp1 = DEFAULT_PARAMS.get("model_path1", MODEL_PATH1)
    mp2 = DEFAULT_PARAMS.get("model_path2", MODEL_PATH2)
    if os.path.isfile(mp1):
        model1.load_state_dict(torch.load(mp1, map_location=device))
    if os.path.isfile(mp2):
        model2.load_state_dict(torch.load(mp2, map_location=device))
    model1.eval(); model2.eval()

    for fname in files:
        img_path = os.path.join(image_folder, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]

        masks1 = _predict_single_model(model1, device, img)
        masks2 = _predict_single_model(model2, device, img)

        all_masks = (masks1 or []) + (masks2 or [])
        combined_mask = _fuse_close_masks_global(all_masks, (H, W))
        if np.count_nonzero(combined_mask) == 0:
            continue

        components_all = _split_components(combined_mask)
        components_all = _filter_masks_geometry(components_all)
        components_kept = _apply_interface_thresholds(components_all)

        visual_mask = np.zeros((H, W), np.uint8)
        for m in components_kept:
            visual_mask = cv2.bitwise_or(visual_mask, (m > 0).astype(np.uint8) * 255)

        _write_pred_images(dir_overlays, dir_pred_masks, dir_pred_color, fname, img, visual_mask)

        pk_start_cm, pk_end_cm = _regex_pk(fname)
        if pk_start_cm is not None and pk_end_cm is not None:
            excel_path = _write_interfaces_excel(
                out_dir_excels=dir_excels,
                fname=fname,
                chantier=chantier,
                pk_start_cm=pk_start_cm,
                pk_end_cm=pk_end_cm,
                fused_masks=components_kept,
                image_shape=img.shape,
            )
            if excel_path:
                df = pd.read_excel(excel_path, sheet_name="interfaces")
                _write_positions_json(dir_positions, fname, df["x_cm"].tolist(), df)

    return {
        "ok": True,
        "output_root": output_root,
        "category": category,
        "num_images": len(files)
    }

def check_excels_exist(output_root: str, category: str) -> Dict[str, object]:
    dir_excels = os.path.join(output_root, category, DEFAULT_PARAMS.get("excels_dirname", EXCELS_DIRNAME))
    exists = os.path.isdir(dir_excels)
    num_excels = 0
    if exists:
        num_excels = len([f for f in os.listdir(dir_excels) if f.lower().endswith(".xlsx")])
    return {"exists": bool(exists and num_excels > 0), "num_excels": num_excels}

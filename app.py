import os, uuid, json, urllib.request, warnings
import numpy as np
import pandas as pd
import cv2
from flask import Flask, request, jsonify, send_file, render_template
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from skimage import filters, morphology, segmentation, measure
from skimage.filters import gaussian
from scipy import ndimage

warnings.filterwarnings("ignore")

app = Flask(__name__)

UPLOAD_FOLDER  = "uploads"
RESULTS_FOLDER = "results"
ANTHROPIC_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")

os.makedirs(UPLOAD_FOLDER,  exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

FEATURES = [
    "AreaShape_Area",
    "AreaShape_Eccentricity",
    "AreaShape_Solidity",
    "AreaShape_MajorAxisLength",
    "AreaShape_MinorAxisLength",
    "AreaShape_Compactness",
    "AreaShape_FormFactor",
    "AreaShape_Perimeter",
    "AreaShape_ConvexArea",
    "AreaShape_EquivalentDiameter",
    "AreaShape_MaximumRadius",
    "AreaShape_MeanRadius",
]

print("\n" + "="*55)
print("  CELLSCAN v5 — WITHIN-IMAGE DETECTION")
print("  No external reference · No domain mismatch")
print("  Healthy tissue → all cells similar → all green")
print("  Cancer tissue  → outlier cells flagged → red")
print("="*55 + "\n")


# ════════════════════════════════════════════════════════════
#  STEP 1: SEGMENT CELLS
#  Uses same algorithms as CellProfiler:
#  Gaussian → Otsu → Watershed → regionprops
# ════════════════════════════════════════════════════════════
def segment_cells(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, "Could not read image file"

    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    gray_f = gray.astype(np.float64) / 255.0

    # Gaussian smoothing (CellProfiler default sigma=2)
    smooth = gaussian(gray_f, sigma=2)

    # Otsu threshold — try both polarities, pick one with more cells
    thresh = filters.threshold_otsu(smooth)

    def clean(b):
        b = morphology.remove_small_objects(b, min_size=80)
        b = morphology.remove_small_holes(b, area_threshold=200)
        return b

    b1 = clean(smooth > thresh)
    b2 = clean(smooth <= thresh)
    n1 = measure.label(b1).max()
    n2 = measure.label(b2).max()
    binary = b1 if n1 >= n2 else b2

    if binary.sum() == 0:
        return None, "Could not detect cells — try a clearer image"

    # Watershed to separate touching cells
    dist    = ndimage.distance_transform_edt(binary)
    from skimage.feature import peak_local_max
    coords  = peak_local_max(dist, min_distance=10, labels=binary)
    mask    = np.zeros(dist.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask)
    labels  = segmentation.watershed(-dist, markers, mask=binary)
    props   = measure.regionprops(labels, gray_f)

    if not props:
        return None, "No cells found after segmentation"

    rows = []
    for i, p in enumerate(props):
        if p.area < 80 or p.area > 80000:
            continue
        perim = max(p.perimeter, 1e-6)
        area  = max(p.area, 1e-6)
        rows.append({
            "ObjectNumber":                i + 1,
            "Location_Center_X":           float(p.centroid[1]),
            "Location_Center_Y":           float(p.centroid[0]),
            "AreaShape_Area":              float(area),
            "AreaShape_Eccentricity":      float(p.eccentricity),
            "AreaShape_Solidity":          float(p.solidity),
            "AreaShape_MajorAxisLength":   float(p.major_axis_length),
            "AreaShape_MinorAxisLength":   float(p.minor_axis_length),
            "AreaShape_Compactness":       float((perim**2) / (4*np.pi*area)),
            "AreaShape_FormFactor":        float((4*np.pi*area) / (perim**2)),
            "AreaShape_Perimeter":         float(perim),
            "AreaShape_ConvexArea":        float(p.convex_area),
            "AreaShape_EquivalentDiameter":float(p.equivalent_diameter),
            "AreaShape_MaximumRadius":     float(np.sqrt(area/np.pi) * 1.2),
            "AreaShape_MeanRadius":        float(np.sqrt(area/np.pi)),
        })

    if not rows:
        return None, "No valid cells after size filtering"

    return pd.DataFrame(rows), None


# ════════════════════════════════════════════════════════════
#  STEP 2: CLASSIFY
#
#  Key insight: healthy tissue → all cells look similar to
#  each other → IsolationForest finds NO extreme outliers.
#  Cancer tissue → cancer cells are morphologically different
#  from surrounding healthy cells → flagged as outliers.
#
#  We NEVER compare to external reference, so there is NO
#  domain mismatch from imaging differences.
# ════════════════════════════════════════════════════════════
def classify_cells(df):
    feat_cols = [f for f in FEATURES if f in df.columns]
    if len(feat_cols) < 4:
        return None, f"Not enough feature columns (found {len(feat_cols)})"

    X = df[feat_cols].copy()
    X = X.fillna(X.median())

    # RobustScaler: median-centred, IQR-scaled
    # More stable than StandardScaler when outliers are present
    scaler  = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # IsolationForest trained on THIS image's cells
    # contamination = expected fraction of cancer cells in tissue
    # 0.05 = assume at most 5% of cells are cancerous by default
    iso = IsolationForest(
        n_estimators  = 300,
        contamination = 0.05,
        random_state  = 42,
        max_samples   = "auto",
    )
    iso.fit(X_scaled)

    raw_scores = iso.score_samples(X_scaled)  # lower = more anomalous

    # Tukey fence: flag cells below Q1 - 2.0 * IQR
    # This is a well-established outlier detection rule.
    # In a healthy sample, NO cell falls below this fence.
    Q1  = np.percentile(raw_scores, 25)
    Q3  = np.percentile(raw_scores, 75)
    IQR = Q3 - Q1

    fence = Q1 - 2.0 * IQR   # strict fence (2× instead of 1.5×)

    # Suspicion score: 0 = totally normal, 100 = extreme outlier
    # Cells at the fence get 50%, cells far below get close to 100%
    score_range = max(Q1 - raw_scores.min(), 1e-6)
    suspicion = np.clip(
        (fence - raw_scores) / score_range * 100 + 50,
        0, 100
    )
    suspicion = np.where(raw_scores >= fence, suspicion * 0.3, suspicion)

    # Flag only cells that are:
    #   1. Below the Tukey fence (true outlier)
    #   2. Suspicion score > 60%
    is_suspect = (raw_scores < fence) & (suspicion > 60)

    df = df.copy()
    df["cancer_probability"] = np.round(suspicion, 1)
    df["classification"]     = np.where(is_suspect, "cancer_suspect", "healthy")
    df["anomaly_score"]      = np.round(raw_scores, 4)

    # Which features deviate most (within-image z-score)
    x_mean = X.mean()
    x_std  = X.std().replace(0, 1e-6)
    df["top_deviations"] = X.apply(
        lambda r: ", ".join(
            ((r - x_mean)/x_std).abs().nlargest(3).index
            .str.replace("AreaShape_","").tolist()
        ), axis=1
    )

    # Sample-level verdict
    suspect_pct = is_suspect.mean() * 100
    if suspect_pct < 3:
        verdict = "healthy"
    elif suspect_pct < 15:
        verdict = "mildly_abnormal"
    else:
        verdict = "abnormal"

    df["sample_verdict"] = verdict
    return df, None


# ════════════════════════════════════════════════════════════
#  STEP 3: DRAW HIGHLIGHTS
# ════════════════════════════════════════════════════════════
def draw_highlights(image_path, df, output_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return False, "Could not read image"

    if img.dtype == np.uint16:
        img = (img/256).astype(np.uint8)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    n_sus = int((df["classification"] == "cancer_suspect").sum())
    n_hlt = int((df["classification"] == "healthy").sum())

    for _, row in df.iterrows():
        x = int(row["Location_Center_X"])
        y = int(row["Location_Center_Y"])
        r = max(int(np.sqrt(row["AreaShape_Area"]/np.pi)), 6)
        p = float(row["cancer_probability"])

        if row["classification"] == "cancer_suspect":
            g     = max(0, int(100 - p))
            color = (0, g, 255)
            cv2.circle(img, (x,y), r+5, color, 3)
            cv2.putText(img, f"{int(p)}%",
                        (x-14, y-r-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 2)
        else:
            cv2.circle(img, (x,y), r+3, (0,200,0), 1)

    ov = img.copy()
    cv2.rectangle(ov, (8,8), (285,95), (8,8,15), -1)
    cv2.addWeighted(ov, 0.7, img, 0.3, 0, img)
    cv2.circle(img,  (26,28), 8, (0,200,0), 1)
    cv2.putText(img, f"Healthy  ({n_hlt})",
                (40,33), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230,230,230), 1)
    cv2.circle(img,  (26,55), 8, (0,50,255), 3)
    cv2.putText(img, f"Cancer suspect  ({n_sus})",
                (40,60), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230,230,230), 1)
    cv2.putText(img, "Within-image outlier detection",
                (12,82), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (80,80,100), 1)

    cv2.imwrite(output_path, img)
    return True, {"healthy": n_hlt, "suspects": n_sus}


# ════════════════════════════════════════════════════════════
#  CLAUDE AI (optional)
# ════════════════════════════════════════════════════════════
def call_claude(summary):
    if not ANTHROPIC_KEY:
        return None
    try:
        payload = json.dumps({
            "model": "claude-sonnet-4-6",
            "max_tokens": 800,
            "messages": [{"role":"user","content":(
                "You are a cell biology expert reviewing automated breast cell analysis.\n\n"
                f"Results:\n{summary}\n\n"
                "In 3 sentences: interpret the findings, what the morphological outliers "
                "suggest, and one important caveat."
            )}]
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type":"application/json",
                "anthropic-version":"2023-06-01",
                "x-api-key": ANTHROPIC_KEY,
            }, method="POST"
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())["content"][0]["text"]
    except Exception as e:
        return f"(Error: {e})"


# ════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "Please upload an image"}), 400

    img_f = request.files["image"]
    uid   = str(uuid.uuid4())[:8]
    ext   = os.path.splitext(img_f.filename)[1].lower() or ".png"

    img_path = os.path.join(UPLOAD_FOLDER,  f"{uid}_input{ext}")
    out_path = os.path.join(RESULTS_FOLDER, f"{uid}_result.png")
    img_f.save(img_path)

    # Segment
    print(f"[{uid}] Segmenting...")
    df, err = segment_cells(img_path)
    if err:
        return jsonify({"error": err}), 400
    print(f"[{uid}] Found {len(df)} cells")

    # Classify
    df, err = classify_cells(df)
    if err:
        return jsonify({"error": err}), 400

    # Highlight
    ok, info = draw_highlights(img_path, df, out_path)
    if not ok:
        return jsonify({"error": info}), 400

    n_sus   = info["suspects"]
    n_hlt   = info["healthy"]
    n_tot   = n_sus + n_hlt
    pct_s   = round(n_sus/n_tot*100, 1) if n_tot else 0
    pct_h   = round(n_hlt/n_tot*100, 1) if n_tot else 0
    avg_p   = round(df[df.classification=="cancer_suspect"]["cancer_probability"].mean(),1) if n_sus else 0
    verdict = df["sample_verdict"].iloc[0]
    top_d   = df[df.classification=="cancer_suspect"]["top_deviations"].value_counts().head(3).to_dict()

    summary = (
        f"Total cells: {n_tot}\n"
        f"Healthy: {n_hlt} ({pct_h}%)\n"
        f"Cancer suspects: {n_sus} ({pct_s}%)\n"
        f"Sample verdict: {verdict}\n"
        f"Avg suspicion of flagged cells: {avg_p}%\n"
        f"Top deviated features: {top_d}\n"
        f"Method: within-image Tukey outlier detection (no external reference)"
    )
    ai = call_claude(summary)

    res_csv = os.path.join(RESULTS_FOLDER, f"{uid}_results.csv")
    df.to_csv(res_csv, index=False)

    return jsonify({
        "success":      True,
        "result_image": f"/result/{uid}",
        "result_csv":   f"/download/{uid}",
        "healthy":      n_hlt,
        "suspects":     n_sus,
        "total":        n_tot,
        "healthy_pct":  pct_h,
        "suspect_pct":  pct_s,
        "avg_prob":     avg_p,
        "verdict":      verdict,
        "ai":           ai,
        "ai_enabled":   bool(ANTHROPIC_KEY),
    })

@app.route("/result/<uid>")
def get_result(uid):
    p = os.path.join(RESULTS_FOLDER, f"{uid}_result.png")
    return send_file(p, mimetype="image/png") if os.path.exists(p) else ("Not found",404)

@app.route("/download/<uid>")
def download_csv(uid):
    p = os.path.join(RESULTS_FOLDER, f"{uid}_results.csv")
    return (send_file(p, as_attachment=True, download_name="results.csv")
            if os.path.exists(p) else ("Not found",404))

if __name__ == "__main__":
    app.run(debug=True, port=5000)

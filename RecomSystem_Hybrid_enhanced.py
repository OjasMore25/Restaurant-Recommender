import os
import math
from datetime import datetime
from typing import List, Dict, Any, Optional

from flask import Flask, request, jsonify

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split as surprise_split

# Config


RESTAURANT_CSV = os.environ.get("RESTAURANT_CSV", "bangalore_dataset_with_coords.csv")
USER_ORDERS_CSV = os.environ.get("USER_ORDERS_CSV", "UserOrdersData.csv")

TOPN_SHORTLIST = int(os.environ.get("TOPN_SHORTLIST", "200"))  # shortlist size from content model
DEFAULT_KM_RADIUS = float(os.environ.get("DEFAULT_KM_RADIUS", "5"))
DEFAULT_TOP_N = int(os.environ.get("DEFAULT_TOP_N", "10"))



restaurants = pd.read_csv(RESTAURANT_CSV)
users = pd.read_csv(USER_ORDERS_CSV)

# Normalize dtypes
for col in ["rest_id", "user_id"]:
    if col in users.columns:
        users[col] = users[col].astype(str)
if "rest_id" in restaurants.columns:
    restaurants["rest_id"] = restaurants["rest_id"].astype(str)

# Fill important text fields
for col in ["Cuisines", "KnownFor", "Area", "Timing"]:
    if col in restaurants.columns:
        restaurants[col] = restaurants[col].fillna("Unknown")

# Feature engineering for content-based
restaurants["CombinedFeatures"] = (
    restaurants.get("Cuisines", "").astype(str) + " " + restaurants.get("KnownFor", "").astype(str)
)

# TF-IDF and similarity
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(restaurants["CombinedFeatures"])
content_sim = cosine_similarity(tfidf_matrix)

# Index helpers
restid_to_index = {rid: i for i, rid in enumerate(restaurants["rest_id"].tolist())}
index_to_restid = {i: rid for rid, i in restid_to_index.items()}


# Surprise SVD for Collaborative


# Ensure "rating" in users data; if not present, try to create a placeholder from any "rating" like column
if "rating" not in users.columns:
    raise ValueError("UserOrdersData.csv must contain a 'rating' column.")

reader = Reader(rating_scale=(1, 5))
surprise_data = Dataset.load_from_df(users[["user_id", "rest_id", "rating"]], reader)
trainset, testset = surprise_split(surprise_data, test_size=0.2, random_state=42)

svd = SVD()
svd.fit(trainset)

# Precompute evaluation metrics on startup (safe for demo; disable for very large datasets)
test_predictions = svd.test(testset)
rmse_value = accuracy.rmse(test_predictions, verbose=False)


# Utility: Haversine distance


def haversine_km(lat1, lon1, lat2, lon2):
    """Compute Haversine distance in KM between two (lat, lon)."""
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


# Time slot parsing


def infer_time_slot(dt: Optional[datetime] = None) -> str:
    """Return one of {'breakfast','lunch','dinner','late_night'} based on hour."""
    dt = dt or datetime.now()
    h = dt.hour
    if 6 <= h < 11:
        return "breakfast"
    if 11 <= h < 16:
        return "lunch"
    if 16 <= h < 22:
        return "dinner"
    return "late_night"

def timing_supports_slot(timing_str: str, slot: str) -> bool:
    """
    Very lightweight parser that checks if a restaurant timing string mentions a slot-ish keyword.
    This is heuristic only. If 'Timing' is structured differently, you can enhance it.
    """
    s = (timing_str or "").lower()
    slot_keywords = {
        "breakfast": ["breakfast", "8am", "9am", "10am", "morning"],
        "lunch": ["lunch", "12pm", "1pm", "2pm", "afternoon"],
        "dinner": ["dinner", "7pm", "8pm", "9pm", "evening"],
        "late_night": ["late", "11pm", "12am", "1am", "2am"]
    }
    keys = slot_keywords.get(slot, [])
    return any(k in s for k in keys) or s == "unknown"


# Filtering helpers


def filter_by_budget(df: pd.DataFrame, budget_max: Optional[float]) -> pd.DataFrame:
    if budget_max is None:
        return df
    if "AverageCost" not in df.columns:
        return df
    return df[df["AverageCost"].fillna(1e9) <= budget_max]

def filter_by_service_mode(df: pd.DataFrame, mode: Optional[str]) -> pd.DataFrame:
    if not mode:
        return df
    mode = mode.lower()
    col_map = {
        "delivery": "IsHomeDelivery",
        "takeaway": "isTakeaway",
        "dinner": "isIndoorSeating",
        "dinein": "isIndoorSeating",
        "indoor": "isIndoorSeating",
    }
    col = col_map.get(mode)
    if col and col in df.columns:
        return df[df[col].fillna(0).astype(int) == 1]
    return df

def filter_by_veg(df: pd.DataFrame, veg_only: Optional[bool]) -> pd.DataFrame:
    if veg_only is None:
        return df
    col = "isVegOnly"
    if col in df.columns:
        if veg_only:
            return df[df[col].fillna(0).astype(int) == 1]
        else:
            return df  
    return df

def filter_by_time_slot(df: pd.DataFrame, slot: Optional[str]) -> pd.DataFrame:
    if not slot:
        return df
    if "Timing" not in df.columns:
        return df
    mask = df["Timing"].apply(lambda s: timing_supports_slot(s, slot))
    return df[mask]

def filter_by_geo(df: pd.DataFrame, user_lat: Optional[float], user_lng: Optional[float], max_km: Optional[float]) -> pd.DataFrame:
    if user_lat is None or user_lng is None or "lat" not in df.columns or "lng" not in df.columns:
        return df
    lat = df["lat"].astype(float)
    lng = df["lng"].astype(float)
    dists = haversine_km(user_lat, user_lng, lat, lng)
    df = df.copy()
    df["distance_km"] = dists
    return df[df["distance_km"] <= (max_km or DEFAULT_KM_RADIUS)]


# Core recommender steps


def shortlist_by_content(seed_rest_id: str, top_n: int = TOPN_SHORTLIST) -> List[str]:
    """Return a shortlist of similar restaurants (rest_id) by content similarity to the seed rest_id."""
    if seed_rest_id not in restid_to_index:
        return []
    idx = restid_to_index[seed_rest_id]
    sims = content_sim[idx]
    
    similar_idx = np.argsort(-sims)
    similar_idx = [i for i in similar_idx if i != idx][:top_n]
    return [index_to_restid[i] for i in similar_idx]

def rank_for_user(user_id: str, rest_ids: List[str], top_n: int = DEFAULT_TOP_N) -> List[Dict[str, Any]]:
    """Rank a list of rest_ids for a user using the SVD model; return top_n with predicted score."""
    preds = []
    for rid in rest_ids:
        try:
            pred = svd.predict(str(user_id), str(rid))
            preds.append((rid, float(pred.est)))
        except Exception:
            
            continue
    preds.sort(key=lambda x: x[1], reverse=True)
    preds = preds[:top_n]
    # decorate with restaurant details
    items = []
    rmap = restaurants.set_index("rest_id")
    for rid, score in preds:
        if rid in rmap.index:
            row = rmap.loc[rid]
            items.append({
                "rest_id": rid,
                "name": row.get("Name", ""),
                "average_cost": row.get("AverageCost", None),
                "cuisines": row.get("Cuisines", ""),
                "known_for": row.get("KnownFor", ""),
                "area": row.get("Area", ""),
                "url": row.get("URL", ""),
                "pred_rating": round(score, 3),
                "distance_km": float(row.get("distance_km")) if "distance_km" in row else None,
            })
    return items

def hybrid_recommend(
    user_id: str,
    seed_rest_id: str,
    top_n: int = DEFAULT_TOP_N,
    budget_max: Optional[float] = None,
    service_mode: Optional[str] = None,
    veg_only: Optional[bool] = None,
    user_lat: Optional[float] = None,
    user_lng: Optional[float] = None,
    max_km: Optional[float] = None,
    time_slot: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Full hybrid pipeline: shortlist by content -> filter -> rank by SVD -> top_n."""
    shortlist_ids = shortlist_by_content(seed_rest_id, top_n=TOPN_SHORTLIST)
    if not shortlist_ids:
        return []

    short_df = restaurants[restaurants["rest_id"].isin(shortlist_ids)].copy()

    #= filters
    short_df = filter_by_budget(short_df, budget_max)
    short_df = filter_by_service_mode(short_df, service_mode)
    short_df = filter_by_veg(short_df, veg_only)
    if time_slot:
        short_df = filter_by_time_slot(short_df, time_slot.lower())
    short_df = filter_by_geo(short_df, user_lat, user_lng, max_km)

    if short_df.empty:
        return []

    
    ranked = rank_for_user(user_id, short_df["rest_id"].tolist(), top_n=top_n)
    return ranked


# Evaluation metrics (Top-N)


def build_topn_recommendations(model, trainset, n=5) -> Dict[str, List[str]]:
    """
    Classic Surprise top-N: for each user and each item not seen in train,
    predict and take top-N items. Warning: can be heavy on large datasets.
    """
    anti_testset = trainset.build_anti_testset()
    predictions = model.test(anti_testset)

    # Aggregate by user
    user_pred = {}
    for uid, iid, true_r, est, _ in predictions:
        user_pred.setdefault(uid, []).append((iid, est))

    # Take top n
    topn = {}
    for uid, item_ratings in user_pred.items():
        item_ratings.sort(key=lambda x: x[1], reverse=True)
        topn[uid] = [iid for (iid, _) in item_ratings[:n]]
    return topn

def precision_recall_at_k(topn: Dict[str, List[str]], testset, k=5, threshold=3.5):
    """
    Compute overall Precision@K and Recall@K using Surprise predictions testset.
    A "relevant" item is one with true rating >= threshold.
    """
    
    truth = {}
    for uid, iid, true_r in testset:
        truth.setdefault(uid, set()).add((iid, true_r))
    
    precisions = []
    recalls = []
    for uid, rec_items in topn.items():
       
        rel_items = {iid for (iid, r) in truth.get(uid, set()) if r >= threshold}
        if not rec_items:
            continue
        rec_at_k = set(rec_items[:k])
        hit = len(rec_at_k & rel_items)
        prec = hit / k if k > 0 else 0.0
        rec = hit / len(rel_items) if len(rel_items) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
    precision = float(np.mean(precisions)) if precisions else 0.0
    recall = float(np.mean(recalls)) if recalls else 0.0
    return precision, recall


try:
    topn5 = build_topn_recommendations(svd, trainset, n=5)
    p5, r5 = precision_recall_at_k(topn5, testset, k=5, threshold=3.5)
except Exception:
    p5, r5 = 0.0, 0.0


# Flask App


app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify({"status": "ok", "rmse": rmse_value, "precision@5": p5, "recall@5": r5})

@app.get("/past_orders")
def past_orders():
    user_id = request.args.get("user_id", type=str)
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    df = users[users["user_id"] == user_id].copy()
    if df.empty:
        return jsonify({"user_id": user_id, "orders": []})
    # Join with restaurant names
    rmap = restaurants.set_index("rest_id")
    items = []
    for _, row in df.iterrows():
        rid = str(row["rest_id"])
        name = rmap.loc[rid]["Name"] if rid in rmap.index else ""
        items.append({
            "rest_id": rid,
            "name": name,
            "rating": float(row.get("rating", np.nan)) if not pd.isna(row.get("rating", np.nan)) else None,
            "cost": float(row.get("cost", np.nan)) if "cost" in row and not pd.isna(row["cost"]) else None,
            "timestamp": row.get("timestamp", None)
        })
    return jsonify({"user_id": user_id, "orders": items})

@app.post("/recommend")
def recommend():
    payload = request.get_json(force=True) or {}
    user_id = str(payload.get("user_id", "")).strip()
    seed_rest_id = str(payload.get("seed_rest_id", "")).strip()
    top_n = int(payload.get("top_n", DEFAULT_TOP_N))
    budget_max = payload.get("budget_max", None)
    service_mode = payload.get("service_mode", None)  # "delivery", "takeaway", "dinner"
    veg_only = payload.get("veg_only", None)

    user_lat = payload.get("user_lat", None)
    user_lng = payload.get("user_lng", None)
    max_km = payload.get("max_km", DEFAULT_KM_RADIUS)
    time_slot = payload.get("time_slot", None)  # "breakfast","lunch","dinner","late_night"

    if not user_id or not seed_rest_id:
        return jsonify({"error": "user_id and seed_rest_id are required"}), 400

    # Auto-infer time_slot if requested
    if time_slot == "auto":
        time_slot = infer_time_slot()

    try:
        recs = hybrid_recommend(
            user_id=user_id,
            seed_rest_id=seed_rest_id,
            top_n=top_n,
            budget_max=float(budget_max) if budget_max is not None else None,
            service_mode=service_mode,
            veg_only=bool(veg_only) if veg_only is not None else None,
            user_lat=float(user_lat) if user_lat is not None else None,
            user_lng=float(user_lng) if user_lng is not None else None,
            max_km=float(max_km) if max_km is not None else None,
            time_slot=str(time_slot).lower() if time_slot else None,
        )
        return jsonify({"user_id": user_id, "seed_rest_id": seed_rest_id, "top_n": top_n, "results": recs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/metrics")
def metrics():
    return jsonify({
        "rmse": rmse_value,
        "precision@5": p5,
        "recall@5": r5
    })

if __name__ == "__main__":
   
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)


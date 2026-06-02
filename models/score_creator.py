"""
models/score_creator.py  —  THE INTEGRATION BRIDGE
═══════════════════════════════════════════════════
Connects the agent/UI engine to the Data+ML layer that lives in the sibling
repo `ratefluencer-copilot/proj`. This is the single function the orchestrator
calls when USE_REAL_MODELS = True.

It:
  1. puts the ML repo on sys.path (no copying of code),
  2. loads the trained True-Impact model + SHAP explainer once (cached),
  3. for a creator id, runs all five ML outputs (impact, authenticity,
     brand-match, growth, drivers) against the demo brief,
  4. maps the ML output dicts into the EXACT shape the UI expects (the same
     shape as utils/dummy_scores.py — impact / authenticity / match /
     predicted_roi / success_prob / drivers / authenticity_detail / audience_fit
     plus the creator profile fields).

If the ML layer or trained model is missing, every public function degrades
gracefully (returns {} / [] ) so the orchestrator falls back to dummy data and
the UI never crashes.

Brief bridging: the engine's parsed brief uses
    {category, target_audience:{age_min,age_max,gender,geo}, goal, tone, budget}
with geo as a country NAME ("India"). The ML BrandBrief uses target_geo=["IN"]
and 2-letter region codes, so we translate here.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 1) Locate the ML repo and put it on sys.path
# ─────────────────────────────────────────────────────────────────────────────
# This file:  .../Ratefluencer-Engine-main/models/score_creator.py
# ML proj:    .../ratefluencer-copilot/proj
_ENGINE_ROOT = Path(__file__).resolve().parents[1]          # Ratefluencer-Engine-main
_WORKSPACE   = _ENGINE_ROOT.parent                          # d:\ratefluencer
# Allow an env override; otherwise use the conventional sibling layout.
_PROJ = Path(os.environ.get(
    "RATEFLUENCER_PROJ",
    _WORKSPACE / "ratefluencer-copilot" / "proj",
))

ML_AVAILABLE = (_PROJ / "config.py").exists()
if ML_AVAILABLE and str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

# Point the ML layer's storage at its own data/ + models_store/ (absolute), so it
# finds the trained model and seeded DB regardless of the cwd Streamlit runs from.
if ML_AVAILABLE:
    os.environ.setdefault("RATEFLUENCER_DATA_DIR",   str(_PROJ / "data"))
    os.environ.setdefault("RATEFLUENCER_MODELS_DIR", str(_PROJ / "models_store"))
    os.environ.setdefault("RATEFLUENCER_DB",         str(_PROJ / "data" / "ratefluencer.db"))


# ─────────────────────────────────────────────────────────────────────────────
# 2) Geo / brief translation helpers
# ─────────────────────────────────────────────────────────────────────────────
_GEO_TO_CODE = {
    "india": "IN", "in": "IN",
    "usa": "US", "united states": "US", "america": "US", "us": "US",
    "uk": "UK", "united kingdom": "UK",
    "uae": "AE", "ae": "AE",
    "global": "IN", "worldwide": "IN",   # default the catch-alls to the demo market
}


def _geo_code(geo: str) -> str:
    return _GEO_TO_CODE.get((geo or "").strip().lower(), "IN")


def _to_brand_brief(parsed_brief: dict | None):
    """Translate the engine's parsed brief (or None) into an ML BrandBrief."""
    from src.store.schema import BrandBrief

    pb = parsed_brief or {}
    ta = pb.get("target_audience", {}) or {}
    geo_name = ta.get("geo", "India")
    return BrandBrief(
        brief_id="engine_live",
        raw_text=pb.get("raw", "") or pb.get("category", ""),
        category=pb.get("category"),
        target_age_min=ta.get("age_min"),
        target_age_max=ta.get("age_max"),
        target_gender=(ta.get("gender") if ta.get("gender") in ("female", "male") else None),
        target_geo=[_geo_code(geo_name)],
        target_interests=[pb.get("category")] if pb.get("category") else [],
        goal=pb.get("goal"),
        tone=pb.get("tone"),
        budget=pb.get("budget"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3) Cached model + explainer (loaded once per process)
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_models():
    """Return (impact_model, explainer) or (None, None) if unavailable."""
    if not ML_AVAILABLE:
        return None, None
    try:
        from src.models.true_impact import TrueImpactModel
        from src.models.explain import Explainer
        from src.data import embeddings as emb
        model = TrueImpactModel.load()
        explainer = Explainer(model)
        # Warm the MiniLM encoder ONCE now (its first .encode() pays a large
        # lazy-init cost ~30s on CPU). Doing it here means we pay it a single
        # time per process, not on the first creator the user scores.
        try:
            emb.embed_text("warmup")
        except Exception:
            pass
        return model, explainer
    except Exception as e:                                    # no trained model yet
        print(f"[score_creator] could not load trained model: {e}")
        return None, None


@lru_cache(maxsize=1)
def _vector_store():
    """Cached handle to the FAISS vector store (creator embeddings from ingest).
    Returns None if unavailable, so callers fall back to live embedding."""
    if not ML_AVAILABLE:
        return None
    try:
        from src.store.vector import VectorStore
        vs = VectorStore()
        return vs if len(vs) > 0 else None
    except Exception:
        return None


@lru_cache(maxsize=1)
def _growth_model():
    """Cached, pre-fitted Growth model.

    Critical for performance: GrowthModel.fit() calibrates against the ENTIRE
    population (15k creators) — ~45s. Without caching, predict_growth() refits it
    on every single creator score, which is what made the pipeline hang. We fit
    once here and reuse. Returns None if unavailable (growth just shows 0)."""
    if not ML_AVAILABLE:
        return None
    try:
        from src.models.growth import GrowthModel
        # Prefer a pre-fitted model saved by prepare_demo (instant load); only
        # refit (~45s over the whole population) if none exists.
        try:
            return GrowthModel.load()
        except (FileNotFoundError, OSError):
            return GrowthModel().fit()
    except Exception as e:
        print(f"[score_creator] could not fit growth model: {e}")
        return None


@lru_cache(maxsize=1)
def _default_brief():
    """A BrandBrief for the canonical demo, used when no parsed brief is passed."""
    return _to_brand_brief({
        "category": "skincare",
        "target_audience": {"age_min": 22, "age_max": 35, "gender": "female", "geo": "India"},
        "goal": "sales", "tone": "authentic",
        "raw": "DTC skincare for women 22-35 in India, goal sales",
    })


# ─────────────────────────────────────────────────────────────────────────────
# 4) Output mapping helpers  (ML dicts → UI dict shape)
# ─────────────────────────────────────────────────────────────────────────────
def _map_drivers(ml_drivers: list[dict]) -> list[dict]:
    """ML driver {feature,label,effect,value,shap} → UI {feature,effect,value}.

    The UI shows a human label + a short value string; it does not show the raw
    SHAP number. We surface the friendly label and a readable value.
    """
    out = []
    for d in ml_drivers or []:
        label = d.get("label") or d.get("feature", "")
        val = d.get("value")
        # render the feature value compactly (rates as %, big numbers with commas)
        if isinstance(val, (int, float)):
            if 0 < abs(val) < 1:
                val_str = f"{val*100:.1f}%"
            elif abs(val) >= 1000:
                val_str = f"{val:,.0f}"
            else:
                val_str = f"{val:.2f}"
        else:
            val_str = str(val) if val is not None else ""
        out.append({
            "feature": str(label).capitalize(),
            "effect":  "+" if d.get("effect", "+") == "+" else "-",
            "value":   val_str,
        })
    return out


def _audience_fit(influencer_id: str, brief) -> dict:
    """Per-axis audience match (age / gender / geo) as 0-100 ints for the UI bars."""
    try:
        from src.store import repo
        from src.data.features import _audience_demo_match  # reused, but we want a breakdown
        aud = repo.get_audience_profile(influencer_id)
        if aud is None:
            return {"age_match": 50, "gender_match": 50, "geo_match": 50}

        # Age overlap with the target band.
        age = 50
        if brief.target_age_min is not None and brief.target_age_max is not None and aud.age_distribution:
            bands = {"13-17": (13, 17), "18-24": (18, 24), "25-34": (25, 34),
                     "35-44": (35, 44), "45+": (45, 99)}
            covered = sum(
                (aud.age_distribution.get(b, 0.0) or 0.0)
                for b, (lo, hi) in bands.items()
                if hi >= brief.target_age_min and lo <= brief.target_age_max
            )
            age = int(round(covered * 100))

        # Gender match.
        gender = 50
        if brief.target_gender and aud.gender_split:
            gender = int(round((aud.gender_split.get(brief.target_gender, 0.0) or 0.0) * 100))

        # Geo match.
        geo = 50
        if brief.target_geo and aud.geo_distribution:
            geo = int(round(min(1.0, sum(
                (aud.geo_distribution.get(str(g), 0.0) or 0.0) for g in brief.target_geo
            )) * 100))

        return {"age_match": age, "gender_match": gender, "geo_match": geo}
    except Exception:
        return {"age_match": 50, "gender_match": 50, "geo_match": 50}


def _brand_match_from_vec(influencer_id: str, brief, vec: dict) -> dict:
    """Brand-match score reusing the already-built feature vec (no recompute).

    Mirrors src.models.brand_match.brand_match but takes the fit features from the
    shared vec (brand_fit_similarity + audience_demo_match are already in it), so
    we don't rebuild features or re-embed. Returns {brand_match_score}.
    """
    try:
        from src.models import brand_match as bm_mod
        from src.store import repo
        inf = repo.get_influencer(influencer_id)
        semantic = float(vec.get("brand_fit_similarity", 0.0))
        audience = float(vec.get("audience_demo_match", 0.0))
        cat = bm_mod._category_match(inf.content_category if inf else None, brief.category)
        raw = bm_mod.SEM_W * semantic + bm_mod.AUD_W * audience + bm_mod.CAT_W * cat
        score = int(max(0, min(100, round(raw * 100))))
        return {"brand_match_score": score}
    except Exception:
        return {"brand_match_score": 50}


def _profile_fields(influencer_id: str) -> dict:
    """Creator profile fields the UI renders (handle, followers, etc.)."""
    from src.store import repo
    inf = repo.get_influencer(influencer_id)
    if inf is None:
        return {}
    return {
        "influencer_id":    inf.influencer_id,
        "handle":           inf.handle,
        "display_name":     inf.display_name or inf.handle,
        "platform":         inf.platform or "instagram",
        "followers":        inf.followers or 0,
        "content_category": inf.content_category or "",
        "region":           inf.region or "",
        "verified":         bool(inf.verified),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5) THE public function the orchestrator calls
# ─────────────────────────────────────────────────────────────────────────────
def score_creator(influencer_id: str, brief_id: str | None = None,
                  parsed_brief: dict | None = None) -> dict:
    """Full real-model score for one creator, in the UI's expected dict shape.

    Args:
        influencer_id: id present in the ML SQLite DB.
        brief_id:      unused (kept for orchestrator signature compatibility).
        parsed_brief:  optional engine parsed-brief; if omitted, the demo brief.

    Returns:
        UI score dict, or {} if the creator/model is unavailable (→ dummy fallback).
    """
    if not ML_AVAILABLE:
        return {}

    model, explainer = _load_models()
    if model is None:
        return {}

    try:
        from src.models.explain import template_rationale
        from src.store import repo
        from src.data.features import compute_creator_features, compute_fit_features

        brief = _to_brand_brief(parsed_brief) if parsed_brief else _default_brief()

        profile = _profile_fields(influencer_id)
        if not profile:
            return {}

        # --- Build the feature vector ONCE and reuse it everywhere ---------- #
        # (Previously impact, brand-match, and SHAP each recomputed it + re-embedded
        #  the brief — ~3x the work per creator. Compute once → big CPU saving.)
        # Reuse the creator's embedding from the FAISS store (built at ingest) so
        # we don't re-run the CPU-expensive MiniLM encoder per creator at serve time.
        vs = _vector_store()
        creator_vec = vs.get_vector(influencer_id) if vs is not None else None
        vec = compute_creator_features(influencer_id)
        vec.update(compute_fit_features(influencer_id, brief, creator_vec=creator_vec))

        # --- True-Impact (ROI + success + 0-100 score) from the shared vec --- #
        X = model._row(vec)
        roi_pred = float(model.predict_roi(X)[0])
        success_prob = float(model.predict_success(X)[0])
        true_impact_score = int(model.impact_score(roi_pred, success_prob))
        impact = {"true_impact_score": true_impact_score,
                  "predicted_roi": round(roi_pred, 4),
                  "success_prob": round(success_prob, 4)}

        # --- Authenticity (precomputed at data-prep time; read from DB) ----- #
        auth = repo.get_authenticity(influencer_id)
        if auth is not None:
            authenticity_score = int(auth.authenticity_score)
            auth_detail = {
                "bot_follower_pct":    round((auth.bot_follower_pct or 0.0) * 100, 1),
                "engagement_pod_flag": bool(auth.engagement_pod_flag),
                "spike_anomaly_score": round(float(auth.spike_anomaly_score or 0.0), 2),
                "comment_spam_ratio":  round(float(auth.comment_spam_ratio or 0.0), 2),
                "flags":               list(auth.flags or []),
            }
        else:
            authenticity_score = 70
            auth_detail = {"bot_follower_pct": 0.0, "engagement_pod_flag": False,
                           "spike_anomaly_score": 0.0, "comment_spam_ratio": 0.0, "flags": []}

        # --- Brand match (compute from the shared vec, no recompute) -------- #
        bm = _brand_match_from_vec(influencer_id, brief, vec)

        # --- Growth (use the cached, pre-fitted model — see _growth_model) -- #
        try:
            gm = _growth_model()
            growth_score = int(gm.score(influencer_id).get("growth_potential_score", 0)) if gm else 0
        except Exception:
            growth_score = 0

        # --- SHAP drivers + a rationale (reuse the same vec) --------------- #
        ml_drivers = explainer.drivers_for(vec, top_n=4)
        drivers = _map_drivers(ml_drivers)
        rationale = template_rationale(ml_drivers, auth_detail["flags"], impact["true_impact_score"])

        result = {
            # headline scores the UI reads
            "impact":        int(impact["true_impact_score"]),
            "authenticity":  authenticity_score,
            "match":         int(bm["brand_match_score"]),
            "growth":        growth_score,
            "predicted_roi": float(impact["predicted_roi"]),
            "success_prob":  float(impact["success_prob"]),
            # status/flag filled by the ranker's fraud guardrail, but seed sane defaults
            "status":        "recommended",
            "flag_reason":   None,
            "rationale":     rationale,
            "drivers":            drivers,
            "authenticity_detail": auth_detail,
            "audience_fit":       _audience_fit(influencer_id, brief),
        }
        result.update(profile)   # handle / followers / display_name / ...
        return result

    except Exception as e:
        print(f"[score_creator] scoring {influencer_id} failed: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# 6) Candidate retrieval backed by the ML FAISS vector store
# ─────────────────────────────────────────────────────────────────────────────
# The two hand-crafted "reveal" creators — the heart of the demo (gem vs fraud).
# They live among 15k creators, so plain top-k semantic retrieval won't always
# surface them. We PIN them into the candidate set so the reveal reliably appears,
# then they rank naturally (gem near top, fraud flagged to the bottom). Pinning is
# on by default for the demo; set RATEFLUENCER_PIN_SEEDS=0 to disable.
REVEAL_SEED_IDS = ("inf_9001", "inf_9002")  # gem @minimal.skin, fraud @famous.face


def get_candidates_real(parsed_brief: dict, top_k: int = 20) -> list[str]:
    """Retrieve candidate ids from the ML vector store (FAISS). [] on any failure.

    The reveal seed creators are pinned in (deduped) so the gem-vs-fraud story
    always shows in the shortlist regardless of where they fall in cosine rank.
    """
    if not ML_AVAILABLE:
        return []
    try:
        from src.models.brand_match import get_candidates as ml_get_candidates
        brief = _to_brand_brief(parsed_brief)
        hits = ml_get_candidates(brief, k=top_k)
        ids = [h["influencer_id"] for h in hits]

        # Pin the reveal seed creators ONLY for briefs in their own category
        # (skincare) — otherwise a skincare creator would pollute, e.g., a finance
        # shortlist. The seeds are the demo's hero examples for the skincare brief.
        if os.environ.get("RATEFLUENCER_PIN_SEEDS", "1") == "1" and (brief.category or "").lower() == "skincare":
            from src.store import repo
            seeds = [s for s in REVEAL_SEED_IDS if repo.get_influencer(s) is not None]
            ids = list(dict.fromkeys(seeds + ids))   # seeds first, deduped
        return ids
    except Exception as e:
        print(f"[score_creator] get_candidates_real failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# 7) Quick self-test (run directly):  py models/score_creator.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("ML_AVAILABLE:", ML_AVAILABLE, "| PROJ:", _PROJ)
    for iid in ("inf_9001", "inf_9002"):
        s = score_creator(iid)
        if s:
            print(f"\n{iid}  {s['handle']}  ({s['followers']:,} followers)")
            print(f"  impact={s['impact']} auth={s['authenticity']} match={s['match']} "
                  f"growth={s['growth']} roi={s['predicted_roi']:.1f}x")
            print(f"  status seed={s['status']} flags={s['authenticity_detail']['flags']}")
            print(f"  rationale: {s['rationale']}")
        else:
            print(f"{iid}: no score (fallback)")

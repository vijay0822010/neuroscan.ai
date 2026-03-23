"""
app/routers/analysis.py
POST /api/v1/analyse — full multimodal assessment pipeline.
"""

import uuid, json, logging, hashlib, math
from datetime import datetime

import httpx
from fastapi import APIRouter, File, UploadFile, Form, HTTPException

from app.services.resnet_service   import ResNetService
from app.services.wav2vec2_service import Wav2Vec2Service
from app.services.fusion_service   import FusionService
from app.models.schemas            import AnalysisResponse
from app.core.config               import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Singleton ML services — initialised once at startup
resnet_svc  = ResNetService()
wav2vec_svc = Wav2Vec2Service()
fusion_svc  = FusionService()

# In-memory result cache keyed by report_id
result_cache: dict[str, dict] = {}


# ─────────────────────────────────────────────────────────────────────────────
# DUMMY BACKEND: Clinical Analysis Structure
#
# This function shows HOW the backend builds the clinical analysis object
# from model outputs. In production the Groq API call (_call_groq) is used.
# This dummy is the FALLBACK when the API call fails — it generates
# structured clinical text directly from biomarker values.
#
# Comment lines explain every step of the analysis generation.
# ─────────────────────────────────────────────────────────────────────────────

def _dummy_clinical_analysis(
    stroke:   dict,   # ResNet-512 output: 8 stroke biomarkers
    acoustic: dict,   # Wav2Vec2-768 output: 8 acoustic biomarkers
    nss:      float,  # NSS score from fusion pipeline
    z:        float,  # Z-score = Sigmoid(W·X + b)
    risk:     str,    # LOW / MODERATE / HIGH
    conf:     float,  # model confidence 0.85-0.98
    img_s:    float,  # image layer-weighted score
    aud_s:    float,  # audio layer-weighted score
    patient:  str = "",
) -> dict:
    # ── Step 1: Read biomarkers from ResNet stroke classification head ─────────
    tremor   = stroke.get("tremor_index", 0.5)         # pen tremor 0-1
    cont     = stroke.get("stroke_continuity", 0.5)    # stroke smoothness 0-1
    micro    = stroke.get("micrographia_score", 0.4)   # letter size abnormality
    pvar     = stroke.get("pressure_variance", 0.5)    # ink pressure consistency
    bdev     = stroke.get("baseline_deviation", 0.3)   # writing plane stability
    angular  = stroke.get("angular_consistency", 0.5)  # stroke angle regularity
    spacing  = stroke.get("letter_spacing_std", 0.3)   # letter spacing variation
    penlift  = stroke.get("pen_lift_freq", 0.4)        # pen-lift frequency

    # ── Step 2: Read biomarkers from Wav2Vec2 acoustic classification head ─────
    speech_r = acoustic.get("speech_rate", 120.0)      # words per minute
    vtremor  = acoustic.get("voice_tremor", 0.4)       # vocal tremor index
    dysarth  = acoustic.get("dysarthria_index", 0.3)   # dysarthria severity
    clarity  = acoustic.get("articulation_clarity", 0.6) # phoneme clarity
    breath   = acoustic.get("breath_support", 0.6)     # breath pressure quality
    pause    = acoustic.get("pause_duration_mean", 0.3) # mean pause length (s)
    pitch_v  = acoustic.get("pitch_variability", 30.0) # F0 variation (Hz)
    phon_cv  = acoustic.get("phoneme_duration_cv", 0.2) # phoneme timing variation

    # ── Step 3: Apply clinical thresholds to classify each biomarker ──────────
    tremor_hi  = tremor  > 0.70   # severe pen tremor
    tremor_mod = 0.40 < tremor <= 0.70  # moderate pen tremor
    micro_sig  = micro   > 0.55   # clinically significant micrographia
    cont_poor  = cont    < 0.45   # poor stroke continuity
    vtrem_hi   = vtremor > 0.55   # significant voice tremor
    dys_hi     = dysarth > 0.50   # significant dysarthria
    slow       = speech_r < 80.0  # bradylalia
    fast       = speech_r > 160.0 # accelerated speech (festination)
    paused     = pause   > 0.60   # prolonged pauses

    # ── Step 4: Build clinical summary from threshold results ─────────────────
    tremor_txt = (
        f"Marked pen tremor (index {tremor:.3f}) consistent with action or resting tremor."
        if tremor_hi else
        f"Moderate tremor signature ({tremor:.3f}) indicating mild motor pathway involvement."
        if tremor_mod else
        f"Tremor index {tremor:.3f} within normal range."
    )
    micro_txt = (
        f"Clinically significant micrographia ({micro:.3f} > 0.55); progressive "
        "letter size reduction suggestive of basal ganglia involvement."
        if micro_sig else
        f"No significant micrographia (score {micro:.3f})."
    )
    vtrem_txt = (
        f"Pathological voice tremor ({vtremor:.3f} > 0.55) detected across temporal "
        "transformer layers; consistent with tremor-modulated phonation."
        if vtrem_hi else
        f"Normal phonatory stability; voice tremor {vtremor:.3f} within range."
    )
    rate_txt = (
        f"Bradylalia present ({speech_r:.0f} wpm < 80); possible motor initiation difficulty."
        if slow else
        f"Accelerated speech rate ({speech_r:.0f} wpm); possible festination pattern."
        if fast else
        f"Speech rate {speech_r:.0f} wpm within normal range."
    )

    # ── Step 5: Compose clinical summary paragraph ────────────────────────────
    summary = (
        f"Assessment of {patient or 'this patient'} yields NSS={nss:.4f} (Z={z:.4f}), "
        f"{risk} neurological risk, confidence {conf*100:.1f}%. "
        f"{tremor_txt} {vtrem_txt} {rate_txt} "
        f"Multimodal fusion (img={img_s:.4f}, aud={aud_s:.4f}) confirms {risk} classification."
    )

    # ── Step 6: Build handwriting-specific findings from ResNet layer output ───
    hw = (
        f"{tremor_txt} "
        f"Stroke continuity {cont:.3f} {'poor — frequent pen interruptions' if cont_poor else 'adequate'}. "
        f"{micro_txt} "
        f"Baseline deviation {bdev:.3f}; angular consistency {angular:.3f}; "
        f"pen-lift frequency {penlift:.3f}."
    )

    # ── Step 7: Build speech-specific findings from Wav2Vec2 transformer output
    sp = (
        f"{vtrem_txt} "
        f"Dysarthria index {dysarth:.3f} {'(significant)' if dys_hi else '(normal)'}; "
        f"articulation clarity {clarity:.3f}. "
        f"{rate_txt} "
        f"Breath support {breath:.3f}; mean pause {pause:.3f}s "
        f"{'(prolonged)' if paused else '(normal)'}; "
        f"pitch variability {pitch_v:.1f}Hz; phoneme CV {phon_cv:.3f}."
    )

    # ── Step 8: Generate 5 neurological indicators with biomarker values ──────
    indicators = [
        f"Pen tremor index {tremor:.4f} — "
        + ("severe resting/action tremor" if tremor_hi else "moderate tremor" if tremor_mod else "normal"),
        f"Micrographia score {micro:.4f} — "
        + ("clinically significant (>0.55)" if micro_sig else "within normal range"),
        f"Voice tremor index {vtremor:.4f} — "
        + ("pathological (>0.55)" if vtrem_hi else "normal phonatory control"),
        f"Dysarthria index {dysarth:.4f} — "
        + ("significant motor speech disorder" if dys_hi else "normal speech production"),
        f"NSS {nss:.5f} / Z {z:.5f} — {risk} risk, confidence {conf*100:.1f}%",
    ]

    # ── Step 9: Risk-level specific recommendations ───────────────────────────
    if risk == "HIGH":
        recs = [
            "Urgent movement disorder specialist referral within 2-4 weeks",
            f"Quantitative tremor analysis (accelerometry) — current index {tremor:.3f}",
            "Brain MRI with DaTscan for dopaminergic system assessment",
            "Formal UPDRS motor subscale evaluation",
            "Occupational therapy for adaptive writing aids",
        ]
    elif risk == "MODERATE":
        recs = [
            "Neurological consultation within 4-8 weeks",
            f"Baseline UPDRS assessment (tremor index {tremor:.3f})",
            "Speech and language therapy for early dysarthria intervention",
            "Repeat multimodal assessment in 3 months to track progression",
            "Physiotherapy for fine motor rehabilitation exercises",
        ]
    else:
        recs = [
            "Routine neurological follow-up in 12 months",
            f"Biomarkers within normal range (NSS {nss:.4f})",
            "Maintain regular physical exercise emphasising fine motor activities",
            "Annual screening if family history of movement disorders",
        ]

    # ── Step 10: Follow-up plan based on risk tier ────────────────────────────
    if risk == "HIGH":
        follow = (
            f"Week 1-2: Refer to movement disorder clinic; blood panel. "
            f"Week 3-4: Brain MRI + DaTscan. Month 2: Full UPDRS-III exam. "
            f"Month 3: Repeat NeuroScan (compare tremor {tremor:.3f}, "
            f"voice tremor {vtremor:.3f}). Month 6: MDT review."
        )
    elif risk == "MODERATE":
        follow = (
            f"Month 1: Neurology consultation. Month 2: SLT assessment. "
            f"Month 3: Repeat NeuroScan (micrographia {micro:.3f}, "
            f"dysarthria {dysarth:.3f}). Month 6: Full re-evaluation."
        )
    else:
        follow = (
            f"Year 1: Repeat NeuroScan. Watch for tremor >0.40 "
            f"(current {tremor:.3f}), speech rate <80 wpm "
            f"(current {speech_r:.0f}), or new micrographia."
        )

    # ── Step 11: Confidence note from model agreement ─────────────────────────
    conf_note = (
        f"Confidence {conf*100:.1f}% driven by NSS distance from boundaries and "
        f"cross-modal agreement (img={img_s:.4f}, aud={aud_s:.4f}). "
        + ("High agreement — reliable." if conf >= 0.92 else
           "Moderate agreement — repeat with longer samples recommended."
           if conf < 0.88 else "Good agreement.")
    )

    # ── Step 12: Risk rationale citing specific biomarker values ──────────────
    if risk == "HIGH":
        rationale = (
            f"HIGH risk: NSS {nss:.4f} < 0.50. Drivers: tremor {tremor:.3f}, "
            f"micrographia {micro:.3f}, voice tremor {vtremor:.3f}, "
            f"dysarthria {dysarth:.3f}."
        )
    elif risk == "MODERATE":
        rationale = (
            f"MODERATE risk: NSS {nss:.4f} in 0.50-0.75 range. "
            f"Tremor {tremor:.3f}, voice tremor {vtremor:.3f}, "
            f"speech rate {speech_r:.0f} wpm."
        )
    else:
        rationale = (
            f"LOW risk: NSS {nss:.4f} >= 0.75. "
            f"Tremor {tremor:.3f}, voice tremor {vtremor:.3f}, "
            f"clarity {clarity:.3f} all within normal limits."
        )

    # ── Step 13: Differential diagnosis from pattern matching ─────────────────
    if risk == "HIGH" and tremor_hi and micro_sig:
        diff = ["Parkinson's Disease (idiopathic, early-mid stage)",
                "Drug-induced parkinsonism", "MSA-P (parkinsonian variant)"]
    elif risk == "HIGH" and vtrem_hi and dys_hi:
        diff = ["Essential tremor with vocal involvement",
                "Cerebellar ataxia with dysarthria",
                "ALS — bulbar onset"]
    elif risk == "MODERATE":
        diff = ["Early-stage Parkinson's Disease",
                "Essential tremor (action-predominant)",
                "Mild cognitive impairment with motor features"]
    else:
        diff = ["No significant neurological condition indicated",
                "Benign essential tremor (if minor features present)",
                "Age-related motor changes (if applicable)"]

    # ── Step 14: Lifestyle suggestions based on risk ──────────────────────────
    if risk in ("HIGH", "MODERATE"):
        life = [
            "Aerobic exercise 30 min/day, 5 days/week for dopaminergic support",
            "Daily handwriting practice with OT-guided exercises",
            "Vocal exercises and reading aloud 10-15 min/day",
            f"Reduce caffeine/alcohol — both exacerbate tremor (index {tremor:.3f})",
        ]
    else:
        life = [
            "Regular fine motor activities (drawing, music, crafts)",
            "Mediterranean diet for neuroprotective benefit",
            "Adequate sleep hygiene for motor system health",
            "Mental engagement through language or instrument practice",
        ]

    # ── Step 15: Return assembled clinical analysis dict ─────────────────────
    return {
        "clinical_summary":        summary,
        "handwriting_findings":    hw,
        "speech_findings":         sp,
        "neurological_indicators": indicators,
        "recommendations":         recs,
        "follow_up":               follow,
        "confidence_note":         conf_note,
        "risk_rationale":          rationale,
        "differential_diagnosis":  diff,
        "lifestyle_suggestions":   life,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GROQ API CALL — real clinical analysis via LLM
# The dummy above is the fallback; this is the primary analysis path.
# ─────────────────────────────────────────────────────────────────────────────

_PROMPT = """\
You are a specialist neurologist AI. Analyse this patient's multimodal \
neurological assessment and produce a clinical report.

Patient: {patient_name}
NSS Score: {nss_score}   Z-Score: {z_score}
Risk: {risk_level}   Confidence: {confidence}%

HANDWRITING (ResNet-512):
tremor_index={tremor_index}  pressure_variance={pressure_variance}
stroke_continuity={stroke_continuity}  letter_spacing_std={letter_spacing_std}
baseline_deviation={baseline_deviation}  micrographia_score={micrographia_score}
pen_lift_freq={pen_lift_freq}  angular_consistency={angular_consistency}

SPEECH (Wav2Vec2-768):
speech_rate={speech_rate}  pause_duration_mean={pause_duration_mean}
pitch_variability={pitch_variability}  voice_tremor={voice_tremor}
articulation_clarity={articulation_clarity}  dysarthria_index={dysarthria_index}
breath_support={breath_support}  phoneme_duration_cv={phoneme_duration_cv}

FUSION:
img_score={img_score}  aud_score={aud_score}  fused={fused_feature}

Respond ONLY with valid JSON (no markdown, no extra text):
{{
  "clinical_summary": "3-4 sentence expert interpretation citing specific biomarker values",
  "handwriting_findings": "2-3 sentence interpretation of stroke biomarkers",
  "speech_findings": "2-3 sentence interpretation of acoustic biomarkers",
  "neurological_indicators": ["5 indicators with biomarker values cited"],
  "recommendations": ["4-5 specific clinical recommendations for this risk level"],
  "follow_up": "Detailed follow-up plan with timeline and tests",
  "confidence_note": "Note on what drives confidence in this assessment",
  "risk_rationale": "Why this exact risk level citing specific values",
  "differential_diagnosis": ["3 neurological conditions matching this pattern"],
  "lifestyle_suggestions": ["3-4 evidence-based modifications for these findings"]
}}"""


async def _call_groq(
    stroke:   dict,
    acoustic: dict,
    nss:      float,
    z:        float,
    risk:     str,
    conf:     float,
    img_s:    float,
    aud_s:    float,
    fused:    float,
    patient:  str,
) -> dict:
    # Format metric values — large floats (speech_rate, pitch_variability) use 2dp
    def fmt(v): return f"{v:.2f}" if isinstance(v, float) and abs(v) >= 10 else (f"{v:.4f}" if isinstance(v, float) else str(v))
    all_metrics = {**stroke, **acoustic}

    prompt = _PROMPT.format(
        patient_name=patient or "Anonymous",
        nss_score=f"{nss:.4f}", z_score=f"{z:.4f}",
        risk_level=risk, confidence=f"{conf*100:.1f}",
        img_score=f"{img_s:.6f}", aud_score=f"{aud_s:.6f}",
        fused_feature=f"{fused:.6f}",
        **{k: fmt(v) for k, v in all_metrics.items()},
    )

    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {settings.GROQ_API_KEY}",
    }
    payload = {
        "model":       settings.GROQ_MODEL,
        "messages":    [{"role": "user", "content": prompt}],
        "max_tokens":  1500,
        "temperature": 0.20,
    }

    logger.info("Groq: sending clinical analysis request…")
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(settings.GROQ_API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    raw = data["choices"][0]["message"]["content"]
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
    clean = clean.strip()

    return json.loads(clean)


# ── Upload helper ─────────────────────────────────────────────────────────────

async def _read_upload(upload: UploadFile, max_mb: int = 50) -> bytes:
    data = await upload.read()
    if len(data) == 0:
        raise HTTPException(422, f"File '{upload.filename}' is empty.")
    if len(data) > max_mb * 1024 * 1024:
        raise HTTPException(413, f"File '{upload.filename}' exceeds {max_mb} MB limit.")
    return data


def _blocks_to_dicts(blocks) -> list[dict]:
    return [{"block_id":b.block_id,"dim_start":b.dim_start,"dim_end":b.dim_end,
             "mean":b.mean,"variance":b.variance,"l2_norm":b.l2_norm,
             "skewness":b.skewness,"kurtosis":b.kurtosis,
             "entropy":b.entropy,"activation":b.activation}
            for b in blocks]


# ─────────────────────────────────────────────────────────────────────────────
# ROUTE: POST /api/v1/analyse
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/analyse", response_model=AnalysisResponse)
async def analyse(
    image:        UploadFile = File(...),
    audio:        UploadFile = File(...),
    patient_name: str        = Form(default=""),
):
    report_id = f"NSA-{uuid.uuid4().hex[:10].upper()}"
    timestamp = datetime.utcnow().isoformat() + "Z"
    logger.info(f"[{report_id}] start | patient={patient_name!r}")

    # Step 1: Read uploaded file bytes
    img_bytes = await _read_upload(image)
    aud_bytes = await _read_upload(audio)

    # Step 2: ResNet-512 — preprocess image → run full residual pipeline → embeddings + biomarkers
    try:
        img_res = resnet_svc.analyse(img_bytes)
    except Exception as e:
        raise HTTPException(422, f"Image analysis failed: {e}")

    # Step 3: Wav2Vec2-768 — preprocess audio → CNN + transformer → embeddings + biomarkers
    try:
        aud_res = wav2vec_svc.analyse(aud_bytes)
    except Exception as e:
        raise HTTPException(422, f"Audio analysis failed: {e}")

    # Step 4: Multimodal fusion — biomarker score + layer analysis → NSS
    # Biomarkers from both models are passed directly so the pathology score
    # is computed from the actual clinical values, not just embedding statistics.
    fusion = fusion_svc.fuse(
        img_embedding    = img_res.embedding,
        aud_embedding    = aud_res.embedding,
        img_bytes_seed   = hashlib.sha256(img_bytes[:512]).hexdigest(),
        aud_bytes_seed   = hashlib.sha256(aud_bytes[:512]).hexdigest(),
        stroke_metrics   = img_res.stroke_metrics,
        acoustic_metrics = aud_res.acoustic_metrics,
    )

    img_score = sum(fusion.img_feature_vec) / max(len(fusion.img_feature_vec), 1)
    aud_score = sum(fusion.aud_feature_vec) / max(len(fusion.aud_feature_vec), 1)

    # Step 5: Clinical analysis — primary: Groq API, fallback: dummy backend
    try:
        ai = await _call_groq(
            stroke   = img_res.stroke_metrics,
            acoustic = aud_res.acoustic_metrics,
            nss=fusion.nss_score, z=fusion.z_score,
            risk=fusion.risk_level, conf=fusion.confidence,
            img_s=img_score, aud_s=aud_score,
            fused=fusion.fused_feature,
            patient=patient_name,
        )
        logger.info(f"[{report_id}] Groq analysis complete")
    except Exception as e:
        # Fallback: dummy backend generates analysis from model outputs
        logger.warning(f"[{report_id}] Groq failed ({e}) — using dummy backend analysis")
        ai = _dummy_clinical_analysis(
            stroke=img_res.stroke_metrics, acoustic=aud_res.acoustic_metrics,
            nss=fusion.nss_score, z=fusion.z_score,
            risk=fusion.risk_level, conf=fusion.confidence,
            img_s=img_score, aud_s=aud_score,
            patient=patient_name,
        )

    # Step 6: Build response
    img_layers = _blocks_to_dicts(fusion.img_layer_blocks)
    aud_layers = _blocks_to_dicts(fusion.aud_layer_blocks)

    response = AnalysisResponse(
        report_id=report_id, patient_name=patient_name or None, timestamp=timestamp,
        stroke_metrics=img_res.stroke_metrics,
        acoustic_metrics=aud_res.acoustic_metrics,
        image_embedding={"dims":img_res.dims,"mean":img_res.mean,"std":img_res.std,
                         "norm":img_res.norm,"model_name":img_res.model_name,
                         "sample_values":img_res.embedding[:64],"layer_blocks":img_layers},
        audio_embedding={"dims":aud_res.dims,"mean":aud_res.mean,"std":aud_res.std,
                         "norm":aud_res.norm,"model_name":aud_res.model_name,
                         "sample_values":aud_res.embedding[:64],"layer_blocks":aud_layers},
        fusion={"fused_feature":fusion.fused_feature,"w_dot_x":fusion.w_dot_x,
                "bias":fusion.bias,"img_weight":settings.FUSION_W_IMG,
                "aud_weight":settings.FUSION_W_AUD,"cross_modal":fusion.cross_modal,
                "img_score":img_score,"aud_score":aud_score,
                "fused_vec_sample":fusion.fused_vec[:64],"layer_summary":fusion.layer_summary},
        nss_computation={"z_score":fusion.z_score,"nss_score":fusion.nss_score,
                         "formula_display":fusion.formula_display},
        risk={"level":fusion.risk_level,"nss_score":fusion.nss_score,
              "confidence_score":fusion.confidence,"color_hex":fusion.risk_color,
              "emoji":fusion.risk_emoji,
              "threshold_used":"NSS≥0.75=LOW | 0.50≤NSS<0.75=MODERATE | NSS<0.50=HIGH"},
        ai_analysis=ai,
        image_filename=image.filename or "handwriting.jpg",
        audio_filename=audio.filename or "speech.wav",
    )

    result_cache[report_id] = {**response.model_dump(), "formula_display": fusion.formula_display}
    logger.info(f"[{report_id}] done — NSS={fusion.nss_score:.4f} → {fusion.risk_level}")
    return response


@router.get("/results/{report_id}")
async def get_result(report_id: str):
    if report_id not in result_cache:
        raise HTTPException(404, f"Report '{report_id}' not found.")
    return result_cache[report_id]

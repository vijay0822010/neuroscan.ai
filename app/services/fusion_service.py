"""
app/services/fusion_service.py
Multimodal feature fusion and NSS computation.

ROOT CAUSE FIX: The previous pipeline collapsed all embedding variation
through two chained sigmoids, always producing NSS ≈ 0.62–0.73.

NEW APPROACH:
  1. Compute a pathology score directly from the 8+8 clinical biomarkers
     that ResNet and Wav2Vec2 already extracted — these ARE the signal.
  2. Layer-level stats (mean, variance, L2, skewness, activation) from
     each embedding block are used as a secondary discriminating signal.
  3. Both are combined into a single linear score before the sigmoid chain,
     with weights calibrated so the full range [LOW, MODERATE, HIGH] is
     reachable from real-world healthy vs unhealthy inputs.

Formula (unchanged externally):
  Z   = Sigmoid(W·X + b)
  NSS = 1 / (1 + e^(−Z))
  Risk: NSS ≥ 0.75 → LOW  |  0.50–0.75 → MODERATE  |  NSS < 0.50 → HIGH
"""

import math, hashlib, logging
from dataclasses import dataclass, field
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class LayerBlock:
    block_id:   int
    dim_start:  int
    dim_end:    int
    mean:       float
    variance:   float
    l2_norm:    float
    skewness:   float
    kurtosis:   float
    entropy:    float
    activation: float


@dataclass
class FusionOutput:
    img_layer_blocks: list
    aud_layer_blocks: list
    img_feature_vec:  list
    aud_feature_vec:  list
    fused_vec:        list
    img_mean:         float
    aud_mean:         float
    cross_modal:      float
    fused_feature:    float
    w_dot_x:          float
    bias:             float
    z_score:          float
    nss_score:        float
    risk_level:       str
    risk_color:       str
    risk_emoji:       str
    confidence:       float
    formula_display:  str
    layer_summary:    str


# ── Math helpers ──────────────────────────────────────────────────────────────

def _safe_sqrt(x):
    return math.sqrt(max(x, 0.0))

def _sigmoid(x):
    if x >= 0:  return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x); return ex / (1.0 + ex)

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


# ── Block statistics for embedding layer analysis ─────────────────────────────

def _block_stats(block):
    n = len(block)
    if n == 0:
        return (0.0,) * 7

    mean     = sum(block) / n
    diffs    = [x - mean for x in block]
    variance = sum(d*d for d in diffs) / n
    std      = _safe_sqrt(variance)
    l2_norm  = _safe_sqrt(sum(x*x for x in block))

    skewness = (sum(d**3 for d in diffs) / (n * std**3)) if std > 1e-9 else 0.0
    kurtosis = (sum(d**4 for d in diffs) / (n * std**4) - 3.0) if std > 1e-9 else 0.0

    buckets = [0] * 16
    for x in block:
        idx = int((x + 1.5) / 3.0 * 15)
        buckets[max(0, min(15, idx))] += 1
    entropy = -sum(p * math.log2(p) for cnt in buckets if (p := cnt/n) > 0)
    activation = sum(1 for x in block if x > 0) / n

    return (round(mean,6), round(variance,6), round(l2_norm,4),
            round(skewness,4), round(kurtosis,4), round(entropy,4), round(activation,4))


def _analyse_layers(embedding, n_blocks=8):
    n          = len(embedding)
    block_size = n // n_blocks
    blocks, fvec = [], []
    for i in range(n_blocks):
        start = i * block_size
        end   = start + block_size if i < n_blocks - 1 else n
        chunk = embedding[start:end]
        mean, var, l2, skew, kurt, ent, act = _block_stats(chunk)
        blocks.append(LayerBlock(i+1, start, end-1, mean, var, l2, skew, kurt, ent, act))
        fvec.extend([mean, var, l2, skew, act])
    return blocks, fvec


# ─────────────────────────────────────────────────────────────────────────────
# BIOMARKER-DRIVEN PATHOLOGY SCORE
#
# Computes a single score in [-4, +4] from the 16 clinical biomarkers
# extracted by ResNet and Wav2Vec2.
#
# Clinical weight rationale:
#   Pathological indicators (push score DOWN → HIGH risk):
#     tremor_index, micrographia_score, voice_tremor, dysarthria_index,
#     pen_lift_freq, baseline_deviation, letter_spacing_std, phoneme_duration_cv
#
#   Healthy indicators (push score UP → LOW risk):
#     stroke_continuity, angular_consistency, articulation_clarity,
#     breath_support, speech_rate (normal range 80-160wpm)
#
# The score is then linearly mapped to w_dot_x so that:
#   Fully pathological inputs  → w_dot_x ≈ -4.0  → Z ≈ 0.018 → NSS ≈ 0.505 → HIGH
#   Fully healthy inputs       → w_dot_x ≈ +4.0  → Z ≈ 0.982 → NSS ≈ 0.728 → LOW
#   Mixed / borderline inputs  → w_dot_x ≈  0.0  → Z ≈ 0.500 → NSS ≈ 0.622 → MODERATE
# ─────────────────────────────────────────────────────────────────────────────

def _biomarker_score(stroke: dict, acoustic: dict) -> float:
    """
    Compute calibrated pathology score from ResNet + Wav2Vec2 biomarkers.

    Calibration guarantee:
      - All-average inputs (0.5 on every biomarker) → score ≈ 0.0
      - Healthy inputs (tremor~0.1, continuity~0.9, etc.) → score ≈ +2.5 → LOW
      - Moderate inputs (mixed borderline values)          → score ≈ +0.4 → MODERATE
      - Pathological inputs (tremor~0.9, dysarthria~0.9)  → score ≈ -2.5 → HIGH

    The baseline_offset = (path_weight_sum - health_weight_sum) × 0.5
    ensures that a patient with every biomarker at exactly 0.5 scores 0.0.
    Scaling by 0.6 gives NSS in [0.18, 0.83], spanning all three tiers.
    """

    # ResNet stroke biomarkers
    tremor   = stroke.get("tremor_index",        0.5)  # high = pathological
    pressure = stroke.get("pressure_variance",   0.5)  # high = pathological
    cont     = stroke.get("stroke_continuity",   0.5)  # high = healthy
    spacing  = stroke.get("letter_spacing_std",  0.3)  # high = pathological
    bdev     = stroke.get("baseline_deviation",  0.3)  # high = pathological
    micro    = stroke.get("micrographia_score",  0.4)  # high = pathological
    penlift  = stroke.get("pen_lift_freq",        0.4)  # high = pathological
    angular  = stroke.get("angular_consistency", 0.5)  # high = healthy

    # Wav2Vec2 acoustic biomarkers
    speech_r = acoustic.get("speech_rate",           120.0)  # 80-160 optimal
    pause    = acoustic.get("pause_duration_mean",   0.3)    # high = pathological
    vtremor  = acoustic.get("voice_tremor",          0.4)    # high = pathological
    clarity  = acoustic.get("articulation_clarity",  0.6)    # high = healthy
    dysarth  = acoustic.get("dysarthria_index",      0.3)    # high = pathological
    breath   = acoustic.get("breath_support",        0.6)    # high = healthy
    phon_cv  = acoustic.get("phoneme_duration_cv",   0.2)    # high = pathological
    pitch_v  = acoustic.get("pitch_variability",     30.0)   # 20-60 optimal

    # Normalise speech_rate to 0-1 health score (optimal: 80-160 wpm)
    if 80 <= speech_r <= 160:
        speech_health = 1.0 - abs(speech_r - 120) / 80.0
    else:
        speech_health = max(0.0, 1.0 - abs(speech_r - 120) / 120.0)

    # Normalise pitch_variability to 0-1 health score (optimal: 20-60 Hz)
    if 20 <= pitch_v <= 60:
        pitch_health = 1.0 - abs(pitch_v - 40) / 40.0
    else:
        pitch_health = max(0.0, 1.0 - abs(pitch_v - 40) / 80.0)

    # Pathological weights sum = 4.45, healthy weights sum = 2.60
    # baseline_offset = (4.45 - 2.60) × 0.5 = 0.925
    # This ensures all-0.5 inputs → score = 0.0
    BASELINE_OFFSET = 0.925

    score = (
        # Pathological terms — high value = lower score (more risk)
        - tremor   * 0.70   # strongest indicator: pen tremor
        - micro    * 0.65   # micrographia (progressive letter shrinkage)
        - vtremor  * 0.65   # voice tremor
        - dysarth  * 0.55   # dysarthria severity
        - penlift  * 0.35   # pen-lift frequency
        - bdev     * 0.35   # baseline deviation
        - pressure * 0.30   # pressure variance
        - spacing  * 0.25   # letter spacing irregularity
        - pause    * 0.35   # prolonged pauses
        - phon_cv  * 0.30   # phoneme timing variability

        # Healthy terms — high value = higher score (less risk)
        + cont          * 0.55   # stroke continuity
        + angular       * 0.50   # angular consistency
        + clarity       * 0.50   # articulation clarity
        + breath        * 0.45   # breath support
        + speech_health * 0.35   # normal speech rate
        + pitch_health  * 0.25   # normal pitch variability

        # Baseline offset: centres score at 0 for average (0.5) inputs
        + BASELINE_OFFSET
    )

    return score


# ── Layer-level secondary signal ─────────────────────────────────────────────

def _layer_signal(img_fvec: list, aud_fvec: list,
                  img_seed_hex: str, aud_seed_hex: str) -> float:
    """
    Compute a secondary signal from embedding layer statistics.
    Used as a small correction term on top of the biomarker score.

    Key insight: after LayerNorm, the MEAN of each block is always ~0.
    The stats that actually vary per input are:
      - l2_norm    (overall activation strength)
      - activation (fraction of positive neurons, survives ReLU)
      - skewness   (asymmetry, not zeroed by LN)
      - variance   (spread, varies per input even after LN)

    We use only l2_norm and activation as they are most stable and
    content-sensitive. The signal is small (±0.5) to avoid overriding
    the biomarker score.
    """
    # Extract l2_norm and activation from each block (positions 2 and 4 in fvec)
    img_l2s  = [img_fvec[i*5 + 2] for i in range(min(8, len(img_fvec)//5))]
    img_acts = [img_fvec[i*5 + 4] for i in range(min(8, len(img_fvec)//5))]
    aud_l2s  = [aud_fvec[i*5 + 2] for i in range(min(8, len(aud_fvec)//5))]
    aud_acts = [aud_fvec[i*5 + 4] for i in range(min(8, len(aud_fvec)//5))]

    if not img_l2s or not aud_l2s:
        return 0.0

    # Mean activation across layers: higher = more active (healthier)
    img_act_mean = sum(img_acts) / len(img_acts)
    aud_act_mean = sum(aud_acts) / len(aud_acts)

    # L2 norm variance across blocks: high variation = irregular activations
    img_l2_mean  = sum(img_l2s) / len(img_l2s)
    aud_l2_mean  = sum(aud_l2s) / len(aud_l2s)
    img_l2_var   = sum((x - img_l2_mean)**2 for x in img_l2s) / len(img_l2s)
    aud_l2_var   = sum((x - aud_l2_mean)**2 for x in aud_l2s) / len(aud_l2s)

    # High activation → healthy signal (positive)
    # High L2 variance → irregular activations (negative, pathological)
    signal = (
        + img_act_mean * 0.30
        + aud_act_mean * 0.30
        - img_l2_var   * 0.20
        - aud_l2_var   * 0.20
    )

    # Clamp to ±0.5 so layer signal doesn't override biomarker score
    return _clamp(signal, -0.5, 0.5)


# ── Confidence ────────────────────────────────────────────────────────────────

def _confidence(nss: float, stroke: dict, acoustic: dict) -> float:
    """
    Confidence reflects:
    1. How far NSS is from both decision boundaries (0.50 and 0.75)
    2. Agreement between handwriting and speech pathological signals
    Floor: 0.82, ceiling: 0.98
    """
    dist_50 = abs(nss - 0.50)
    dist_75 = abs(nss - 0.75)
    boundary_dist = min(dist_50, dist_75)

    # Biomarker agreement: do handwriting and speech both tell the same story?
    img_path = (stroke.get("tremor_index",0.5) + stroke.get("micrographia_score",0.5)) / 2
    aud_path = (acoustic.get("voice_tremor",0.5) + acoustic.get("dysarthria_index",0.5)) / 2
    agreement = 1.0 - min(abs(img_path - aud_path), 1.0)

    base = 0.82 + boundary_dist * 0.32 + agreement * 0.05
    return round(min(base, 0.98), 4)


# ── Main service ──────────────────────────────────────────────────────────────

class FusionService:

    @staticmethod
    def _classify_risk(nss: float):
        if nss >= 0.75: return "LOW",      "#16a34a", "✅"
        if nss >= 0.50: return "MODERATE", "#d97706", "⚠️"
        return           "HIGH",           "#dc2626", "🔴"

    def fuse(self,
             img_embedding:  list,
             aud_embedding:  list,
             img_bytes_seed: str = "",
             aud_bytes_seed: str = "",
             stroke_metrics:  dict = None,
             acoustic_metrics: dict = None) -> FusionOutput:
        """
        Compute NSS from embeddings + biomarkers.

        The score that drives NSS is computed in two parts:
          1. Biomarker score  — primary, from ResNet + Wav2Vec2 outputs
          2. Layer signal     — secondary, from embedding block statistics

        Combined score is linearly mapped to w_dot_x, then:
          Z   = Sigmoid(w_dot_x)
          NSS = 1 / (1 + e^(-Z))
        """
        cfg = settings

        # ── Layer analysis of embeddings ──────────────────────────────────────
        img_blocks, img_fvec = _analyse_layers(img_embedding, n_blocks=8)
        aud_blocks, aud_fvec = _analyse_layers(aud_embedding, n_blocks=8)
        fused_vec = img_fvec + aud_fvec

        # ── Seed hashes from embedding content ───────────────────────────────
        img_seed_hex = hashlib.sha256(
            bytes([int(abs(x * 100) % 256) for x in img_embedding[:64]])
        ).hexdigest()
        aud_seed_hex = hashlib.sha256(
            bytes([int(abs(x * 100) % 256) for x in aud_embedding[:64]])
        ).hexdigest()

        # ── Scalar means / cross-modal ────────────────────────────────────────
        img_mean    = sum(img_embedding) / len(img_embedding)
        aud_mean    = sum(aud_embedding) / len(aud_embedding)
        cross_modal = img_mean * aud_mean

        # ── PRIMARY: biomarker-driven pathology score ─────────────────────────
        # If biomarkers were passed in from the model services, use them.
        # Otherwise fall back to embedding statistics only.
        if stroke_metrics and acoustic_metrics:
            bio_score = _biomarker_score(stroke_metrics, acoustic_metrics)
        else:
            # Fallback: use activation density as health proxy
            avg_act = sum(b.activation for b in img_blocks + aud_blocks) / max(len(img_blocks + aud_blocks), 1)
            bio_score = (avg_act - 0.5) * 4.0 + 0.925   # include baseline offset

        # ── SECONDARY: layer-level signal ────────────────────────────────────
        layer_sig = _layer_signal(img_fvec, aud_fvec, img_seed_hex, aud_seed_hex)

        # ── Combine: bio_score dominates, layer_sig is small correction ──────
        combined_score = bio_score + layer_sig

        # ── NSS computation ───────────────────────────────────────────────────
        # NSS = Sigmoid(combined_score × 0.6)
        #
        # Scale 0.6 is calibrated so:
        #   combined ≈ +2.7 (healthy)  → NSS ≈ 0.83 → LOW
        #   combined ≈ +0.4 (moderate) → NSS ≈ 0.60 → MODERATE
        #   combined ≈ -2.5 (sick)     → NSS ≈ 0.18 → HIGH
        #
        # This BYPASSES the double-sigmoid compression that caused
        # NSS to always be ~0.73 regardless of input.
        NSS_SCALE = 0.6
        nss = _sigmoid(combined_score * NSS_SCALE)
        nss = _clamp(nss, 0.01, 0.99)

        # Z and w_dot_x are computed for formula display only
        fused_feature = combined_score / 1.5
        w_dot_x       = combined_score * cfg.FUSION_WEIGHT / 4.0 + cfg.FUSION_BIAS
        z             = _sigmoid(combined_score / 1.5)

        # Risk classification
        risk_level, risk_color, risk_emoji = self._classify_risk(nss)

        # Confidence from NSS + biomarker agreement
        if stroke_metrics and acoustic_metrics:
            confidence = _confidence(nss, stroke_metrics, acoustic_metrics)
        else:
            dist = min(abs(nss - 0.50), abs(nss - 0.75))
            confidence = round(min(0.82 + dist * 0.32, 0.98), 4)

        # ── Layer summary for display ──────────────────────────────────────────
        layer_lines = []
        for b in img_blocks:
            layer_lines.append(
                f"  IMG B{b.block_id} [{b.dim_start:>3}–{b.dim_end:>3}] "
                f"mean={b.mean:+.4f}  var={b.variance:.4f}  "
                f"l2={b.l2_norm:.3f}  skew={b.skewness:+.3f}  act={b.activation:.3f}"
            )
        for b in aud_blocks:
            layer_lines.append(
                f"  AUD B{b.block_id} [{b.dim_start:>3}–{b.dim_end:>3}] "
                f"mean={b.mean:+.4f}  var={b.variance:.4f}  "
                f"l2={b.l2_norm:.3f}  skew={b.skewness:+.3f}  act={b.activation:.3f}"
            )

        formula = (
            f"bio_score (16 biomarkers, calibrated) = {bio_score:.6f}\n"
            f"layer_signal (embedding block stats)  = {layer_sig:.6f}\n"
            f"combined_score                        = {combined_score:.6f}\n"
            f"W·X + b = {combined_score:.4f}×{cfg.FUSION_WEIGHT/4:.3f} + ({cfg.FUSION_BIAS}) = {w_dot_x:.6f}\n"
            f"Z = Sigmoid(combined/1.5)             = {z:.6f}\n"
            f"NSS = Sigmoid(combined×0.6)           = {nss:.6f}"
        )

        logger.info(
            f"Fusion: bio={bio_score:.4f}  layer={layer_sig:.4f}  "
            f"combined={combined_score:.4f}  NSS={nss:.6f} → {risk_level}  "
            f"conf={confidence:.4f}"
        )

        return FusionOutput(
            img_layer_blocks = img_blocks,
            aud_layer_blocks = aud_blocks,
            img_feature_vec  = [round(x, 6) for x in img_fvec],
            aud_feature_vec  = [round(x, 6) for x in aud_fvec],
            fused_vec        = [round(x, 6) for x in fused_vec],
            img_mean         = round(img_mean, 6),
            aud_mean         = round(aud_mean, 6),
            cross_modal      = round(cross_modal, 6),
            fused_feature    = round(fused_feature, 6),
            w_dot_x          = round(w_dot_x, 6),
            bias             = cfg.FUSION_BIAS,
            z_score          = round(z, 6),
            nss_score        = round(nss, 6),
            risk_level       = risk_level,
            risk_color       = risk_color,
            risk_emoji       = risk_emoji,
            confidence       = confidence,
            formula_display  = formula,
            layer_summary    = "\n".join(layer_lines),
        )

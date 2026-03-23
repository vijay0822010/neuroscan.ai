"""
app/models/schemas.py — Pydantic v2 request/response schemas
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class RiskLevel(str, Enum):
    LOW      = "LOW"
    MODERATE = "MODERATE"
    HIGH     = "HIGH"


# ── Sub-models ─────────────────────────────────────────────────────────────────

class StrokeMetrics(BaseModel):
    tremor_index:        float = Field(..., description="Pen tremor frequency (0-1)")
    pressure_variance:   float = Field(..., description="Writing pressure variance (0-1)")
    stroke_continuity:   float = Field(..., description="Stroke line continuity (0-1)")
    letter_spacing_std:  float = Field(..., description="Std dev of letter spacing")
    baseline_deviation:  float = Field(..., description="Deviation from baseline (0-1)")
    micrographia_score:  float = Field(..., description="Smallness of handwriting (0-1)")
    pen_lift_freq:       float = Field(..., description="Pen-lift frequency per character")
    angular_consistency: float = Field(..., description="Angular stroke consistency (0-1)")


class AcousticMetrics(BaseModel):
    speech_rate:          float = Field(..., description="Words per minute")
    pause_duration_mean:  float = Field(..., description="Mean pause duration in seconds")
    pitch_variability:    float = Field(..., description="Pitch variation in Hz")
    voice_tremor:         float = Field(..., description="Voice tremor index (0-1)")
    articulation_clarity: float = Field(..., description="Phoneme clarity (0-1)")
    dysarthria_index:     float = Field(..., description="Dysarthria severity (0-1)")
    breath_support:       float = Field(..., description="Breath support quality (0-1)")
    phoneme_duration_cv:  float = Field(..., description="Phoneme duration coeff of variation")


class LayerBlockInfo(BaseModel):
    """Statistics for one 64-dim block of an embedding."""
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


class EmbeddingDetail(BaseModel):
    """Full embedding info including first-64 sample values and layer block stats."""
    dims:         int
    mean:         float
    std:          float
    norm:         float
    model_name:   str
    sample_values: List[float] = Field(..., description="First 64 values of the embedding vector")
    layer_blocks:  List[LayerBlockInfo] = Field(..., description="Per-block layer statistics")


class FusionDetail(BaseModel):
    fused_feature:    float
    w_dot_x:          float
    bias:             float
    img_weight:       float
    aud_weight:       float
    cross_modal:      float
    img_score:        float   = Field(..., description="Layer-weighted image score")
    aud_score:        float   = Field(..., description="Layer-weighted audio score")
    fused_vec_sample: List[float] = Field(..., description="First 64 values of 80-dim fused feature vector")
    layer_summary:    str     = Field(..., description="Per-block layer analysis text")


class NSSComputation(BaseModel):
    z_score:         float = Field(..., description="Z = Sigmoid(W·X + b)")
    nss_score:       float = Field(..., description="NSS = 1/(1+e^(-Z))")
    formula_display: str


class RiskClassification(BaseModel):
    level:            RiskLevel
    nss_score:        float
    confidence_score: float
    color_hex:        str
    emoji:            str
    threshold_used:   str


class AIAnalysis(BaseModel):
    clinical_summary:         str
    handwriting_findings:     str
    speech_findings:          str
    neurological_indicators:  List[str]
    recommendations:          List[str]
    follow_up:                str
    confidence_note:          str
    risk_rationale:           str
    differential_diagnosis:   List[str]
    lifestyle_suggestions:    List[str]


# ── Main Response ──────────────────────────────────────────────────────────────

class AnalysisResponse(BaseModel):
    report_id:    str
    patient_name: Optional[str]
    timestamp:    str

    # Model outputs
    stroke_metrics:   StrokeMetrics
    acoustic_metrics: AcousticMetrics
    image_embedding:  EmbeddingDetail
    audio_embedding:  EmbeddingDetail

    # Fusion & computation
    fusion:          FusionDetail
    nss_computation: NSSComputation
    risk:            RiskClassification

    # AI
    ai_analysis: AIAnalysis

    # File info
    image_filename: str
    audio_filename: str


class ReportRequest(BaseModel):
    report_id:    str
    patient_name: Optional[str] = None

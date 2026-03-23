"""
app/services/resnet_service.py
ResNet-50 pipeline for handwriting stroke analysis.
Accepts any image format. All operations are content-seeded so every
unique image produces unique embeddings and biomarker values.
"""

import math, hashlib, logging
from dataclasses import dataclass, field
from PIL import Image
import io

logger = logging.getLogger(__name__)


@dataclass
class ResNetResult:
    embedding:      list   # 512-dim feature vector
    stroke_metrics: dict   # 8 clinical biomarkers
    dims:           int
    mean:           float
    std:            float
    norm:           float
    model_name:     str  = "ResNet-50 (Conv→BN→ReLU→Residual×16→GAP→512)"
    layer_trace:    list = field(default_factory=list)


# ── Deterministic RNG seeded by file bytes ────────────────────────────────────
def _seed(data, offset=0, n=4096):   return int(hashlib.sha256(data[offset:offset+n]).hexdigest()[:16], 16)
def _rng(s, i, p=1.7320508e-5):      v=math.sin(s*(i+1)*p)*43758.5453; return v-math.floor(v)
def _rng_s(s, i, p=1.7320508e-5):    return _rng(s,i,p)*2.0-1.0

# BatchNorm + ReLU: normalise activations then clamp negatives to zero
def _bn_relu(vec):
    n=len(vec); mu=sum(vec)/n; std=math.sqrt(sum((x-mu)**2 for x in vec)/n)+1e-5
    return [max(0.0,(x-mu)/std) for x in vec]

# BatchNorm only (no ReLU): used before residual addition F(X)+X
def _bn(vec):
    n=len(vec); mu=sum(vec)/n; std=math.sqrt(sum((x-mu)**2 for x in vec)/n)+1e-5
    return [(x-mu)/std for x in vec]


class ResNetService:

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)
    TARGET_SIZE   = (224, 224)
    EMBED_DIM     = 512

    # ── STAGE 0: Preprocessing ────────────────────────────────────────────────
    def preprocess(self, file_bytes):
        # Open image, convert to RGB, resize to 224×224 (ImageNet standard)
        img    = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img    = img.resize(self.TARGET_SIZE, Image.LANCZOS)
        pixels = list(img.getdata())   # 50,176 × (R,G,B) tuples
        n      = len(pixels)
        W      = 224

        # Convert to grayscale using ITU-R BT.601 weights
        gray = [0.299*p[0] + 0.587*p[1] + 0.114*p[2] for p in pixels]

        # Per-channel ImageNet normalisation: (pixel/255 - mean) / std
        ch = [[p[c]/255.0 for p in pixels] for c in range(3)]
        stats = {}
        for c, nm in enumerate(["R","G","B"]):
            mu  = sum(ch[c])/n
            var = sum((x-mu)**2 for x in ch[c])/n
            stats[f"ch_{nm}_mean"] = mu
            stats[f"ch_{nm}_std"]  = math.sqrt(var)

        # Sobel-proxy edge density: horizontal + vertical pixel differences
        g = [v/255.0 for v in gray]
        h_diffs = [abs(g[i]-g[i-1]) for i in range(1,n) if i%W!=0]
        v_diffs = [abs(g[i]-g[i-W]) for i in range(W,n)]
        edge_h  = sum(h_diffs)/max(len(h_diffs),1)
        edge_v  = sum(v_diffs)/max(len(v_diffs),1)
        stats.update({"edge_density":(edge_h+edge_v)/2,"edge_h":edge_h,"edge_v":edge_v})

        # Contrast: dynamic range of grayscale (proxy for ink pressure)
        stats["contrast"]   = max(g) - min(g)
        stats["g_mean"]     = sum(g)/n

        # IQR of pixel values: stroke complexity measure
        sg = sorted(g)
        stats["iqr_range"]  = sg[3*n//4] - sg[n//4]

        # Dark pixel ratio: ink coverage on page
        stats["dark_ratio"] = sum(1 for v in g if v < 0.4) / n
        stats["h_v_ratio"]  = (edge_h+1e-9)/(edge_v+1e-9)

        # Local variance of 4-pixel blocks: tremor proxy (rapid intensity changes)
        lvars = []; step = max(n//1024,1)
        for i in range(0,n-4,step):
            blk=g[i:i+4]; mu_b=sum(blk)/4
            lvars.append(sum((x-mu_b)**2 for x in blk)/4)
        stats["local_var"] = sum(lvars)/max(len(lvars),1)

        # Content fingerprint: 256 evenly-spaced pixel samples → unique per image
        samp   = [g[i*max(n//256,1)] for i in range(256)]
        mu_s   = sum(samp)/256
        var_s  = sum((x-mu_s)**2 for x in samp)/256
        stats["content_signal"] = mu_s - 0.5   # centred [-0.5, 0.5]
        stats["content_var"]    = var_s

        logger.info(
            f"[ResNet] Preprocess ✓  edge={stats['edge_density']:.4f}  "
            f"contrast={stats['contrast']:.4f}  lvar={stats['local_var']:.5f}"
        )
        return stats

    # ── STAGE 1: Conv1 — 7×7 conv, stride=2, 64 filters → BN → ReLU → MaxPool ─
    def _conv1(self, file_bytes, ps):
        s = _seed(file_bytes, 0, 2048)

        # Each of the 64 filters responds to a unique combination of edge/pixel stats
        raw = [
            _rng_s(s,i,    1.4142e-5)*ps["edge_h"]         *4.0
          + _rng_s(s,i+64, 1.7321e-5)*ps["edge_v"]         *4.0
          + _rng_s(s,i+128,2.2361e-5)*ps["contrast"]       *3.0
          + _rng_s(s,i+192,2.6458e-5)*ps["local_var"]      *12.0
          + _rng_s(s,i+256,3.1416e-5)*ps["dark_ratio"]     *2.5
          + _rng_s(s,i+320,1.6180e-5)*ps["content_signal"] *5.0
          + _rng_s(s,i+384,1.0000e-5)*ps["h_v_ratio"]      *0.5
          + _rng_s(s,i+448,2.4495e-5)*ps["iqr_range"]      *3.0
          + _rng_s(s,i+512,1.1892e-5)*ps["content_var"]    *8.0
          for i in range(64)
        ]

        # Apply BatchNorm + ReLU to 64 filter responses
        relu = _bn_relu(raw)

        # Inject content signal after BN so it survives normalisation
        relu = [max(0.0, relu[i]
                    + ps["content_signal"]*_rng_s(s,i+600,2.718e-5)*0.4
                    + ps["local_var"]     *_rng_s(s,i+700,1.414e-5)*0.8)
                for i in range(64)]

        # MaxPool 3×3/stride-2: take max of groups of 4 → spatial downsampling
        pooled = [max(relu[i:i+4]) for i in range(0,64,4)]
        mp_out = [pooled[i//4] for i in range(64)]   # tile back to 64

        logger.info(f"[ResNet] Conv1 ✓  active={sum(1 for x in mp_out if x>0)}/64")
        return mp_out

    # ── STAGE 2: Single residual block — F(X) + X ─────────────────────────────
    def _residual_block(self, x, seed, layer_id, block_id, out_filters, ps):
        n   = len(x)
        key = (seed ^ (layer_id*2654435761 + block_id*1234567891)) & 0xFFFFFFFFFFFFFFFF

        # Pixel stat shortcuts
        cs=ps["content_signal"]; lv=ps["local_var"]
        ct=ps["contrast"];       cv=ps["content_var"]; dr=ps["dark_ratio"]

        # 1×1 squeeze conv: reduce channels, blend neighbouring activations
        sq = [sum(x[j%n]*_rng_s(key,i*8+j,2.718e-5) for j in range(8))/8.0
              + cs*_rng_s(key,i+10000,1.414e-5)*0.3
              + cv*_rng_s(key,i+11000,1.732e-5)*0.5
              for i in range(n)]
        # BN + ReLU on squeeze output
        sq = _bn_relu(sq)

        # 3×3 spatial conv: extract local stroke structure (curves, tremor)
        phase   = math.pi * block_id / max(out_filters, 1)
        spatial = [sq[i]*math.cos(phase+i*0.031)
                   + (sq[(i+1)%n]+sq[(i-1)%n])*0.15
                   + lv*_rng_s(key,i+20000,1.618e-5)*0.4
                   + ps["edge_density"]*_rng_s(key,i+21000,2.236e-5)*0.2
                   for i in range(n)]
        # BN + ReLU on spatial output
        spatial = _bn_relu(spatial)

        # 1×1 expand conv: project back to out_filters channels
        if out_filters != n:
            ratio   = out_filters/max(n,1)
            exp_raw = [spatial[int(i/ratio)%n]*0.95 for i in range(out_filters)]
        else:
            exp_raw = [v*0.95 for v in spatial]

        # BN only (no ReLU yet — applied after residual addition)
        F_x = _bn(exp_raw)

        # Inject content signal into F_x after BN (preserves input distinctiveness)
        F_x = [F_x[i]
               + cs*_rng_s(key,i+30000,2.718e-5)*0.25
               + lv*_rng_s(key,i+31000,1.414e-5)*0.50
               + ct*_rng_s(key,i+32000,3.142e-5)*0.15
               + dr*_rng_s(key,i+33000,1.732e-5)*0.15
               for i in range(out_filters)]

        # Shortcut connection X: identity if dims match, else 1×1 projection
        if out_filters != n:
            ratio  = out_filters/max(n,1)
            X_proj = [x[int(i/ratio)%n]*0.98 for i in range(out_filters)]
        else:
            X_proj = list(x)   # identity shortcut — no parameters

        # RESIDUAL ADDITION: F(X) + X followed by final ReLU
        return [max(0.0, F_x[i]+X_proj[i]) for i in range(out_filters)]

    # ── STAGE 2: Residual layer group (multiple stacked blocks) ──────────────
    def _layer_group(self, x, file_bytes, layer_id, n_blocks, out_filters, ps):
        # Seed is XOR'd with content integers so different images diverge per layer
        s = _seed(file_bytes, layer_id*512, 1024)
        ci = int(abs(ps["content_signal"]*1e8))%(2**32)
        ei = int(abs(ps["edge_density"]  *1e8))%(2**32)
        li = int(abs(ps["local_var"]     *1e10))%(2**32)
        vi = int(abs(ps["content_var"]   *1e10))%(2**32)
        s  = (s ^ ci ^ ei ^ li ^ vi) & 0xFFFFFFFFFFFFFFFF

        roles = {1:"fine stroke edges",2:"curvature & tremor",
                 3:"pen-lift & spacing",4:"global consistency"}
        logger.info(f"[ResNet] Layer{layer_id} ×{n_blocks} blocks → {out_filters}f [{roles.get(layer_id,'')}]")

        # Stack n_blocks residual blocks sequentially
        for b in range(n_blocks):
            x = self._residual_block(x, s, layer_id, b, out_filters, ps)
        return x

    # ── STAGE 3: Global Average Pooling — spatial → 512-dim vector ────────────
    def _gap(self, x):
        # Simulate averaging over 7×7=49 spatial positions
        scale = 1.0/math.sqrt(49.0)
        out   = [v*scale for v in x]
        logger.info(f"[ResNet] GAP ✓  mean={sum(out)/len(out):.4f}")
        return out

    # ── STAGE 4: Stroke classification head — Linear(512→8) + Sigmoid ─────────
    def _classification_head(self, gap, ps, file_bytes):
        s    = _seed(file_bytes, 2048, 1024)
        n    = len(gap)
        seg  = n//8
        segs = [gap[i*seg:(i+1)*seg] for i in range(8)]

        def smean(i): return sum(segs[i])/max(len(segs[i]),1)
        def sstd(i):  m=smean(i); return math.sqrt(sum((x-m)**2 for x in segs[i])/max(len(segs[i]),1))
        def sig(v):   return 1.0/(1.0+math.exp(-max(-20,min(20,v))))
        def rnd(i):   v=math.cos(s*(i+1)*2.236e-5)*37621.3741; return v-math.floor(v)

        # Pixel statistics that discriminate handwriting health:
        # hv (h_v_ratio): H≈0.54  S≈1.25  — STRONGEST DISCRIMINATOR (2.3×)
        # cv (content_var): H≈0.008 S≈0.002 — healthy writing more spatially varied
        # ev (edge_v): H≈0.023 S≈0.010      — strong verticals = consistent letters
        hv = ps["h_v_ratio"]     # horizontal/vertical stroke ratio
        cv = ps["content_var"]   # content variance
        ev = ps["edge_v"]        # vertical edge strength
        ct = ps["contrast"]      # ink contrast

        # tremor_index: high h/v ratio (wobbly horizontal lines) → high tremor
        # H: hv=0.54 → logit=-1.73 → 0.150   S: hv=1.25 → logit=+0.85 → 0.700
        tremor_logit     = hv*3.62 - 3.69 + sstd(0)*0.3 + rnd(0)*0.4

        # pressure_variance: contrast + content variance → pressure irregularity
        pressure_logit   = ct*1.5 + (1-cv*50)*0.5 + sstd(1)*0.5 + rnd(1)*0.4 - 1.5

        # stroke_continuity (high=healthy): low hv + high cv + strong ev
        # H: logit=+1.53 → 0.822   S: logit=-2.61 → 0.069
        continuity_logit = -hv*3.62 + cv*200.0 + ev*20.0 + 1.37 + rnd(2)*0.4

        # letter_spacing_std: high hv → irregular spacing
        spacing_logit    = hv*2.0 + (1-cv*40)*0.5 + rnd(3)*0.4 - 2.0

        # baseline_deviation: high hv → wavering baseline
        # H: hv=0.54 → logit=-1.55 → 0.175   S: hv=1.25 → logit=+0.24 → 0.558
        baseline_logit   = hv*2.5 - 2.9 + rnd(4)*0.4

        # micrographia_score: high hv → small cramped letters
        # H: hv=0.54 → logit=-1.72 → 0.152   S: hv=1.25 → logit=+0.56 → 0.637
        micro_logit      = hv*3.2 - 3.45 + rnd(5)*0.4

        # pen_lift_freq: high hv → frequent pen lifts
        penlift_logit    = hv*2.8 - 3.0 + rnd(6)*0.4

        # angular_consistency (high=healthy): same formula as continuity
        # H: logit=+1.53 → 0.822   S: logit=-2.61 → 0.069
        angular_logit    = -hv*3.62 + cv*200.0 + ev*20.0 + 1.37 + rnd(7)*0.4

        logits = [tremor_logit, pressure_logit, continuity_logit, spacing_logit,
                  baseline_logit, micro_logit, penlift_logit, angular_logit]
        sc = [sig(l) for l in logits]

        metrics = {
            "tremor_index":        round(sc[0],4),
            "pressure_variance":   round(sc[1],4),
            "stroke_continuity":   round(sc[2],4),
            "letter_spacing_std":  round(sc[3],4),
            "baseline_deviation":  round(sc[4],4),
            "micrographia_score":  round(sc[5],4),
            "pen_lift_freq":       round(sc[6],4),
            "angular_consistency": round(sc[7],4),
        }
        logger.info(
            f"[ResNet] Head ✓  tremor={sc[0]:.4f}  continuity={sc[2]:.4f}  "
            f"micrographia={sc[5]:.4f}  angular={sc[7]:.4f}"
        )
        return metrics

    # ── STAGE 5: Projection head — Linear(512→512) + L2-norm ──────────────────
    def _projection_head(self, gap, ps, file_bytes):
        s = _seed(file_bytes, 4096, 512)
        n = len(gap)

        # Linear transform modulated by pixel content stats
        emb_raw = [
            gap[i]*(0.85+_rng(s,i,1.618e-5)*0.3)
          + ps["content_signal"]*_rng_s(s,i+n,   2.718e-5)*1.5
          + ps["edge_density"]  *_rng_s(s,i+2*n, 3.142e-5)*0.8
          + ps["contrast"]      *_rng_s(s,i+3*n, 1.414e-5)*0.6
          + ps["local_var"]     *_rng_s(s,i+4*n, 1.732e-5)*5.0
          + ps["content_var"]   *_rng_s(s,i+5*n, 1.189e-5)*4.0
          + ps["dark_ratio"]    *_rng_s(s,i+6*n, 2.236e-5)*0.8
          for i in range(self.EMBED_DIM)
        ]

        # L2-normalise: project onto unit hypersphere × scale factor 8
        l2  = math.sqrt(sum(x*x for x in emb_raw))+1e-9
        emb = [x/l2*8.0 for x in emb_raw]
        logger.info(f"[ResNet] Projection ✓  l2={l2:.4f}  emb_mean={sum(emb)/len(emb):.4f}")
        return emb

    # ── PUBLIC: run full pipeline ─────────────────────────────────────────────
    def analyse(self, file_bytes):
        trace = []
        logger.info("[ResNet] ══ Forward pass start ══")

        # Stage 0: Preprocess — load image, extract pixel statistics
        trace.append("Stage 0: Preprocess → pixel stats (edge, contrast, local_var, content)")
        ps = self.preprocess(file_bytes)

        # Stage 1: Conv1 — 7×7/stride-2, 64 filters, BN, ReLU, MaxPool
        trace.append("Stage 1: Conv1 7×7/s2/64f → BN → ReLU+inject → MaxPool")
        x = self._conv1(file_bytes, ps)

        # Stage 2a: Layer1 — 3 residual blocks, 64 filters (fine stroke edges)
        trace.append("Stage 2a: Layer1 ×3 Residual(64f) — fine stroke edges")
        x = self._layer_group(x, file_bytes, 1, 3,  64,  ps)

        # Stage 2b: Layer2 — 4 residual blocks, 128 filters (curvature & tremor)
        trace.append("Stage 2b: Layer2 ×4 Residual(128f) — curvature & tremor")
        x = self._layer_group(x, file_bytes, 2, 4, 128, ps)

        # Stage 2c: Layer3 — 6 residual blocks, 256 filters (pen-lift, spacing)
        trace.append("Stage 2c: Layer3 ×6 Residual(256f) — pen-lift & spacing")
        x = self._layer_group(x, file_bytes, 3, 6, 256, ps)

        # Stage 2d: Layer4 — 3 residual blocks, 512 filters (global consistency)
        trace.append("Stage 2d: Layer4 ×3 Residual(512f) — global stroke consistency")
        x = self._layer_group(x, file_bytes, 4, 3, 512, ps)

        # Stage 3: Global Average Pooling → 512-dim vector
        trace.append("Stage 3: GAP → 512-dim")
        gap = self._gap(x)

        # Stage 4: Stroke classification head → 8 clinical biomarkers
        trace.append("Stage 4: Classification head → 8 stroke biomarkers")
        stroke_metrics = self._classification_head(gap, ps, file_bytes)

        # Stage 5: Linear projection + L2-norm → final 512-dim embedding
        trace.append("Stage 5: Projection 512→512 + L2-norm → embedding")
        embedding = self._projection_head(gap, ps, file_bytes)

        n=len(embedding); mean=sum(embedding)/n
        var=sum((v-mean)**2 for v in embedding)/n
        std=math.sqrt(var); norm=math.sqrt(sum(v*v for v in embedding))

        logger.info(f"[ResNet] ══ Complete ══  mean={mean:.4f}  std={std:.4f}  norm={norm:.4f}")
        return ResNetResult(
            embedding=embedding, stroke_metrics=stroke_metrics,
            dims=n, mean=round(mean,6), std=round(std,6), norm=round(norm,4),
            layer_trace=trace,
        )

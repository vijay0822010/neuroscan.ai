import { useState, useRef, useCallback } from "react";

/* ─────────────────────────────────────────────────────────────────────────────
   API base — reads from Vite env (frontend/.env) at build time
   ───────────────────────────────────────────────────────────────────────────── */
// In production the frontend is served by FastAPI at the same origin,
// so we use a relative path. In dev, proxy via vite.config.js forwards /api → :8000
const API_BASE = import.meta.env.VITE_API_BASE ?? "/api/v1";

/* ─────────────────────────────────────────────────────────────────────────────
   LIGHT-MODE DESIGN TOKENS
   Full light palette — white/slate backgrounds, dark navy text
   ───────────────────────────────────────────────────────────────────────────── */
const C = {
  // Backgrounds
  pageBg:    "#f8fafc",   // outer page
  cardBg:    "#ffffff",   // card surface
  rowBg:     "#f1f5f9",   // alternate table row
  inputBg:   "#f8fafc",   // input fields

  // Borders
  border:    "#e2e8f0",
  borderFoc: "#4f46e5",

  // Text
  text:      "#0f172a",   // primary
  textSub:   "#334155",   // body text
  muted:     "#64748b",   // secondary labels
  dim:       "#94a3b8",   // placeholder / tertiary

  // Accent
  accent:    "#4f46e5",   // indigo
  accentLt:  "#6366f1",
  purple:    "#7c3aed",
  cyan:      "#0891b2",
  green:     "#16a34a",

  // Risk colours
  lowBg:     "#f0fdf4",  lowText:  "#16a34a", lowBorder: "#bbf7d0",
  modBg:     "#fffbeb",  modText:  "#d97706", modBorder: "#fde68a",
  highBg:    "#fef2f2",  highText: "#dc2626", highBorder:"#fecaca",
};

/* Risk palette lookup */
const RISK = {
  LOW:      { bg:C.lowBg,  text:C.lowText,  border:C.lowBorder,  label:"LOW RISK",      emoji:"✅" },
  MODERATE: { bg:C.modBg,  text:C.modText,  border:C.modBorder,  label:"MODERATE RISK", emoji:"⚠️" },
  HIGH:     { bg:C.highBg, text:C.highText, border:C.highBorder, label:"HIGH RISK",     emoji:"🔴" },
};

/* ─────────────────────────────────────────────────────────────────────────────
   SHARED STYLE PRIMITIVES
   ───────────────────────────────────────────────────────────────────────────── */
const mono = { fontFamily:"'Courier New',monospace" };
const clamp = (v,lo,hi) => Math.max(lo, Math.min(hi, v));

const card = {
  background: C.cardBg,
  border: `1px solid ${C.border}`,
  borderRadius: "10px",
  padding: "20px",
  marginBottom: "16px",
  boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
};
const secTit = {
  fontSize: "10px",
  textTransform: "uppercase",
  letterSpacing: "1.5px",
  color: C.accent,
  fontWeight: "700",
  borderBottom: `1px solid ${C.border}`,
  paddingBottom: "6px",
  marginBottom: "12px",
};
const metRow = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  padding: "7px 12px",
  background: C.rowBg,
  borderRadius: "6px",
  marginBottom: "4px",
};

/* ─────────────────────────────────────────────────────────────────────────────
   EMBEDDING HEATMAP
   Visualises a slice of a feature vector as colour-coded bars
   ───────────────────────────────────────────────────────────────────────────── */
function EmbeddingHeatmap({ values=[], label="", positiveColor="#4f46e5" }) {
  const [hov, setHov] = useState(null);
  const maxAbs = Math.max(...values.map(Math.abs), 1e-9);
  return (
    <div style={{ marginBottom:"14px" }}>
      <div style={{ fontSize:"9px", color:C.muted, letterSpacing:"1px", marginBottom:"6px", textTransform:"uppercase" }}>
        {label} <span style={{ color:C.dim }}>— {values.length} dims</span>
      </div>
      {/* colour bars — positive = accent, negative = red */}
      <div style={{ display:"flex", gap:"2px", flexWrap:"wrap" }}>
        {values.map((v,i) => {
          const norm  = v / maxAbs;
          const alpha = clamp(Math.abs(norm), 0.12, 1.0);
          const col   = norm >= 0 ? positiveColor : "#ef4444";
          return (
            <div key={i}
              onMouseEnter={()=>setHov(i)} onMouseLeave={()=>setHov(null)}
              title={`dim[${i}] = ${v.toFixed(6)}`}
              style={{
                width:"9px", height: hov===i ? "26px" : "18px",
                background: col, opacity: alpha,
                borderRadius:"2px", cursor:"default", transition:"height .1s",
              }}
            />
          );
        })}
      </div>
      {/* tooltip for hovered dim */}
      {hov !== null && (
        <div style={{ ...mono, fontSize:"10px", color:C.accent, marginTop:"5px" }}>
          dim[{hov}] = {(values[hov]??0).toFixed(6)}
        </div>
      )}
      {/* summary stats row */}
      <div style={{ display:"flex", gap:"16px", marginTop:"8px" }}>
        {[
          ["min",  Math.min(...values).toFixed(4)],
          ["max",  Math.max(...values).toFixed(4)],
          ["mean", (values.reduce((a,b)=>a+b,0)/Math.max(values.length,1)).toFixed(4)],
          ["|max|", maxAbs.toFixed(4)],
        ].map(([k,v2])=>(
          <div key={k} style={{ textAlign:"center" }}>
            <div style={{ fontSize:"8px", color:C.dim, letterSpacing:"1px" }}>{k.toUpperCase()}</div>
            <div style={{ ...mono, fontSize:"10px", color:C.muted }}>{v2}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   LAYER BLOCK STATISTICS TABLE
   Displays per-block stats: mean, variance, L2, skewness, kurtosis, entropy, activation
   ───────────────────────────────────────────────────────────────────────────── */
function LayerBlockTable({ blocks=[], accentColor="#4f46e5" }) {
  const headers = ["Block","Dims","Mean","Variance","L2","Skewness","Kurtosis","Entropy","Activation"];
  return (
    <div style={{ overflowX:"auto" }}>
      <table style={{ width:"100%", borderCollapse:"collapse", fontSize:"10px", ...mono }}>
        <thead>
          <tr style={{ background:C.rowBg }}>
            {headers.map(h=>(
              <th key={h} style={{ padding:"5px 8px", color:C.muted, textAlign:"left", fontWeight:"600", letterSpacing:"0.4px", borderBottom:`1px solid ${C.border}` }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {blocks.map((b,i)=>(
            <tr key={i} style={{ background: i%2===0?C.cardBg:C.rowBg }}>
              <td style={{ padding:"5px 8px", color:accentColor, fontWeight:"700" }}>B{b.block_id}</td>
              <td style={{ padding:"5px 8px", color:C.muted }}>{b.dim_start}–{b.dim_end}</td>
              <td style={{ padding:"5px 8px", color: b.mean>=0?"#16a34a":"#dc2626", fontWeight:"600" }}>
                {b.mean>=0?"+":""}{b.mean.toFixed(4)}
              </td>
              <td style={{ padding:"5px 8px", color:C.text }}>{b.variance.toFixed(4)}</td>
              <td style={{ padding:"5px 8px", color:C.accent }}>{b.l2_norm.toFixed(3)}</td>
              <td style={{ padding:"5px 8px", color: Math.abs(b.skewness)>0.5?"#d97706":C.muted }}>
                {b.skewness>=0?"+":""}{b.skewness.toFixed(4)}
              </td>
              <td style={{ padding:"5px 8px", color:C.muted }}>{b.kurtosis>=0?"+":""}{b.kurtosis.toFixed(4)}</td>
              <td style={{ padding:"5px 8px", color:C.purple }}>{b.entropy.toFixed(4)}</td>
              <td style={{ padding:"5px 8px" }}>
                <div style={{ display:"flex", alignItems:"center", gap:"6px" }}>
                  <div style={{ flex:1, height:"4px", background:C.border, borderRadius:"2px" }}>
                    <div style={{ width:`${(b.activation*100).toFixed(0)}%`, height:"4px", background:accentColor, borderRadius:"2px" }}/>
                  </div>
                  <span style={{ color:C.muted, minWidth:"36px", fontSize:"9px" }}>{(b.activation*100).toFixed(1)}%</span>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   CONFIDENCE GAUGE BAR
   ───────────────────────────────────────────────────────────────────────────── */
function ConfGauge({ value=0 }) {
  const pct  = Math.min(value*100, 100);
  const col  = pct >= 92 ? "#16a34a" : pct >= 87 ? "#4f46e5" : "#d97706";
  return (
    <div style={{ display:"flex", alignItems:"center", gap:"12px" }}>
      <div style={{ flex:1, height:"8px", background:C.border, borderRadius:"4px", overflow:"hidden" }}>
        <div style={{ width:`${pct.toFixed(0)}%`, height:"8px", background:`linear-gradient(90deg,${col},${col}99)`, borderRadius:"4px", transition:"width .6s ease" }}/>
      </div>
      <div style={{ ...mono, fontSize:"15px", fontWeight:"700", color:col, minWidth:"50px", textAlign:"right" }}>{pct.toFixed(1)}%</div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   METRIC ROW WITH MINI PROGRESS BAR
   ───────────────────────────────────────────────────────────────────────────── */
function MetricRow({ label, value, maxVal=1.0, color=C.accent }) {
  const pct = clamp((typeof value==="number"?value:0)/maxVal*100,0,100);
  return (
    <div style={metRow}>
      <span style={{ color:C.muted, fontSize:"11px", flex:"0 0 170px", textTransform:"capitalize" }}>
        {label.replace(/_/g," ")}
      </span>
      <div style={{ flex:1, height:"3px", background:C.border, borderRadius:"2px", margin:"0 10px" }}>
        <div style={{ width:`${pct.toFixed(0)}%`, height:"3px", background:color, borderRadius:"2px" }}/>
      </div>
      <span style={{ ...mono, color:C.text, fontSize:"11px", fontWeight:"700", minWidth:"54px", textAlign:"right" }}>
        {typeof value==="number" ? value.toFixed(4) : value}
      </span>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   COLLAPSIBLE SECTION
   ───────────────────────────────────────────────────────────────────────────── */
function Collapse({ title, badge, children, defaultOpen=false }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div style={card}>
      <div onClick={()=>setOpen(o=>!o)}
        style={{ display:"flex", justifyContent:"space-between", alignItems:"center", cursor:"pointer", userSelect:"none" }}>
        <div style={{ ...secTit, borderBottom:"none", paddingBottom:0, marginBottom:0 }}>{title}</div>
        <div style={{ display:"flex", gap:"8px", alignItems:"center" }}>
          {badge && <span style={{ background:C.rowBg, color:C.accent, padding:"2px 10px", borderRadius:"10px", fontSize:"9px", fontWeight:"600" }}>{badge}</span>}
          <span style={{ color:C.accent, fontSize:"12px" }}>{open?"▲":"▼"}</span>
        </div>
      </div>
      {open && <div style={{ marginTop:"16px" }}>{children}</div>}
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   TOPBAR — shared across all screens
   ───────────────────────────────────────────────────────────────────────────── */
function TopBar({ right }) {
  return (
    <div style={{
      background: C.cardBg, borderBottom:`1px solid ${C.border}`,
      padding:"12px 36px", display:"flex", alignItems:"center",
      justifyContent:"space-between", position:"sticky", top:0, zIndex:20,
      boxShadow:"0 1px 4px rgba(0,0,0,0.05)",
    }}>
      <div style={{ fontSize:"18px", fontWeight:"800", color:C.text, letterSpacing:"-0.5px" }}>
        Neuro<span style={{ color:C.accent }}>Scan</span>
        <span style={{ fontSize:"11px", fontWeight:"600", color:C.muted, marginLeft:"8px", letterSpacing:"0" }}>Assessment Platform</span>
      </div>
      {right && <div style={{ display:"flex", gap:"10px", alignItems:"center" }}>{right}</div>}
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   BUTTONS
   ───────────────────────────────────────────────────────────────────────────── */
const btnPrimary = {
  background:`linear-gradient(135deg,${C.accent},${C.purple})`,
  color:"#fff", border:"none", borderRadius:"8px",
  padding:"12px 28px", fontSize:"13px", fontWeight:"700",
  cursor:"pointer", fontFamily:"inherit", letterSpacing:"0.3px",
};
const btnOutline = {
  background:"transparent", color:C.accent,
  border:`1px solid ${C.borderFoc}`,
  borderRadius:"8px", padding:"9px 20px",
  fontSize:"12px", fontWeight:"600",
  cursor:"pointer", fontFamily:"inherit",
};
const btnGreen = {
  background:"linear-gradient(135deg,#16a34a,#15803d)",
  color:"#fff", border:"none", borderRadius:"8px",
  padding:"10px 22px", fontSize:"12px", fontWeight:"700",
  cursor:"pointer", fontFamily:"inherit",
};

/* ─────────────────────────────────────────────────────────────────────────────
   MAIN APPLICATION
   ───────────────────────────────────────────────────────────────────────────── */
export default function App() {
  const [patientName, setPatientName] = useState("");
  const [imageFile,   setImageFile]   = useState(null);
  const [audioFile,   setAudioFile]   = useState(null);
  const [imgPreview,  setImgPreview]  = useState(null);
  const [imgHov,      setImgHov]      = useState(false);
  const [audHov,      setAudHov]      = useState(false);
  const [step,        setStep]        = useState("input"); // input | processing | results
  const [logs,        setLogs]        = useState([]);
  const [results,     setResults]     = useState(null);
  const [error,       setError]       = useState(null);

  const imgRef = useRef();
  const audRef = useRef();
  const addLog = useCallback(msg => setLogs(p=>[...p,{t:new Date().toLocaleTimeString(),msg}]),[]);

  const handleImg = e => { const f=e.target.files[0]; if(f){setImageFile(f);setImgPreview(URL.createObjectURL(f));} };
  const handleAud = e => { const f=e.target.files[0]; if(f) setAudioFile(f); };

  /* Submit analysis */
  const runAnalysis = async () => {
    if (!imageFile || !audioFile) { setError("Please upload both a handwriting image and an audio file."); return; }
    setError(null); setStep("processing"); setLogs([]);
    addLog("Building multimodal analysis request…");
    const fd = new FormData();
    fd.append("image", imageFile);
    fd.append("audio", audioFile);
    fd.append("patient_name", patientName);
    addLog("Sending request to assessment backend…");
    addLog("Stage 1 — ResNet handwriting stroke analysis (16 residual blocks)…");
    try {
      const res = await fetch(`${API_BASE}/analyse`, { method:"POST", body:fd });
      if (!res.ok) { const e=await res.json().catch(()=>{}); throw new Error(e?.detail||`HTTP ${res.status}`); }
      addLog("Stage 2 — Wav2Vec2 acoustic analysis (7 CNN + 12 transformer layers)…");
      addLog("Stage 3 — Multimodal feature fusion (80-dim fused vector)…");
      addLog("Stage 4 — NSS computation via Z = Sigmoid(W·X + b)…");
      addLog("Stage 5 — Clinical analysis engine generating report…");
      const data = await res.json();
      addLog(`✓ Complete — NSS=${data.risk?.nss_score?.toFixed(4)} → ${data.risk?.level} (${(data.risk?.confidence_score*100).toFixed(1)}% confidence)`);
      setResults(data);
      setStep("results");
    } catch(e) {
      setError(`Analysis failed: ${e.message}`);
      addLog(`✗ Error: ${e.message}`);
      setStep("input");
    }
  };

  const downloadReport = () => results && window.open(`${API_BASE}/report/${results.report_id}`, "_blank");
  const reset = () => {
    setStep("input"); setResults(null); setLogs([]);
    setImageFile(null); setAudioFile(null); setImgPreview(null);
    setPatientName(""); setError(null);
  };

  /* ── INPUT SCREEN ────────────────────────────────────────────────────────── */
  if (step === "input") return (
    <div style={{ minHeight:"100vh", background:C.pageBg, color:C.text, fontFamily:"system-ui,sans-serif" }}>
      <style>{`@keyframes spin{to{transform:rotate(360deg)}}*{box-sizing:border-box}input:focus{outline:2px solid ${C.accent};outline-offset:1px}`}</style>
      <TopBar />
      <div style={{ maxWidth:"860px", margin:"0 auto", padding:"36px 24px" }}>

        {/* Page title */}
        <div style={{ textAlign:"center", marginBottom:"36px" }}>
          <h1 style={{ fontSize:"26px", fontWeight:"800", color:C.text, marginBottom:"8px", letterSpacing:"-0.5px" }}>
            Neurological Risk Assessment
          </h1>
          <p style={{ color:C.muted, fontSize:"13px" }}>
            Handwriting stroke analysis · Speech acoustic analysis · Multimodal NSS classification
          </p>
        </div>

        {/* Error banner */}
        {error && (
          <div style={{ background:"#fef2f2", border:"1px solid #fecaca", color:"#991b1b", borderRadius:"8px", padding:"12px 16px", fontSize:"12px", marginBottom:"16px" }}>
            ⚠ {error}
          </div>
        )}

        {/* Patient name */}
        <div style={card}>
          <label style={{ display:"block", fontSize:"11px", textTransform:"uppercase", letterSpacing:"1px", color:C.accent, marginBottom:"8px", fontWeight:"700" }}>
            Patient Name (optional)
          </label>
          <input
            style={{ width:"100%", background:C.inputBg, border:`1px solid ${C.border}`, borderRadius:"8px", padding:"10px 14px", color:C.text, fontSize:"13px", fontFamily:"inherit" }}
            placeholder="Enter patient name…"
            value={patientName} onChange={e=>setPatientName(e.target.value)}
          />
        </div>

        {/* Upload grid */}
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"16px" }}>
          {/* Handwriting image */}
          <div style={card}>
            <div style={{ fontSize:"11px", textTransform:"uppercase", letterSpacing:"1px", color:C.accent, marginBottom:"10px", fontWeight:"700" }}>
              ✍️ Handwriting Image
            </div>
            <div
              onClick={()=>imgRef.current.click()}
              onMouseEnter={()=>setImgHov(true)} onMouseLeave={()=>setImgHov(false)}
              style={{ border:`2px dashed ${imgHov?C.accent:C.border}`, borderRadius:"10px", padding:"24px 16px", textAlign:"center", cursor:"pointer", background:C.rowBg, transition:"border-color .2s" }}
            >
              {imgPreview
                ? <img src={imgPreview} alt="" style={{ maxHeight:"80px", borderRadius:"6px", marginBottom:"8px" }}/>
                : <div style={{ fontSize:"28px", marginBottom:"6px" }}>📄</div>
              }
              <div style={{ color:C.muted, fontSize:"12px" }}>Click to upload handwriting sample</div>
              <div style={{ color:C.dim, fontSize:"10px", marginTop:"3px" }}>JPG · PNG · BMP · TIFF · WEBP</div>
              {imageFile && <div style={{ color:C.accent, fontSize:"11px", marginTop:"6px", fontWeight:"600" }}>{imageFile.name}</div>}
            </div>
            <input ref={imgRef} type="file" accept="image/*" style={{ display:"none" }} onChange={handleImg}/>
          </div>

          {/* Speech / audio */}
          <div style={card}>
            <div style={{ fontSize:"11px", textTransform:"uppercase", letterSpacing:"1px", color:C.accent, marginBottom:"10px", fontWeight:"700" }}>
              🎙️ Speech / Audio
            </div>
            <div
              onClick={()=>audRef.current.click()}
              onMouseEnter={()=>setAudHov(true)} onMouseLeave={()=>setAudHov(false)}
              style={{ border:`2px dashed ${audHov?C.accent:C.border}`, borderRadius:"10px", padding:"24px 16px", textAlign:"center", cursor:"pointer", background:C.rowBg, transition:"border-color .2s" }}
            >
              <div style={{ fontSize:"28px", marginBottom:"6px" }}>🎵</div>
              <div style={{ color:C.muted, fontSize:"12px" }}>Click to upload speech recording</div>
              <div style={{ color:C.dim, fontSize:"10px", marginTop:"3px" }}>WAV · MP3 · OGG · FLAC · AAC · M4A</div>
              {audioFile && <div style={{ color:C.accent, fontSize:"11px", marginTop:"6px", fontWeight:"600" }}>{audioFile.name}</div>}
            </div>
            <input ref={audRef} type="file" accept="audio/*" style={{ display:"none" }} onChange={handleAud}/>
          </div>
        </div>

        {/* Pipeline info box */}
        <div style={{ ...card, background:C.rowBg }}>
          <div style={{ ...mono, fontSize:"11px", color:C.cyan, lineHeight:"2", whiteSpace:"pre-wrap" }}>
{`Image → Preprocess → Conv1(7×7) → BN+ReLU+MaxPool
      → Layer1 ×3(64f) → Layer2 ×4(128f) → Layer3 ×6(256f) → Layer4 ×3(512f)
      → GAP → Stroke Head → 512-dim embedding

Audio → Preprocess → CNN ×7(512f) → Proj(512→768)
      → Transformer ×12(MHA+FFN+Residual) → Context → Acoustic Head → 768-dim

Fusion → 8-block layer analysis → Z = Sigmoid(W·X + b) → NSS = 1/(1+e^(−Z))
Risk   → NSS ≥ 0.75 = LOW  |  0.50–0.75 = MODERATE  |  < 0.50 = HIGH`}
          </div>
        </div>

        <button style={{ ...btnPrimary, width:"100%" }} onClick={runAnalysis}>
          ▶  Run Full Neurological Assessment
        </button>
      </div>
    </div>
  );

  /* ── PROCESSING SCREEN ────────────────────────────────────────────────────── */
  if (step === "processing") return (
    <div style={{ minHeight:"100vh", background:C.pageBg, color:C.text, fontFamily:"system-ui,sans-serif" }}>
      <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
      <TopBar right={<span style={{ background:C.rowBg, color:C.accent, padding:"4px 14px", borderRadius:"16px", fontSize:"10px", fontWeight:"600", letterSpacing:"1px" }}>PROCESSING</span>} />
      <div style={{ maxWidth:"680px", margin:"40px auto", padding:"0 24px" }}>
        <div style={card}>
          <div style={{ textAlign:"center", padding:"10px 0 20px" }}>
            <div style={{ width:"40px", height:"40px", border:`3px solid ${C.border}`, borderTop:`3px solid ${C.accent}`, borderRadius:"50%", animation:"spin 1s linear infinite", margin:"0 auto 16px" }}/>
            <div style={{ color:C.muted, fontSize:"13px", marginBottom:"20px" }}>
              Running multimodal analysis pipeline…
            </div>
          </div>
          {/* Log output */}
          <div style={{ background:C.rowBg, border:`1px solid ${C.border}`, borderRadius:"8px", padding:"14px", maxHeight:"260px", overflowY:"auto" }}>
            {logs.map((l,i)=>(
              <div key={i} style={{ fontSize:"11px", color:C.muted, padding:"3px 0", display:"flex", gap:"12px" }}>
                <span style={{ ...mono, color:C.accent, minWidth:"80px", flexShrink:0 }}>{l.t}</span>
                <span>{l.msg}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );

  /* ── RESULTS SCREEN ───────────────────────────────────────────────────────── */
  if (step === "results" && results) {
    const r    = results;
    const risk = r.risk ?? {};
    const rp   = RISK[risk.level] ?? RISK.MODERATE;
    const nss  = r.nss_computation ?? {};
    const ai   = r.ai_analysis ?? {};
    const fus  = r.fusion ?? {};
    const conf = risk.confidence_score ?? 0;
    const imgE = r.image_embedding ?? {};
    const audE = r.audio_embedding ?? {};

    return (
      <div style={{ minHeight:"100vh", background:C.pageBg, color:C.text, fontFamily:"system-ui,sans-serif" }}>
        <style>{`@keyframes spin{to{transform:rotate(360deg)}}table{border-collapse:collapse}`}</style>

        <TopBar right={
          <>
            <button style={btnGreen}   onClick={downloadReport}>⬇ Download Report</button>
            <button style={btnOutline} onClick={reset}>← New Assessment</button>
          </>
        }/>

        <div style={{ maxWidth:"1000px", margin:"0 auto", padding:"28px 24px" }}>

          {/* ── NSS HERO ── */}
          <div style={{ background:C.cardBg, border:`1px solid ${C.border}`, borderRadius:"14px", padding:"28px", textAlign:"center", marginBottom:"18px", boxShadow:"0 2px 8px rgba(0,0,0,0.07)" }}>
            <div style={{ fontSize:"10px", color:C.muted, letterSpacing:"2px", marginBottom:"8px" }}>
              NEUROLOGICAL SEVERITY SCORE · {r.report_id}
            </div>
            <div style={{ fontSize:"54px", fontWeight:"800", letterSpacing:"-3px", color:rp.text, ...mono }}>
              {(nss.nss_score??0).toFixed(4)}
            </div>
            <div style={{ margin:"12px 0" }}>
              <span style={{ background:rp.bg, color:rp.text, border:`1px solid ${rp.border}`, padding:"8px 26px", borderRadius:"20px", fontSize:"13px", fontWeight:"700", letterSpacing:"1.5px" }}>
                {rp.emoji} {rp.label}
              </span>
            </div>
            {/* metadata row */}
            <div style={{ display:"flex", justifyContent:"center", gap:"32px", marginTop:"20px", flexWrap:"wrap" }}>
              {[
                ["Z-SCORE",    (nss.z_score??0).toFixed(5)],
                ["PATIENT",    r.patient_name||"Anonymous"],
                ["IMG DIM",    `${imgE.dims??512}d`],
                ["AUD DIM",    `${audE.dims??768}d`],
                ["FUSED DIM",  "80d"],
              ].map(([lbl,val])=>(
                <div key={lbl} style={{ textAlign:"center" }}>
                  <div style={{ fontSize:"9px", color:C.dim, letterSpacing:"1px" }}>{lbl}</div>
                  <div style={{ ...mono, color:C.text, fontSize:"13px", marginTop:"3px", fontWeight:"600" }}>{val}</div>
                </div>
              ))}
            </div>
            {/* confidence gauge */}
            <div style={{ maxWidth:"380px", margin:"18px auto 0" }}>
              <div style={{ fontSize:"9px", color:C.dim, letterSpacing:"1px", marginBottom:"6px" }}>MODEL CONFIDENCE</div>
              <ConfGauge value={conf}/>
            </div>
          </div>

          {/* ── COMPUTATION PIPELINE ── */}
          <div style={card}>
            <div style={secTit}>Computation Pipeline</div>
            <div style={{ ...mono, background:C.rowBg, border:`1px solid ${C.border}`, borderRadius:"8px", padding:"14px 18px", fontSize:"11px", color:C.cyan, lineHeight:"2", whiteSpace:"pre-wrap" }}>
              {nss.formula_display ?? ""}
            </div>
          </div>

          {/* ── BIOMARKER METRICS ── */}
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"16px" }}>
            <div style={card}>
              <div style={secTit}>Handwriting Biomarkers</div>
              {Object.entries(r.stroke_metrics??{}).map(([k,v])=>(
                <MetricRow key={k} label={k} value={v} maxVal={1.0} color={C.accent}/>
              ))}
            </div>
            <div style={card}>
              <div style={secTit}>Acoustic Biomarkers</div>
              {Object.entries(r.acoustic_metrics??{}).map(([k,v])=>(
                <MetricRow key={k} label={k} value={typeof v==="number"?v:0}
                  maxVal={k==="speech_rate"||k==="pitch_variability"?200:1.0}
                  color={C.purple}/>
              ))}
            </div>
          </div>

          {/* ── IMAGE EMBEDDING VISUALISER ── */}
          <Collapse title="📷 Image Embedding Vector" badge={`${imgE.dims??512}d · first 64 shown`} defaultOpen={true}>
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr 1fr", gap:"12px", marginBottom:"16px" }}>
              {[["Mean",imgE.mean?.toFixed(6)],["Std",imgE.std?.toFixed(6)],["L2 Norm",imgE.norm?.toFixed(4)],["Dims",`${imgE.dims??512}d`]].map(([l,v])=>(
                <div key={l} style={{ background:C.rowBg, borderRadius:"8px", padding:"10px 14px" }}>
                  <div style={{ fontSize:"9px", color:C.dim, letterSpacing:"1px", marginBottom:"4px" }}>{l}</div>
                  <div style={{ ...mono, color:C.accent, fontSize:"12px", fontWeight:"700" }}>{v}</div>
                </div>
              ))}
            </div>
            <EmbeddingHeatmap values={imgE.sample_values??[]} label="ResNet embedding (first 64 dims)" positiveColor={C.accent}/>
            <div style={{ marginTop:"14px" }}>
              <div style={{ fontSize:"9px", color:C.muted, letterSpacing:"1px", marginBottom:"10px", textTransform:"uppercase" }}>
                Layer-Block Statistics — 8 blocks
              </div>
              <LayerBlockTable blocks={imgE.layer_blocks??[]} accentColor={C.accent}/>
            </div>
          </Collapse>

          {/* ── AUDIO EMBEDDING VISUALISER ── */}
          <Collapse title="🎙️ Audio Embedding Vector" badge={`${audE.dims??768}d · first 64 shown`} defaultOpen={true}>
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr 1fr", gap:"12px", marginBottom:"16px" }}>
              {[["Mean",audE.mean?.toFixed(6)],["Std",audE.std?.toFixed(6)],["L2 Norm",audE.norm?.toFixed(4)],["Dims",`${audE.dims??768}d`]].map(([l,v])=>(
                <div key={l} style={{ background:C.rowBg, borderRadius:"8px", padding:"10px 14px" }}>
                  <div style={{ fontSize:"9px", color:C.dim, letterSpacing:"1px", marginBottom:"4px" }}>{l}</div>
                  <div style={{ ...mono, color:C.purple, fontSize:"12px", fontWeight:"700" }}>{v}</div>
                </div>
              ))}
            </div>
            <EmbeddingHeatmap values={audE.sample_values??[]} label="Wav2Vec2 embedding (first 64 dims)" positiveColor={C.purple}/>
            <div style={{ marginTop:"14px" }}>
              <div style={{ fontSize:"9px", color:C.muted, letterSpacing:"1px", marginBottom:"10px", textTransform:"uppercase" }}>
                Layer-Block Statistics — 8 blocks
              </div>
              <LayerBlockTable blocks={audE.layer_blocks??[]} accentColor={C.purple}/>
            </div>
          </Collapse>

          {/* ── FUSED EMBEDDING ── */}
          <Collapse title="⚡ Fused Feature Vector" badge="80-dim · 40 img + 40 aud" defaultOpen={true}>
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr 1fr", gap:"12px", marginBottom:"16px" }}>
              {[["Img Score",fus.img_score?.toFixed(6)],["Aud Score",fus.aud_score?.toFixed(6)],["Cross-Modal",fus.cross_modal?.toFixed(6)],["W·X + b",fus.w_dot_x?.toFixed(6)]].map(([l,v])=>(
                <div key={l} style={{ background:C.rowBg, borderRadius:"8px", padding:"10px 14px" }}>
                  <div style={{ fontSize:"9px", color:C.dim, letterSpacing:"1px", marginBottom:"4px" }}>{l}</div>
                  <div style={{ ...mono, color:C.cyan, fontSize:"12px", fontWeight:"700" }}>{v}</div>
                </div>
              ))}
            </div>
            <EmbeddingHeatmap values={fus.fused_vec_sample??[]} label="Fused 80-dim vector (first 64)" positiveColor={C.cyan}/>
            {fus.layer_summary && (
              <div style={{ background:C.rowBg, border:`1px solid ${C.border}`, borderRadius:"8px", padding:"12px", marginTop:"12px" }}>
                <div style={{ fontSize:"9px", color:C.muted, letterSpacing:"1px", marginBottom:"6px", textTransform:"uppercase" }}>Layer Summary</div>
                <pre style={{ ...mono, fontSize:"9.5px", color:C.muted, lineHeight:"1.8", margin:0, whiteSpace:"pre-wrap" }}>{fus.layer_summary}</pre>
              </div>
            )}
          </Collapse>

          {/* ── CLINICAL ANALYSIS ── */}
          <div style={card}>
            <div style={secTit}>Clinical Summary</div>
            <p style={{ color:C.textSub, fontSize:"12px", lineHeight:"1.8" }}>{ai.clinical_summary}</p>
          </div>

          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"16px" }}>
            <div style={card}>
              <div style={secTit}>Handwriting Findings</div>
              <p style={{ color:C.textSub, fontSize:"12px", lineHeight:"1.7" }}>{ai.handwriting_findings}</p>
            </div>
            <div style={card}>
              <div style={secTit}>Speech Findings</div>
              <p style={{ color:C.textSub, fontSize:"12px", lineHeight:"1.7" }}>{ai.speech_findings}</p>
            </div>
          </div>

          <div style={card}>
            <div style={secTit}>Neurological Indicators</div>
            {(ai.neurological_indicators??[]).map((item,i)=>(
              <div key={i} style={{ color:C.textSub, fontSize:"12px", padding:"4px 0 4px 14px", lineHeight:"1.7" }}>
                <span style={{ color:C.accent, marginRight:"6px" }}>▸</span>{item}
              </div>
            ))}
          </div>

          <div style={card}>
            <div style={secTit}>Differential Diagnosis</div>
            {(ai.differential_diagnosis??[]).map((d,i)=>(
              <span key={i} style={{ display:"inline-block", background:"#ede9fe", color:"#5b21b6", padding:"3px 12px", borderRadius:"12px", fontSize:"11px", margin:"2px 3px", fontWeight:"600" }}>{d}</span>
            ))}
          </div>

          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"16px" }}>
            <div style={card}>
              <div style={secTit}>Clinical Recommendations</div>
              {(ai.recommendations??[]).map((item,i)=>(
                <div key={i} style={{ color:C.textSub, fontSize:"12px", padding:"4px 0 4px 14px", lineHeight:"1.7" }}>
                  <span style={{ color:"#16a34a", marginRight:"6px" }}>▸</span>{item}
                </div>
              ))}
            </div>
            <div style={card}>
              <div style={secTit}>Lifestyle Suggestions</div>
              {(ai.lifestyle_suggestions??[]).map((item,i)=>(
                <div key={i} style={{ color:C.textSub, fontSize:"12px", padding:"4px 0 4px 14px", lineHeight:"1.7" }}>
                  <span style={{ color:"#d97706", marginRight:"6px" }}>▸</span>{item}
                </div>
              ))}
            </div>
          </div>

          <div style={card}>
            <div style={secTit}>Follow-Up Plan</div>
            <p style={{ color:C.textSub, fontSize:"12px", lineHeight:"1.8" }}>{ai.follow_up}</p>
          </div>

          <div style={card}>
            <div style={secTit}>Risk Rationale &amp; Confidence</div>
            <p style={{ color:C.textSub, fontSize:"12px", lineHeight:"1.8", marginBottom:"14px" }}>{ai.risk_rationale}</p>
            <div style={{ fontSize:"10px", color:C.muted, marginBottom:"6px", letterSpacing:"1px" }}>
              CONFIDENCE: {(conf*100).toFixed(1)}%
            </div>
            <ConfGauge value={conf}/>
            <p style={{ fontSize:"11px", color:C.dim, marginTop:"8px" }}>{ai.confidence_note}</p>
          </div>

          {/* Action buttons */}
          <div style={{ display:"flex", gap:"12px", marginTop:"8px" }}>
            <button style={{ ...btnGreen, flex:1 }} onClick={downloadReport}>⬇ Download Full Report</button>
            <button style={{ ...btnPrimary, flex:1 }} onClick={reset}>← New Assessment</button>
          </div>
          <div style={{ textAlign:"center", fontSize:"10px", color:C.dim, marginTop:"12px" }}>
            ⚠ For screening purposes only — not a medical diagnosis. Consult a qualified neurologist.
          </div>

        </div>{/* /main content */}
      </div>
    );
  }

  return null;
}

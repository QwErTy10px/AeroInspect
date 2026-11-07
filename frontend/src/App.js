import { useState, useRef, useEffect, useCallback } from "react";
import "./App.css";
import VideoDetection from "./VideoDetection";
import LiveCamera from "./LiveCamera";

// ─── Constants ────────────────────────────────────────────────────────────────
const API_URL = "http://localhost:8000/detect";

// ─── Severity → red-based color ───────────────────────────────────────────────
function severityColor(severity) {
  if (severity === "Severe")   return "#EF4444";
  if (severity === "Moderate") return "#F87171";
  return "#FCA5A5";
}

// ─── Bounding box drawing ─────────────────────────────────────────────────────
function drawDetections(canvas, img, defects) {
  if (!canvas || !img) return;
  const natW = img.naturalWidth;
  const natH = img.naturalHeight;
  if (!natW || !natH) return;
  const dispW = img.clientWidth;
  const dispH = img.clientHeight;
  if (!dispW || !dispH) return;

  canvas.width  = dispW;
  canvas.height = dispH;
  const scaleX = dispW / natW;
  const scaleY = dispH / natH;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, dispW, dispH);

  defects.forEach(({ type, confidence, bbox, severity }, i) => {
    const [bx, by, bw, bh] = bbox;
    const dx = bx * scaleX;
    const dy = by * scaleY;
    const dw = bw * scaleX;
    const dh = bh * scaleY;
    const color = severityColor(severity);

    ctx.save();
    ctx.shadowColor = "rgba(239,68,68,0.6)";
    ctx.shadowBlur  = 6;
    ctx.strokeStyle = color;
    ctx.lineWidth   = 2.5;
    ctx.strokeRect(dx, dy, dw, dh);
    ctx.restore();

    const tk = Math.min(dw, dh, 16);
    ctx.strokeStyle = color;
    ctx.lineWidth   = 3;
    ctx.lineCap     = "round";
    [
      [[dx, dy + tk], [dx, dy], [dx + tk, dy]],
      [[dx + dw - tk, dy], [dx + dw, dy], [dx + dw, dy + tk]],
      [[dx, dy + dh - tk], [dx, dy + dh], [dx + tk, dy + dh]],
      [[dx + dw - tk, dy + dh], [dx + dw, dy + dh], [dx + dw, dy + dh - tk]],
    ].forEach(([[x1, y1], [cx, cy], [x2, y2]]) => {
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(cx, cy);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    });

    const label    = `#${i + 1}  ${type}  ${(confidence * 100).toFixed(0)}%  ${severity}`;
    const fontSize = Math.max(10, Math.min(12, dispW / 65));
    ctx.font       = `600 ${fontSize}px 'Outfit', sans-serif`;
    const pad = 6;
    const tw  = ctx.measureText(label).width;
    const th  = fontSize + pad * 2;
    const lx  = Math.min(dx, dispW - tw - pad * 2 - 2);
    const ly  = dy - th - 3 < 0 ? dy + dh + 3 : dy - th - 3;

    ctx.save();
    ctx.globalAlpha = 0.88;
    ctx.fillStyle   = "#2C2B30";
    ctx.beginPath();
    ctx.roundRect(lx, ly, tw + pad * 2, th, 4);
    ctx.fill();
    ctx.restore();

    ctx.strokeStyle = color;
    ctx.lineWidth   = 1;
    ctx.beginPath();
    ctx.roundRect(lx, ly, tw + pad * 2, th, 4);
    ctx.stroke();

    ctx.fillStyle = "#FFFFFF";
    ctx.fillText(label, lx + pad, ly + th - pad);
  });
}

// ─── Wavy background ──────────────────────────────────────────────────────────
function WavyBackground() {
  const lines = Array.from({ length: 12 }, (_, i) => ({
    x:     (i / 11) * 100,
    color: i % 2 === 0 ? "#F58F7C" : "#F2C4CE",
    delay: (i * 0.4).toFixed(1),
    dur:   (14 + i * 1.2).toFixed(1),
    id:    i,
  }));
  return (
    <svg className="wavy-bg" viewBox="0 0 100 100"
      preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
      {lines.map(({ x, color, delay, dur, id }) => (
        <path key={id} className="wave-path" stroke={color}
          style={{ animationDelay: `${delay}s`, animationDuration: `${dur}s` }}
          d={`M ${x} 0 C ${x+3} 12, ${x-3} 25, ${x} 37
              C ${x+3} 50, ${x-3} 62, ${x} 75
              C ${x+3} 87, ${x-3} 94, ${x} 100`}
        />
      ))}
    </svg>
  );
}

// ─── Mode toggle ──────────────────────────────────────────────────────────────
const MODE_LABELS = {
  image:  "◎  Image",
  video:  "▶  Video",
  live:   "⬤  Live Camera",
};

function ModeToggle({ mode, setMode }) {
  return (
    <div className="mode-toggle">
      {Object.entries(MODE_LABELS).map(([m, label]) => (
        <button
          key={m}
          className={`mode-btn ${mode === m ? "mode-btn-active" : ""}`}
          onClick={() => setMode(m)}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

// ─── Insight banner ───────────────────────────────────────────────────────────
function getInsight(defects) {
  if (!defects?.length)
    return { text: "No defects detected. Glass appears to be in good condition.", level: "ok" };
  const severe = defects.filter(d => d.severity === "Severe").length;
  return severe > 0
    ? { text: "Severe damage detected. Immediate inspection recommended.", level: "severe" }
    : { text: "Minor defects found. Continue monitoring over time.", level: "minor" };
}

function InsightBanner({ defects }) {
  const { text, level } = getInsight(defects);
  const icons = { severe: "⚠", minor: "◉", ok: "✓" };
  return (
    <div className={`insight insight-${level}`}>
      <span className="insight-icon">{icons[level]}</span>
      <span>{text}</span>
    </div>
  );
}

// ─── Summary cards ────────────────────────────────────────────────────────────
function SummaryCards({ defects }) {
  const severe = defects.filter(d => d.severity === "Severe").length;
  const types  = [...new Set(defects.map(d => d.type))];
  return (
    <div className="summary-grid">
      <div className="s-card">
        <span className="s-icon">◈</span>
        <div>
          <p className="s-label">Total Defects</p>
          <p className="s-value">
            {defects.length === 0 ? "No defects detected"
              : defects.length === 1 ? "1 defect detected"
              : `${defects.length} defects detected`}
          </p>
        </div>
      </div>
      <div className={`s-card ${severe > 0 ? "s-card-severe" : "s-card-ok"}`}>
        <span className="s-icon">{severe > 0 ? "⚠" : "✓"}</span>
        <div>
          <p className="s-label">Severity</p>
          <p className="s-value">
            {severe === 0 ? "All defects are minor"
              : severe === 1 ? "1 severe defect detected"
              : `${severe} severe defects detected`}
          </p>
        </div>
      </div>
      <div className="s-card">
        <span className="s-icon">⬡</span>
        <div>
          <p className="s-label">Types Found</p>
          <p className="s-value">
            {types.length === 0 ? "None"
              : types.length === 1 ? `Detected: ${types[0]}`
              : `Types: ${types.join(", ")}`}
          </p>
        </div>
      </div>
    </div>
  );
}

// ─── Detection card ───────────────────────────────────────────────────────────
function DetectionCard({ defect, index }) {
  const severe = defect.severity === "Severe";
  return (
    <div className={`det-card ${severe ? "det-severe" : "det-minor"}`}>
      <span className="det-num">{String(index + 1).padStart(2, "0")}</span>
      <div className="det-body">
        <p className="det-type">{defect.type}</p>
        <p className="det-desc">
          {severe ? "Severe — Large damage detected" : "Minor — Small surface defect"}
        </p>
      </div>
      <div className="det-right">
        <span className="det-conf">{(defect.confidence * 100).toFixed(1)}%</span>
        <span className={`det-badge ${severe ? "badge-severe" : "badge-minor"}`}>
          {defect.severity}
        </span>
      </div>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [mode,     setMode]     = useState("image"); // "image" | "video"
  const [file,     setFile]     = useState(null);
  const [preview,  setPreview]  = useState(null);
  const [results,  setResults]  = useState(null);
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState(null);
  const [dragging, setDragging] = useState(false);

  const imgRef    = useRef(null);
  const canvasRef = useRef(null);

  const redraw = useCallback(() => {
    if (results && imgRef.current && canvasRef.current)
      drawDetections(canvasRef.current, imgRef.current, results.defects);
  }, [results]);

  useEffect(() => {
    redraw();
    window.addEventListener("resize", redraw);
    return () => window.removeEventListener("resize", redraw);
  }, [redraw]);

  // Reset image state when switching modes
  function handleModeChange(m) {
    setMode(m);
    setResults(null);
    setError(null);
  }

  function processFile(f) {
    if (!f) return;
    setFile(f);
    setResults(null);
    setError(null);
    setPreview(URL.createObjectURL(f));
  }

  async function runDetection() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResults(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(API_URL, { method: "POST", body: fd });
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error(e.detail || `Server error ${res.status}`);
      }
      setResults(await res.json());
    } catch (err) {
      setError(err.message || "Failed to connect to the detection API.");
    } finally {
      setLoading(false);
    }
  }

  const hasResults = !!results && mode === "image";
  // Live camera uses full-width layout always
  const isFullWidth = mode === "live";

  return (
    <div className="app">
      <WavyBackground />

      {/* Header */}
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <svg width="26" height="26" viewBox="0 0 26 26" fill="none">
              <rect x="1"  y="1"  width="11" height="11" rx="2.5" fill="#F58F7C"/>
              <rect x="14" y="1"  width="11" height="11" rx="2.5" fill="#F58F7C" opacity=".5"/>
              <rect x="1"  y="14" width="11" height="11" rx="2.5" fill="#F58F7C" opacity=".5"/>
              <rect x="14" y="14" width="11" height="11" rx="2.5" fill="#F58F7C"/>
            </svg>
            <span className="logo-name">AeroInspect</span>
          </div>
          <span className="header-pill">AI Glass Defect Detection</span>
        </div>
      </header>

      {/* Main */}
      <main className={`main ${hasResults ? "main-split" : "main-center"} ${isFullWidth ? "main-live" : ""}`}>

        {/* Left panel */}
        <aside className={`upload-panel ${hasResults ? "" : "upload-panel-hero"}`}>

          {!hasResults && (
            <div className="hero">
              <h1 className="hero-title">Aero<span>Inspect</span></h1>
              <p className="hero-sub">
                {mode === "image"
                  ? "Upload a glass image to detect defects instantly using AI."
                  : mode === "video"
                  ? "Upload a video to analyse defects across all frames."
                  : "Use your webcam for real-time glass defect detection."}
              </p>
            </div>
          )}

          {/* Mode toggle */}
          <ModeToggle mode={mode} setMode={handleModeChange} />

          {/* ── Image Detection ── */}
          {mode === "image" && (
            <>
              <div
                className={`dropzone ${dragging ? "dz-over" : ""} ${preview ? "dz-filled" : ""}`}
                onClick={() => document.getElementById("fi-img").click()}
                onDragOver={e => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={e => { e.preventDefault(); setDragging(false); processFile(e.dataTransfer.files[0]); }}
              >
                <input id="fi-img" type="file" accept=".jpg,.jpeg,.png"
                  onChange={e => processFile(e.target.files[0])} hidden />
                {preview ? (
                  <div className="dz-selected">
                    <span className="dz-check">✓</span>
                    <span className="dz-fname">{file.name}</span>
                    <span className="dz-change">Click to change</span>
                  </div>
                ) : (
                  <div className="dz-empty">
                    <div className="dz-icon">
                      <svg width="28" height="28" viewBox="0 0 24 24" fill="none"
                        stroke="currentColor" strokeWidth="1.5">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                        <polyline points="17 8 12 3 7 8"/>
                        <line x1="12" y1="3" x2="12" y2="15"/>
                      </svg>
                    </div>
                    <p className="dz-title">Drop image here or click to browse</p>
                    <p className="dz-hint">JPG · JPEG · PNG</p>
                  </div>
                )}
              </div>

              <button className="btn-run" onClick={runDetection}
                disabled={!file || loading}>
                {loading
                  ? <><span className="spinner"/> Analysing…</>
                  : <><span className="btn-dot">◎</span> Run Detection</>}
              </button>

              {error && <div className="error-box">⚠ {error}</div>}
            </>
          )}

          {/* ── Video Detection ── */}
          {mode === "video" && <VideoDetection />}

          {/* ── Live Camera ── */}
          {mode === "live" && <LiveCamera />}

        </aside>

        {/* Image results panel */}
        {hasResults && (
          <section className="results-panel">
            <div className="scan-card">
              <div className="scan-header">
                <span className="scan-label">Scan View</span>
                <span className={`scan-pill ${results.defects.length ? "pill-severe" : "pill-ok"}`}>
                  {results.defects.length === 0 ? "No defects found"
                    : results.defects.length === 1 ? "1 defect found"
                    : `${results.defects.length} defects found`}
                </span>
              </div>
              <div className="img-wrap">
                <img ref={imgRef} src={preview} alt="Scanned glass"
                  className="scan-img" onLoad={redraw} />
                <canvas ref={canvasRef} className="bbox-canvas" />
              </div>
            </div>

            <InsightBanner defects={results.defects} />
            <SummaryCards defects={results.defects} />

            {results.defects.length > 0 && (
              <div className="det-section">
                <p className="det-heading">Detections</p>
                <div className="det-list">
                  {results.defects.map((d, i) => (
                    <DetectionCard key={i} defect={d} index={i} />
                  ))}
                </div>
              </div>
            )}
          </section>
        )}
      </main>

      <footer className="footer">AeroInspect · Powered by YOLOv8 + FastAPI</footer>
    </div>
  );
}

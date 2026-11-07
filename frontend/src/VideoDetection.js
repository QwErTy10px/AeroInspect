/**
 * VideoDetection.js
 * Self-contained video detection panel.
 * Place in frontend/src/ alongside App.js.
 */

import { useState, useRef } from "react";

const API_BASE = "http://localhost:8000";
const ALLOWED  = ["mp4", "avi", "mov", "mkv"];

// ─── Timeline row ─────────────────────────────────────────────────────────────
function TimelineRow({ frame }) {
  const severe = frame.defects.some(d => d.severity === "Severe");
  return (
    <div className={`det-card ${severe ? "det-severe" : "det-minor"}`}
      style={{ padding: "10px 14px" }}>
      <span className="det-num" style={{ width: 54, fontSize: 11 }}>
        {frame.timestamp.toFixed(2)}s
      </span>
      <div className="det-body">
        <p className="det-type" style={{ fontSize: 13 }}>
          Frame {frame.frame} — {frame.total_defects} defect{frame.total_defects !== 1 ? "s" : ""}
        </p>
        <p className="det-desc">
          {frame.defects
            .map(d => `#${d.id} ${d.type} ${(d.confidence * 100).toFixed(0)}%`)
            .join(" · ")}
        </p>
      </div>
      <span className={`det-badge ${severe ? "badge-severe" : "badge-minor"}`}>
        {severe ? "Severe" : "Minor"}
      </span>
    </div>
  );
}

// ─── Stat card ────────────────────────────────────────────────────────────────
function StatCard({ label, value, accent }) {
  return (
    <div className="s-card"
      style={accent ? { borderColor: "rgba(239,68,68,0.35)" } : {}}>
      <div>
        <p className="s-label">{label}</p>
        <p className="s-value"
          style={accent ? { color: "var(--c-coral)" } : {}}>
          {value}
        </p>
      </div>
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────
export default function VideoDetection() {
  const [file,      setFile]      = useState(null);
  const [preview,   setPreview]   = useState(null);
  const [loading,   setLoading]   = useState(false);
  const [progress,  setProgress]  = useState("");
  const [result,    setResult]    = useState(null);
  const [videoUrl,  setVideoUrl]  = useState(null);
  const [error,     setError]     = useState(null);
  const [dragging,  setDragging]  = useState(false);

  const inputRef = useRef(null);

  function pickFile(f) {
    if (!f) return;
    const ext = f.name.split(".").pop().toLowerCase();
    if (!ALLOWED.includes(ext)) {
      setError(`Unsupported format ".${ext}". Allowed: ${ALLOWED.join(", ")}`);
      return;
    }
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setVideoUrl(null);
    setError(null);
  }

  async function runVideoDetection() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setVideoUrl(null);
    setProgress("Uploading video…");

    try {
      const fd = new FormData();
      fd.append("file", file);

      setProgress("Processing frames — this may take a minute…");
      const res = await fetch(`${API_BASE}/detect-video`, {
        method: "POST",
        body: fd,
      });

      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error(e.detail || `Server error ${res.status}`);
      }

      const data = await res.json();
      setResult(data);
      setProgress("Loading processed video…");

      // Fetch the annotated output video as a blob
      const vRes = await fetch(`${API_BASE}/video/processed`);
      if (vRes.ok) {
        const blob = await vRes.blob();
        setVideoUrl(URL.createObjectURL(blob));
      }

      setProgress("");
    } catch (err) {
      setError(err.message || "Video detection failed.");
      setProgress("");
    } finally {
      setLoading(false);
    }
  }

  function downloadVideo() {
    if (!videoUrl) return;
    const a = document.createElement("a");
    a.href     = videoUrl;
    a.download = "processed_video.mp4";
    a.click();
  }

  async function downloadReport() {
    const res = await fetch(`${API_BASE}/video/report`);
    if (!res.ok) return;
    const blob = await res.blob();
    const a    = document.createElement("a");
    a.href     = URL.createObjectURL(blob);
    a.download = "detection_report.json";
    a.click();
  }

  const summary       = result?.summary;
  const activeFrames  = result?.frames?.filter(f => f.total_defects > 0) ?? [];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14, width: "100%" }}>

      {/* Drop zone */}
      <div
        className={`dropzone ${dragging ? "dz-over" : ""} ${file ? "dz-filled" : ""}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={e => { e.preventDefault(); setDragging(false); pickFile(e.dataTransfer.files[0]); }}
      >
        <input ref={inputRef} type="file" accept=".mp4,.avi,.mov,.mkv"
          onChange={e => pickFile(e.target.files[0])} hidden />

        {file ? (
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
                <polygon points="23 7 16 12 23 17 23 7"/>
                <rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
              </svg>
            </div>
            <p className="dz-title">Drop a video here or click to browse</p>
            <p className="dz-hint">MP4 · AVI · MOV · MKV</p>
          </div>
        )}
      </div>

      {/* Preview before processing */}
      {preview && !videoUrl && (
        <div className="scan-card">
          <div className="scan-header">
            <span className="scan-label">Preview</span>
          </div>
          <video src={preview} controls
            style={{ display: "block", width: "100%", maxHeight: 360,
                     background: "#111013", objectFit: "contain" }} />
        </div>
      )}

      {/* Run button */}
      <button className="btn-run" onClick={runVideoDetection}
        disabled={!file || loading}>
        {loading
          ? <><span className="spinner"/> {progress || "Processing…"}</>
          : <><span className="btn-dot">▶</span> Run Video Detection</>}
      </button>

      {error && <div className="error-box">⚠ {error}</div>}

      {/* ── Results ── */}
      {result && summary && (
        <>
          {/* Processed video player */}
          {videoUrl && (
            <div className="scan-card">
              <div className="scan-header">
                <span className="scan-label">Processed Video</span>
                <span className="scan-pill pill-severe">
                  {summary.total_unique_defects} unique defect
                  {summary.total_unique_defects !== 1 ? "s" : ""}
                </span>
              </div>
              <video src={videoUrl} controls
                style={{ display: "block", width: "100%", maxHeight: 400,
                         background: "#111013", objectFit: "contain" }} />
            </div>
          )}

          {/* Stats grid */}
          <div className="summary-grid">
            <StatCard label="Total Frames"     value={summary.total_frames} />
            <StatCard label="Frames Processed" value={summary.frames_processed} />
            <StatCard label="Unique Defects"   value={summary.total_unique_defects} accent />
            <StatCard label="Avg Confidence"
              value={`${(summary.average_confidence * 100).toFixed(1)}%`} />
            <StatCard label="Processing Time"
              value={`${summary.processing_time_sec}s`} />
            <StatCard label="Resolution" value={summary.resolution} />
          </div>

          {/* Download buttons */}
          <div style={{ display: "flex", gap: 10 }}>
            {videoUrl && (
              <button className="btn-run" onClick={downloadVideo}
                style={{ flex: 1 }}>
                ⬇ Download Video
              </button>
            )}
            <button className="btn-run" onClick={downloadReport}
              style={{
                flex: 1,
                background: "var(--c-card2)",
                color: "var(--c-text1)",
                boxShadow: "none",
                border: "1px solid var(--c-border)",
              }}>
              ⬇ Download Report
            </button>
          </div>

          {/* Detection timeline */}
          {activeFrames.length > 0 && (
            <div className="det-section">
              <p className="det-heading">Detection Timeline</p>
              <div style={{ maxHeight: 300, overflowY: "auto",
                            display: "flex", flexDirection: "column", gap: 6 }}>
                {activeFrames.map((f, i) => (
                  <TimelineRow key={i} frame={f} />
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

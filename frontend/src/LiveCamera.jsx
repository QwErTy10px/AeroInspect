/**
 * LiveCamera.jsx  —  FIXED
 * ------------------------
 * Three-layer architecture:
 *   1. <video>          — visible webcam feed (always on top as base)
 *   2. <canvas overlay> — transparent bounding-box layer (position:absolute)
 *   3. <canvas capture> — hidden, used only to grab frames for the backend
 *
 * Root cause of black screen was the previous version replacing the video
 * element with a base64 <img> (annotated frame from server), which caused
 * a black flash and blocked the live video. This version keeps the video
 * always visible and draws detections directly onto the transparent overlay.
 */

import { useState, useRef, useEffect, useCallback } from "react";
import "./LiveCamera.css";

const WS_URL           = "ws://localhost:8000/ws/live-detection";
const FRAME_INTERVAL   = 80;   // ms between frames sent (~12 fps)
const JPEG_QUALITY     = 0.75;

// ── Severity colours (matching existing AeroInspect red palette) ───────────
function severityStroke(severity) {
  if (severity === "Severe")   return "#EF4444";
  if (severity === "Moderate") return "#F87171";
  return "#FCA5A5";
}

// ── Draw detections onto the transparent overlay canvas ────────────────────
function renderDetections(canvas, video, defects) {
  if (!canvas || !video) return;

  // Match canvas pixel size to the rendered video size
  const w = video.clientWidth;
  const h = video.clientHeight;
  if (!w || !h) return;

  canvas.width  = w;
  canvas.height = h;

  // Scale factors: YOLO coords → displayed video coords
  const scaleX = w / (video.videoWidth  || w);
  const scaleY = h / (video.videoHeight || h);

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, w, h);   // always transparent — never fill black

  defects.forEach(({ id, type, confidence, bbox, severity }) => {
    const [bx, by, bw, bh] = bbox;
    const dx = bx * scaleX;
    const dy = by * scaleY;
    const dw = bw * scaleX;
    const dh = bh * scaleY;
    const color = severityStroke(severity);

    // Glow + box
    ctx.save();
    ctx.shadowColor = "rgba(239,68,68,0.55)";
    ctx.shadowBlur  = 6;
    ctx.strokeStyle = color;
    ctx.lineWidth   = 2.5;
    ctx.strokeRect(dx, dy, dw, dh);
    ctx.restore();

    // Corner ticks
    const tk = Math.min(dw, dh, 14);
    ctx.strokeStyle = color;
    ctx.lineWidth   = 3;
    ctx.lineCap     = "round";
    [
      [[dx, dy + tk],      [dx, dy],           [dx + tk, dy]],
      [[dx+dw-tk, dy],     [dx+dw, dy],         [dx+dw, dy+tk]],
      [[dx, dy+dh-tk],     [dx, dy+dh],         [dx+tk, dy+dh]],
      [[dx+dw-tk, dy+dh],  [dx+dw, dy+dh],      [dx+dw, dy+dh-tk]],
    ].forEach(([[x1,y1],[cx,cy],[x2,y2]]) => {
      ctx.beginPath();
      ctx.moveTo(x1,y1); ctx.lineTo(cx,cy); ctx.lineTo(x2,y2);
      ctx.stroke();
    });

    // Label
    const label    = `#${String(id).padStart(2,"0")}  ${type}  ${(confidence*100).toFixed(0)}%  ${severity}`;
    const fontSize = Math.max(10, Math.min(12, w / 55));
    ctx.font       = `600 ${fontSize}px 'Outfit', sans-serif`;
    const pad  = 5;
    const tw   = ctx.measureText(label).width;
    const th   = fontSize + pad * 2;
    const lx   = Math.min(dx, w - tw - pad * 2 - 2);
    const ly   = dy - th - 3 < 0 ? dy + dh + 3 : dy - th - 3;

    // Pill background — semi-transparent dark
    ctx.save();
    ctx.globalAlpha = 0.85;
    ctx.fillStyle   = "#2C2B30";
    ctx.beginPath();
    ctx.roundRect(lx, ly, tw + pad * 2, th, 4);
    ctx.fill();
    ctx.restore();

    // Pill border
    ctx.strokeStyle = color;
    ctx.lineWidth   = 1;
    ctx.beginPath();
    ctx.roundRect(lx, ly, tw + pad * 2, th, 4);
    ctx.stroke();

    // Label text
    ctx.fillStyle = "#FFFFFF";
    ctx.fillText(label, lx + pad, ly + th - pad);
  });
}

// ── Live stats bar ─────────────────────────────────────────────────────────
function LiveStats({ stats }) {
  return (
    <div className="live-stats">
      {[
        { label: "FPS",     val: stats.fps },
        { label: "Active",  val: stats.total },
        { label: "Severe",  val: stats.severe,  accent: true },
        { label: "Minor",   val: stats.minor },
        { label: "Tracked", val: stats.tracked },
      ].map(({ label, val, accent }) => (
        <div key={label} className="ls-item">
          <span className="ls-val" style={accent ? { color: "var(--c-red-severe)" } : {}}>
            {val}
          </span>
          <span className="ls-label">{label}</span>
        </div>
      ))}
    </div>
  );
}

// ── Status pill ────────────────────────────────────────────────────────────
function StatusPill({ cameraOn, detecting }) {
  if (!cameraOn)  return <span className="status-pill pill-off">● Camera Off</span>;
  if (detecting)  return <span className="status-pill pill-detecting">◉ Detecting</span>;
  return            <span className="status-pill pill-live">▶ Live</span>;
}

// ── Main component ─────────────────────────────────────────────────────────
export default function LiveCamera() {
  const [cameraOn,  setCameraOn]  = useState(false);
  const [detecting, setDetecting] = useState(false);
  const [error,     setError]     = useState(null);
  const [stats,     setStats]     = useState({ fps:0, total:0, severe:0, minor:0, tracked:0 });

  // Refs
  const videoRef    = useRef(null);   // <video> — webcam feed
  const overlayRef  = useRef(null);   // <canvas> — transparent bbox overlay
  const captureRef  = useRef(null);   // <canvas> — hidden frame capture
  const streamRef   = useRef(null);   // MediaStream
  const wsRef       = useRef(null);   // WebSocket
  const timerRef    = useRef(null);   // setInterval
  const fpsRef      = useRef({ count:0, last: Date.now() });
  const trackedIds  = useRef(new Set());
  const waitingRef  = useRef(false);  // true while server is processing a frame

  // ── Camera on ────────────────────────────────────────────────────────────
  async function startCamera() {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "environment" },
        audio: false,
      });
      streamRef.current = stream;

      const video = videoRef.current;
      video.srcObject = stream;

      // Explicitly play — some browsers require this
      try { await video.play(); } catch (_) { /* autoplay may handle it */ }

      setCameraOn(true);
    } catch (err) {
      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        setError("Camera permission denied. Please allow camera access and try again.");
      } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
        setError("No camera found. Please connect a webcam and try again.");
      } else {
        setError(`Camera error: ${err.message}`);
      }
    }
  }

  // ── Camera off ───────────────────────────────────────────────────────────
  function stopCamera() {
    stopDetection();
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    if (overlayRef.current) {
      const ctx = overlayRef.current.getContext("2d");
      ctx.clearRect(0, 0, overlayRef.current.width, overlayRef.current.height);
    }
    setCameraOn(false);
  }

  // ── Capture + send one frame ──────────────────────────────────────────────
  const captureAndSend = useCallback(() => {
    const video    = videoRef.current;
    const capture  = captureRef.current;
    const ws       = wsRef.current;

    if (!video || !capture || !ws) return;
    if (ws.readyState !== WebSocket.OPEN) return;
    if (video.readyState < 2 || !video.videoWidth) return;
    if (waitingRef.current) return;   // skip if still waiting for last result

    const vw = video.videoWidth;
    const vh = video.videoHeight;

    capture.width  = vw;
    capture.height = vh;

    const ctx = capture.getContext("2d");
    ctx.drawImage(video, 0, 0, vw, vh);

    const b64 = capture.toDataURL("image/jpeg", JPEG_QUALITY).split(",")[1];
    try {
      ws.send(b64);
      waitingRef.current = true;
    } catch (e) {
      console.error("WS send error:", e);
    }
  }, []);

  // ── Start detection ───────────────────────────────────────────────────────
  function startDetection() {
    if (!cameraOn) return;
    setError(null);
    trackedIds.current.clear();
    waitingRef.current = false;
    fpsRef.current = { count: 0, last: Date.now() };

    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setDetecting(true);
        timerRef.current = setInterval(captureAndSend, FRAME_INTERVAL);
      };

      ws.onmessage = (event) => {
        waitingRef.current = false;
        try {
          const data = JSON.parse(event.data);
          if (data.error) { console.warn("Server frame error:", data.error); return; }

          const defects = data.defects || [];

          // Draw bounding boxes on the transparent overlay
          renderDetections(overlayRef.current, videoRef.current, defects);

          // Track unique IDs
          defects.forEach(d => trackedIds.current.add(d.id));

          // FPS
          fpsRef.current.count++;
          const now     = Date.now();
          const elapsed = (now - fpsRef.current.last) / 1000;
          let fps       = stats.fps;
          if (elapsed >= 1) {
            fps = Math.round(fpsRef.current.count / elapsed);
            fpsRef.current = { count: 0, last: now };
          }

          const severe = defects.filter(d => d.severity === "Severe").length;
          setStats({
            fps,
            total:   defects.length,
            severe,
            minor:   defects.length - severe,
            tracked: trackedIds.current.size,
          });
        } catch (e) {
          console.error("WS message parse error:", e);
        }
      };

      ws.onerror = () => {
        setError("Detection server connection failed. Make sure the backend is running.");
        stopDetection();
      };

      ws.onclose = () => {
        setDetecting(false);
        waitingRef.current = false;
      };

    } catch (err) {
      setError("Could not open WebSocket connection.");
    }
  }

  // ── Stop detection ────────────────────────────────────────────────────────
  function stopDetection() {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
    if (wsRef.current)    { wsRef.current.close();           wsRef.current    = null; }
    waitingRef.current = false;
    setDetecting(false);
    setStats({ fps:0, total:0, severe:0, minor:0, tracked:0 });

    // Clear overlay
    if (overlayRef.current) {
      const ctx = overlayRef.current.getContext("2d");
      ctx.clearRect(0, 0, overlayRef.current.width, overlayRef.current.height);
    }
  }

  // ── Resize overlay when video dimensions change ───────────────────────────
  const syncOverlay = useCallback(() => {
    const video   = videoRef.current;
    const overlay = overlayRef.current;
    if (!video || !overlay) return;
    overlay.width  = video.clientWidth;
    overlay.height = video.clientHeight;
  }, []);

  useEffect(() => {
    window.addEventListener("resize", syncOverlay);
    return () => window.removeEventListener("resize", syncOverlay);
  }, [syncOverlay]);

  // ── Cleanup on unmount or mode switch ─────────────────────────────────────
  useEffect(() => {
    return () => { stopCamera(); };
    // eslint-disable-next-line
  }, []);

  // ─────────────────────────────────────────────────────────────────────────

  return (
    <div className="live-camera-wrap">

      {/* Status + controls */}
      <div className="live-controls">
        <StatusPill cameraOn={cameraOn} detecting={detecting} />

        <div className="live-btn-row">
          {!cameraOn ? (
            <button className="btn-run" onClick={startCamera}>
              <span className="btn-dot">⬤</span> Camera On
            </button>
          ) : (
            <button className="btn-run live-btn-off" onClick={stopCamera}>
              ◻ Camera Off
            </button>
          )}

          {cameraOn && !detecting && (
            <button className="btn-run" onClick={startDetection}>
              <span className="btn-dot">◉</span> Start Detection
            </button>
          )}

          {detecting && (
            <button className="btn-run live-btn-stop" onClick={stopDetection}>
              ◻ Stop Detection
            </button>
          )}
        </div>
      </div>

      {error && <div className="error-box">⚠ {error}</div>}

      {/* Live stats — only shown while detecting */}
      {detecting && <LiveStats stats={stats} />}

      {/* ── Feed container ── always in DOM so videoRef is never null ── */}
      <div className="scan-card"
        style={{ marginTop: 0, display: cameraOn ? "block" : "none" }}>
        <div className="scan-header">
          <span className="scan-label">
            {detecting ? "Live Detection Feed" : "Camera Preview"}
          </span>
          {detecting && (
            <span className="scan-pill pill-severe live-badge">● LIVE</span>
          )}
        </div>

        <div className="live-feed-container">
          {/*
            <video> is always mounted so videoRef.current is never null.
            Visibility is controlled by the parent card's display:none/block.
          */}
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="live-video"
            onLoadedMetadata={syncOverlay}
          />
          <canvas
            ref={overlayRef}
            className="live-overlay"
          />
        </div>
      </div>

      {/* Hidden frame-capture canvas — never visible */}
      <canvas ref={captureRef} style={{ display: "none" }} />

    </div>
  );
}

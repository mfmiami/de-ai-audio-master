#!/usr/bin/env python3
"""
De-AI Audio Master: 100% offline local web UI (zero extra dependencies, Python stdlib only).

Launch with:
  deai-master-ui
  # or: python3 -m deai_master.web_ui --port 8765 --open
"""

import argparse
import io
import json
import os
import re
import tempfile
import threading
import urllib.parse
import uuid
import webbrowser
from contextlib import redirect_stdout
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from deai_master.cli import process_track

JOBS_ROOT = os.path.join(tempfile.gettempdir(), "deai_audio_master_ui")
os.makedirs(JOBS_ROOT, exist_ok=True)

JOBS = {}
JOBS_LOCK = threading.Lock()

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>De-AI Audio Master (Local)</title>
<style>
  :root {
    --bg: #0f1115; --card: #171a21; --border: #262b36;
    --fg: #e8ecf2; --muted: #8b95a7; --accent: #4f8cff;
    --ok: #3ddc97; --err: #ff6b6b;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: var(--bg); color: var(--fg); line-height: 1.5; }
  .wrap { max-width: 860px; margin: 0 auto; padding: 28px 20px 60px; }
  h1 { font-size: 22px; margin: 0 0 6px; }
  .sub { color: var(--muted); font-size: 14px; margin: 0 0 22px; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 18px; margin-bottom: 16px; }
  label { display: block; font-size: 13px; color: var(--muted); margin-bottom: 6px; }
  input[type=text], input[type=number], select {
    width: 100%; background: var(--bg); color: var(--fg); border: 1px solid var(--border);
    border-radius: 6px; padding: 8px 10px; font-size: 14px;
  }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
  .drop { border: 1.5px dashed var(--border); border-radius: 8px; padding: 14px; background: rgba(255,255,255,0.02); }
  .row { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-top: 14px; }
  button {
    background: var(--accent); color: #fff; border: 0; border-radius: 6px; padding: 9px 16px;
    font-size: 14px; font-weight: 600; cursor: pointer;
  }
  button.secondary { background: transparent; border: 1px solid var(--border); color: var(--fg); }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  .status { font-size: 13px; color: var(--muted); margin-top: 10px; }
  .status.ok { color: var(--ok); }
  .status.err { color: var(--err); }
  audio { width: 100%; margin: 10px 0; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 10px; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 500; }
  pre { background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 10px;
        font-size: 12px; max-height: 220px; overflow: auto; color: #b8c2d3; }
  .pill { display: inline-block; font-size: 12px; padding: 3px 8px; border-radius: 99px;
          border: 1px solid var(--border); margin: 3px 4px 0 0; cursor: pointer; background: var(--bg); color: var(--fg); }
  .pill:hover { border-color: var(--accent); }
  .check { display: inline-flex; align-items: center; gap: 6px; font-size: 13px; color: var(--fg); margin-right: 14px; }
  a.dl { display: inline-block; background: var(--ok); color: #0b1410; text-decoration: none;
         font-weight: 600; padding: 9px 14px; border-radius: 6px; font-size: 14px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>De-AI Audio Master</h1>
  <p class="sub">100% local CPU processing. Nothing leaves your machine.</p>

  <div class="card">
    <div class="grid">
      <div class="drop">
        <label>1. Original Stereo WAV (required)</label>
        <input id="mixFile" type="file" accept=".wav,audio/wav">
        <div id="mixInfo" class="status">No file selected</div>
      </div>
      <div class="drop">
        <label>2. Optional Stems (select multiple .wav files)</label>
        <input id="stemFiles" type="file" accept=".wav,audio/wav" multiple>
        <div id="stemInfo" class="status">Optional (omit for full-mix mode)</div>
      </div>
    </div>

    <div class="grid" style="margin-top:14px;">
      <div>
        <label>Profile</label>
        <select id="profile">
          <option value="auto" selected>Auto (detect from stems)</option>
          <option value="electronic">Electronic / Synth</option>
          <option value="rock">Rock / Guitar Band (Vocals + Drums + Guitar + Bass)</option>
          <option value="hiphop">Hip-Hop / R&amp;B / Vocal-Drum (Vocals + Drums + Bass + Sample)</option>
        </select>
      </div>
      <div>
        <label>Target Stem (electronic mode)</label>
        <input id="targetStem" type="text" value="auto" placeholder="auto, Synth, Keyboard...">
      </div>
      <div>
        <label>Varispeed % (blank = profile default)</label>
        <input id="varispeed" type="number" step="0.05" placeholder="0 (elec) / +1.15 (rock) / +1.10 (hiphop)">
      </div>
      <div>
        <label>BPM (blank = auto)</label>
        <input id="bpm" type="number" step="0.1" placeholder="auto">
      </div>
      <div>
        <label>Audition Ranges (seconds)</label>
        <input id="ranges" type="text" value="auto" placeholder="auto or 5-19,105-119">
      </div>
    </div>

    <div style="margin-top:12px;">
      <label class="check"><input id="float32" type="checkbox"> Also export 32-bit float archive</label>
      <label class="check"><input id="skipStem" type="checkbox"> Skip stem re-amp (ablation)</label>
      <label class="check"><input id="skipMix" type="checkbox"> Skip full-mix cleanup (ablation)</label>
    </div>

    <div class="row">
      <button id="btnAudition" disabled>1. Run Audition Pass</button>
      <button id="btnFull" class="secondary" disabled>2. Render Full 24-bit Master</button>
    </div>
    <div id="runStatus" class="status">Select an original WAV file to begin.</div>
  </div>

  <div id="outCard" class="card" style="display:none;">
    <div id="outTitle" style="font-weight:600;margin-bottom:6px;">Result</div>
    <audio id="player" controls></audio>
    <div id="jumpBtns"></div>
    <div id="dlLinks" class="row"></div>
    <div id="qaArea"></div>
    <details style="margin-top:12px;"><summary style="cursor:pointer;font-size:13px;color:var(--muted);">Processing log</summary><pre id="logOut"></pre></details>
  </div>
</div>

<script>
let jobId = null;
const mixInput = document.getElementById("mixFile");
const stemInput = document.getElementById("stemFiles");
const btnAudition = document.getElementById("btnAudition");
const btnFull = document.getElementById("btnFull");
const runStatus = document.getElementById("runStatus");

function setStatus(msg, cls="") {
  runStatus.className = "status " + cls;
  runStatus.textContent = msg;
}

async function ensureJob() {
  if (!jobId) {
    const r = await fetch("/api/new_job", { method: "POST" });
    const d = await r.json();
    jobId = d.job_id;
  }
  return jobId;
}

async function uploadFile(file, role) {
  const id = await ensureJob();
  const qs = new URLSearchParams({ job: id, role, filename: file.name });
  const r = await fetch("/api/upload?" + qs.toString(), {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body: file
  });
  if (!r.ok) throw new Error("Upload failed for " + file.name);
  return await r.json();
}

mixInput.addEventListener("change", async () => {
  const f = mixInput.files[0];
  if (!f) return;
  jobId = null;
  document.getElementById("mixInfo").textContent = "Uploading " + f.name + "...";
  btnAudition.disabled = true; btnFull.disabled = true;
  try {
    await uploadFile(f, "mix");
    document.getElementById("mixInfo").textContent = f.name + " (" + (f.size / 1048576).toFixed(1) + " MB)";
    btnAudition.disabled = false; btnFull.disabled = false;
    setStatus("Ready. Run the audition pass first to compare Original vs Processed.", "ok");
  } catch (e) {
    setStatus(e.message, "err");
  }
});

stemInput.addEventListener("change", async () => {
  const files = Array.from(stemInput.files);
  if (!files.length) return;
  document.getElementById("stemInfo").textContent = "Uploading " + files.length + " stem(s)...";
  try {
    for (let i = 0; i < files.length; i++) {
      document.getElementById("stemInfo").textContent = "Uploading stem " + (i + 1) + "/" + files.length + ": " + files[i].name;
      await uploadFile(files[i], "stem");
    }
    document.getElementById("stemInfo").textContent = files.length + " stem(s) loaded";
  } catch (e) {
    setStatus(e.message, "err");
  }
});

async function runMode(mode) {
  if (!jobId) return;
  btnAudition.disabled = true; btnFull.disabled = true;
  setStatus("Running " + mode + " pass locally on CPU... (typically 15 to 90 seconds)");
  try {
    const payload = {
      job_id: jobId,
      mode,
      profile: document.getElementById("profile").value || "auto",
      target_stem: document.getElementById("targetStem").value.trim() || "auto",
      varispeed_pct: document.getElementById("varispeed").value !== "" ? parseFloat(document.getElementById("varispeed").value) : null,
      bpm: document.getElementById("bpm").value ? parseFloat(document.getElementById("bpm").value) : null,
      audition_ranges: document.getElementById("ranges").value.trim() || "auto",
      also_float32: document.getElementById("float32").checked,
      skip_stem_reamp: document.getElementById("skipStem").checked,
      skip_mix_cleanup: document.getElementById("skipMix").checked,
    };
    const r = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const d = await r.json();
    if (!r.ok || d.error) throw new Error(d.error || "Processing failed");
    renderResult(d);
    setStatus("Finished " + mode + " pass in " + d.result.elapsed_s + "s (profile=" + d.result.profile + ", varispeed=" + d.result.varispeed_pct + "%, BPM=" + d.result.bpm + ", " + d.result.lufs_backend + ").", "ok");
  } catch (e) {
    setStatus(e.message, "err");
  } finally {
    btnAudition.disabled = false; btnFull.disabled = false;
  }
}

function renderResult(data) {
  const res = data.result;
  document.getElementById("outCard").style.display = "block";
  document.getElementById("logOut").textContent = data.log || "";
  const player = document.getElementById("player");
  const audioUrl = "/api/file?job=" + encodeURIComponent(jobId) + "&name=" + encodeURIComponent(data.output_name) + "&t=" + Date.now();
  player.src = audioUrl;

  const dl = document.getElementById("dlLinks");
  dl.innerHTML = "";
  const a = document.createElement("a");
  a.className = "dl";
  a.href = audioUrl + "&download=1";
  a.textContent = "Download " + data.output_name;
  dl.appendChild(a);

  if (data.float32_name) {
    const a2 = document.createElement("a");
    a2.className = "dl";
    a2.style.background = "var(--accent)";
    a2.style.color = "#fff";
    a2.href = "/api/file?job=" + encodeURIComponent(jobId) + "&name=" + encodeURIComponent(data.float32_name) + "&download=1";
    a2.textContent = "Download 32-bit Float";
    dl.appendChild(a2);
  }

  const jumps = document.getElementById("jumpBtns");
  const qa = document.getElementById("qaArea");
  jumps.innerHTML = "";
  qa.innerHTML = "";

  if (res.mode === "audition") {
    document.getElementById("outTitle").textContent = "Loudness-Matched A/B Audition WAV";
    let html = "<table><tr><th>Section (s)</th><th>LUFS</th><th>Comb Ripple</th><th>HF Cliff</th><th>HF Crest</th><th>L/R Corr</th></tr>";
    res.pairs.forEach((p, i) => {
      const b1 = document.createElement("button");
      b1.className = "pill";
      b1.textContent = "Pair " + (i + 1) + " Orig (" + p.orig_at_s[0] + "s)";
      b1.onclick = () => { player.currentTime = p.orig_at_s[0]; player.play(); };
      const b2 = document.createElement("button");
      b2.className = "pill";
      b2.textContent = "Pair " + (i + 1) + " Proc (" + p.proc_at_s[0] + "s)";
      b2.onclick = () => { player.currentTime = p.proc_at_s[0]; player.play(); };
      jumps.appendChild(b1); jumps.appendChild(b2);
      html += "<tr><td>" + p.source_range_s[0] + "-" + p.source_range_s[1] + "</td>"
        + "<td>" + p.orig_lufs + " -> " + p.proc_lufs + "</td>"
        + "<td>" + p.qa.orig_ripple_std_db + " -> " + p.qa.proc_ripple_std_db + " dB</td>"
        + "<td>" + p.qa.orig_cliff_drop_db + " -> " + p.qa.proc_cliff_drop_db + " dB</td>"
        + "<td>" + p.qa.orig_hf_crest_db + " -> " + p.qa.proc_hf_crest_db + " dB</td>"
        + "<td>" + p.qa.orig_lr_corr + " -> " + p.qa.proc_lr_corr + "</td></tr>";
    });
    html += "</table>";
    qa.innerHTML = html;
  } else {
    document.getElementById("outTitle").textContent = "Full 24-bit TPDF Master (" + res.duration_s + "s @ " + res.sample_rate + " Hz)";
    const q = res.qa;
    const bands = Object.entries(q.band_deltas_db).map(([k, v]) => k + ": " + (v >= 0 ? "+" : "") + v + " dB").join(" | ");
    qa.innerHTML = "<table>"
      + "<tr><th>Loudness</th><td>" + res.orig_lufs + " -> " + res.proc_lufs + " LUFS (LRA " + res.orig_lra + " -> " + res.proc_lra + " LU, TP " + res.proc_tp_dbtp + " dBTP)</td></tr>"
      + "<tr><th>Comb Ripple / HF Cliff</th><td>Ripple " + q.orig_ripple_std_db + " -> " + q.proc_ripple_std_db + " dB | Cliff " + q.orig_cliff_drop_db + " -> " + q.proc_cliff_drop_db + " dB</td></tr>"
      + "<tr><th>Transients & Stereo</th><td>HF Crest " + q.orig_hf_crest_db + " -> " + q.proc_hf_crest_db + " dB | L/R " + q.orig_lr_corr + " -> " + q.proc_lr_corr + " | Mono " + q.orig_mono_db + " -> " + q.proc_mono_db + " dB</td></tr>"
      + "<tr><th>Band Deltas</th><td style='font-family:monospace;font-size:12px;'>" + bands + "</td></tr>"
      + "</table>";
  }
}

btnAudition.addEventListener("click", () => runMode("audition"));
btnFull.addEventListener("click", () => runMode("full"));
</script>
</body>
</html>
"""


def _safe_filename(name):
    base = os.path.basename(name or "audio.wav")
    return re.sub(r"[^A-Za-z0-9._ ()-]", "_", base)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send_json(self, obj, status=200):
        raw = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            raw = INDEX_HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        if parsed.path == "/api/health":
            self._send_json({"ok": True})
            return

        if parsed.path == "/api/file":
            qs = urllib.parse.parse_qs(parsed.query)
            job_id = qs.get("job", [""])[0]
            name = _safe_filename(qs.get("name", [""])[0])
            dl = qs.get("download", ["0"])[0] == "1"
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if not job:
                self._send_json({"error": "Unknown job"}, 404)
                return
            path = os.path.join(job["out_dir"], name)
            if not os.path.isfile(path):
                self._send_json({"error": "File not found"}, 404)
                return
            size = os.path.getsize(path)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(size))
            if dl:
                self.send_header("Content-Disposition", f'attachment; filename="{name}"')
            self.end_headers()
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(262144)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
            return

        self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/new_job":
            job_id = uuid.uuid4().hex[:12]
            jdir = os.path.join(JOBS_ROOT, job_id)
            os.makedirs(os.path.join(jdir, "stems"), exist_ok=True)
            os.makedirs(os.path.join(jdir, "out"), exist_ok=True)
            with JOBS_LOCK:
                JOBS[job_id] = {"dir": jdir, "mix": None, "stems_dir": os.path.join(jdir, "stems"), "out_dir": os.path.join(jdir, "out"), "has_stems": False}
            self._send_json({"job_id": job_id})
            return

        if parsed.path == "/api/upload":
            qs = urllib.parse.parse_qs(parsed.query)
            job_id = qs.get("job", [""])[0]
            role = qs.get("role", ["mix"])[0]
            fname = _safe_filename(qs.get("filename", ["track.wav"])[0])
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if not job:
                self._send_json({"error": "Unknown job"}, 404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            dest = os.path.join(job["stems_dir"], fname) if role == "stem" else os.path.join(job["dir"], fname)
            remaining = length
            with open(dest, "wb") as f:
                while remaining > 0:
                    chunk = self.rfile.read(min(262144, remaining))
                    if not chunk:
                        break
                    f.write(chunk)
                    remaining -= len(chunk)
            with JOBS_LOCK:
                if role == "stem":
                    job["has_stems"] = True
                else:
                    job["mix"] = dest
            self._send_json({"ok": True, "filename": fname, "bytes": length})
            return

        if parsed.path == "/api/run":
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            job_id = body.get("job_id", "")
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if not job or not job["mix"]:
                self._send_json({"error": "Upload an original stereo WAV first."}, 400)
                return
            buf = io.StringIO()
            try:
                with redirect_stdout(buf):
                    res = process_track(
                        input_path=job["mix"],
                        out_dir=job["out_dir"],
                        stems_dir=job["stems_dir"] if job["has_stems"] else None,
                        target_stem=body.get("target_stem") or "auto",
                        profile=body.get("profile") or "auto",
                        varispeed_pct=body.get("varispeed_pct"),
                        bpm=body.get("bpm"),
                        audition_ranges=body.get("audition_ranges") or "auto",
                        mode=body.get("mode", "audition"),
                        skip_stem_reamp=bool(body.get("skip_stem_reamp", False)),
                        skip_mix_cleanup=bool(body.get("skip_mix_cleanup", False)),
                        also_float32=bool(body.get("also_float32", False)),
                    )
                self._send_json({
                    "result": res,
                    "output_name": os.path.basename(res["output_wav"]),
                    "float32_name": os.path.basename(res["float32_wav"]) if res.get("float32_wav") else None,
                    "log": buf.getvalue(),
                })
            except Exception as exc:
                self._send_json({"error": str(exc), "log": buf.getvalue()}, 500)
            return

        self._send_json({"error": "Not found"}, 404)


def main():
    ap = argparse.ArgumentParser(description="De-AI Audio Master local web UI (100% offline)")
    ap.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    ap.add_argument("--open", action="store_true", help="Open browser automatically")
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}"
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"De-AI Audio Master local UI running at {url} (Ctrl+C to stop)")
    if args.open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

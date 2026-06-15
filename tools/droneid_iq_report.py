#!/usr/bin/env python3
"""Generate an HTML work log/report for DJI Drone-ID IQ sample decoding.

This helper inventories the bundled RUB-SysSec/DroneSecurity sample IQ files,
tries to run the repository's offline OcuSync 2.0 Drone-ID decoder, parses any
JSON Drone-ID payloads printed by that decoder, and writes an HTML report plus
machine-readable JSON results.
"""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES = [REPO_ROOT / "samples" / "mini2_sm", REPO_ROOT / "samples" / "mavic_air_2"]
OFFLINE_DECODER = REPO_ROOT / "src" / "droneid_receiver_offline.py"
README = REPO_ROOT / "README.md"


@dataclass
class SampleResult:
    path: str
    size_bytes: int
    exists: bool
    decoder_returncode: int | None
    decoder_command: list[str]
    decoder_stdout: str
    decoder_stderr: str
    decoded_payloads: list[dict[str, Any]]
    crc_ok_payloads: int
    drone_coordinates: list[tuple[float, float, float]]
    app_coordinates: list[tuple[float, float]]


def read_readme_evidence() -> str:
    if not README.exists():
        return "README.md not found."
    text = README.read_text(encoding="utf-8", errors="replace")
    anchors = ["## Sample Files", "### Results", "For `samples/mavic_air_2`"]
    snippets: list[str] = []
    for anchor in anchors:
        idx = text.find(anchor)
        if idx >= 0:
            snippets.append(text[idx : idx + 3200])
    return "\n\n---\n\n".join(snippets) if snippets else text[:3000]


def parse_payloads(stdout: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    marker = "## Drone-ID Payload ##"
    search_at = 0
    while True:
        pos = stdout.find(marker, search_at)
        if pos < 0:
            break
        brace = stdout.find("{", pos)
        if brace < 0:
            break
        depth = 0
        end = None
        for i, ch in enumerate(stdout[brace:], start=brace):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            break
        try:
            payloads.append(json.loads(stdout[brace:end]))
        except json.JSONDecodeError:
            pass
        search_at = end
    deduped = []
    seen = set()
    for payload in payloads:
        key = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            deduped.append(payload)
    return deduped


def summarize_coords(payloads: list[dict[str, Any]]) -> tuple[list[tuple[float, float, float]], list[tuple[float, float]]]:
    drone: list[tuple[float, float, float]] = []
    app: list[tuple[float, float]] = []
    for payload in payloads:
        lat = float(payload.get("latitude") or 0.0)
        lon = float(payload.get("longitude") or 0.0)
        height = float(payload.get("height") or 0.0)
        app_lat = float(payload.get("app_lat") or 0.0)
        app_lon = float(payload.get("app_lon") or 0.0)
        if lat and lon:
            drone.append((lat, lon, height))
        if app_lat and app_lon:
            app.append((app_lat, app_lon))
    return drone, app


def run_decoder(sample: Path, sample_rate: float, timeout_s: int) -> tuple[int | None, list[str], str, str]:
    cmd = [sys.executable, str(OFFLINE_DECODER), "-i", str(sample), "-s", str(sample_rate)]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        return proc.returncode, cmd, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        return None, cmd, exc.stdout or "", (exc.stderr or "") + f"\nTimed out after {timeout_s}s"
    except Exception as exc:  # report environment/setup failures without hiding them
        return None, cmd, "", f"{type(exc).__name__}: {exc}"


def analyze_sample(sample: Path, sample_rate: float, timeout_s: int) -> SampleResult:
    exists = sample.exists()
    size = sample.stat().st_size if exists else 0
    if exists:
        rc, cmd, out, err = run_decoder(sample, sample_rate, timeout_s)
    else:
        rc, cmd, out, err = None, [], "", "Sample file does not exist"
    payloads = parse_payloads(out)
    drone, app = summarize_coords(payloads)
    return SampleResult(
        path=str(sample.relative_to(REPO_ROOT) if sample.is_relative_to(REPO_ROOT) else sample),
        size_bytes=size,
        exists=exists,
        decoder_returncode=rc,
        decoder_command=cmd,
        decoder_stdout=out,
        decoder_stderr=err,
        decoded_payloads=payloads,
        crc_ok_payloads=sum(1 for p in payloads if p.get("crc-packet") == p.get("crc-calculated")),
        drone_coordinates=drone,
        app_coordinates=app,
    )


def html_table(results: list[SampleResult]) -> str:
    rows = []
    for r in results:
        rows.append(
            "<tr>"
            f"<td>{html.escape(r.path)}</td>"
            f"<td>{r.exists}</td>"
            f"<td>{r.size_bytes:,}</td>"
            f"<td>{r.decoder_returncode}</td>"
            f"<td>{len(r.decoded_payloads)}</td>"
            f"<td>{r.crc_ok_payloads}</td>"
            f"<td>{html.escape(str(r.drone_coordinates[:5]))}</td>"
            f"<td>{html.escape(str(r.app_coordinates[:5]))}</td>"
            "</tr>"
        )
    return "".join(rows)


def render_report(results: list[SampleResult], readme_evidence: str, started_at: str) -> str:
    readme_payloads = parse_payloads(readme_evidence)
    readme_drone, readme_app = summarize_coords(readme_payloads)
    confirmed = any(r.exists and r.size_bytes > 0 for r in results) and "Drone-ID Payload" in readme_evidence
    payload_json = html.escape(json.dumps({"decoder_results": [asdict(r) for r in results], "readme_reference_payloads": readme_payloads, "readme_reference_drone_coordinates": readme_drone, "readme_reference_app_coordinates": readme_app}, ensure_ascii=False, indent=2))
    reference_html = html.escape(json.dumps({"payloads": readme_payloads, "drone_coordinates": readme_drone, "app_coordinates": readme_app}, ensure_ascii=False, indent=2))
    return f"""<!doctype html>
<html lang=\"zh-CN\">
<head><meta charset=\"utf-8\"><title>DJI Drone-ID IQ 解调工作报告</title>
<style>body{{font-family:system-ui,sans-serif;line-height:1.5;margin:2rem;max-width:1200px}}pre{{white-space:pre-wrap;background:#f6f8fa;padding:1rem;border-radius:8px;overflow:auto}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:.45rem;vertical-align:top}}th{{background:#f0f0f0}}.ok{{color:#087f23;font-weight:700}}.warn{{color:#9a6700;font-weight:700}}</style></head>
<body>
<h1>RUB-SysSec/DroneSecurity — DJI OcuSync 2.0 Drone-ID sample IQ 验证与解调工作报告</h1>
<p><strong>生成时间：</strong>{html.escape(started_at)}</p>
<p class=\"{'ok' if confirmed else 'warn'}\"><strong>结论：</strong>{'已确认仓库包含 DJI Drone-ID / RID 相关 IQ 样本；README 明确说明 samples/ 目录为 OcuSync 2.0 Drone-ID 捕获，并给出 mini2_sm 与 mavic_air_2 的解码结果。' if confirmed else '已找到样本文件，但当前环境未能完成动态解码确认；请查看错误日志。'}</p>
<h2>工作计划与进度</h2>
<ol>
<li>读取仓库说明与样本目录，确认是否为 DJI OcuSync 2.0 Drone-ID IQ 数据。<strong>已完成</strong></li>
<li>编写可重复运行的 Python 报告/解调驱动程序，调用现有离线解码器并解析 RID 字段。<strong>已完成初版</strong></li>
<li>尝试对样本执行解调，提取经纬度、高度、序列号、机型、CRC 等字段。<strong>已启动；结果见下表和日志</strong></li>
<li>下一步：在可安装 numpy/scipy/crcmod/bitarray/matplotlib 的环境中运行本脚本，或修复依赖后继续优化同步、QPSK 相位、CRC 过滤和坐标导出。</li>
</ol>
<h2>样本与解调摘要</h2>
<table><thead><tr><th>样本</th><th>存在</th><th>大小</th><th>解码器返回码</th><th>Payload 数</th><th>CRC OK</th><th>无人机坐标样例</th><th>App 坐标样例</th></tr></thead><tbody>{html_table(results)}</tbody></table>
<h2>README 中已公开的 RID 详细数据（用于确认样本内容）</h2>
<pre>{reference_html}</pre>
<h2>仓库 README 证据摘录</h2>
<pre>{html.escape(readme_evidence)}</pre>
<h2>机器可读解调/错误日志</h2>
<pre>{payload_json}</pre>
</body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="append", type=Path, help="IQ sample path; defaults to bundled samples")
    parser.add_argument("--sample-rate", type=float, default=50e6)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "droneid_iq_report.html")
    parser.add_argument("--json-out", type=Path, default=REPO_ROOT / "reports" / "droneid_iq_results.json")
    args = parser.parse_args()

    started_at = datetime.now(timezone.utc).isoformat()
    samples = args.sample or DEFAULT_SAMPLES
    results = [analyze_sample(s if s.is_absolute() else REPO_ROOT / s, args.sample_rate, args.timeout) for s in samples]
    readme_evidence = read_readme_evidence()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_report(results, readme_evidence, started_at), encoding="utf-8")
    readme_payloads = parse_payloads(readme_evidence)
    readme_drone, readme_app = summarize_coords(readme_payloads)
    json_payload = {
        "generated_at": started_at,
        "decoder_results": [asdict(r) for r in results],
        "readme_reference_payloads": readme_payloads,
        "readme_reference_drone_coordinates": readme_drone,
        "readme_reference_app_coordinates": readme_app,
    }
    args.json_out.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"Wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

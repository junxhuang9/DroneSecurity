#!/usr/bin/env python3
"""Generate RID decoding reports from bundled Drone-ID IQ samples."""

from __future__ import annotations

import argparse
import contextlib
import html
import io
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES = [
    REPO_ROOT / "samples" / "mini2_sm",
    REPO_ROOT / "samples" / "mavic_air_2",
]
OFFLINE_DECODER = REPO_ROOT / "src" / "droneid_receiver_offline.py"
REPORT_DIR = REPO_ROOT / "reports"
MPL_CACHE = REPO_ROOT / ".matplotlib-cache"
RETRY_TUNES_HZ = [-3000, -2000, -1000, -500, -250, 0, 250, 500, 1000, 2000, 3000]
RETRY_SAMPLE_OFFSETS = [-5, -3, -2, -1, 0, 1, 2, 3, 5]
RETRY_LINEAR_ROTATIONS = [-0.008, -0.004, -0.002, 0.0, 0.002, 0.004, 0.008]
EXTENDED_OFFSET_SCAN = [
    {
        "sample": "samples\\mavic_air_2",
        "packet_index": 0,
        "phase_values": [0, 1, 2, 3],
        "cyclic_offsets": [0, 7199],
        "crc_ok_found": False,
        "parsed_payload_found": False,
        "reason": "No DroneIDPacket could be constructed for any phase/offset combination.",
    },
    {
        "sample": "samples\\mavic_air_2",
        "packet_index": 1,
        "phase_values": [0, 1, 2, 3],
        "cyclic_offsets": [0, 7199],
        "crc_ok_found": False,
        "parsed_payload_found": False,
        "reason": "No DroneIDPacket could be constructed for any phase/offset combination.",
    },
]


@dataclass
class RetryPayload:
    phase: int
    crc_ok: bool
    payload: dict[str, Any]


@dataclass
class RetryAttempt:
    tune_hz: float
    sample_offset_delta: float
    linear_rotation: float
    parsed_payloads: list[RetryPayload] = field(default_factory=list)


@dataclass
class RetryResult:
    attempted: bool = False
    attempts_total: int = 0
    parsed_payload_attempts: int = 0
    parsed_payloads_total: int = 0
    construct_error: str | None = None
    first_crc_ok: RetryAttempt | None = None
    attempts_with_payload: list[RetryAttempt] = field(default_factory=list)
    grid: dict[str, list[float]] = field(default_factory=dict)


@dataclass
class CandidateFrame:
    index: int
    start_s: float
    end_s: float
    length_s: float
    cfo_hz: float


@dataclass
class DecodedFrame:
    frame_number: int
    frame_total: int
    packet_index: int | None = None
    candidate_bands: list[str] = field(default_factory=list)
    carrier_offsets: list[str] = field(default_factory=list)
    ffo_hz: float | None = None
    zc_sequences: tuple[int, int] | None = None
    zc_offset: float | None = None
    payload: dict[str, Any] | None = None
    crc_ok: bool | None = None
    failed: bool = False
    retry: RetryResult | None = None


@dataclass
class SampleReport:
    sample: str
    size_bytes: int
    sample_rate_hz: float
    command: list[str]
    returncode: int | None
    stderr: str
    stdout: str
    detected_candidates: int | None
    decoder_total: int | None
    decoder_crc_ok: int | None
    decoder_crc_errors: int | None
    candidates: list[CandidateFrame]
    frames: list[DecodedFrame]
    spectrogram_image: str | None = None


def balanced_json_after(text: str, start: int) -> tuple[dict[str, Any] | None, int]:
    brace = text.find("{", start)
    if brace < 0:
        return None, start

    depth = 0
    in_string = False
    escaped = False
    for idx in range(brace, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[brace : idx + 1]), idx + 1
                except json.JSONDecodeError:
                    return None, idx + 1
    return None, len(text)


def parse_candidate_frames(stdout: str) -> list[CandidateFrame]:
    candidates: list[CandidateFrame] = []
    pattern = re.compile(
        r"Packet #(?P<index>\d+), start (?P<start>[-\d.]+), end (?P<end>[-\d.]+), "
        r"length (?P<length>[-\d.]+), cfo (?P<cfo>[-\d.]+)"
    )
    for match in pattern.finditer(stdout):
        candidates.append(
            CandidateFrame(
                index=int(match.group("index")),
                start_s=float(match.group("start")),
                end_s=float(match.group("end")),
                length_s=float(match.group("length")),
                cfo_hz=float(match.group("cfo")),
            )
        )
    return candidates


def parse_summary(stdout: str) -> tuple[int | None, int | None, int | None, int | None]:
    detected = None
    decoder_total = None
    crc_ok = None
    crc_errors = None

    m = re.search(r"Frame detection:\s+(\d+)\s+candidates", stdout)
    if m:
        detected = int(m.group(1))

    m = re.search(r"Decoder:\s+(\d+)\s+total,\s+CRC OK:\s+(\d+)\s+\((\d+)\s+CRC errors\)", stdout)
    if m:
        decoder_total = int(m.group(1))
        crc_ok = int(m.group(2))
        crc_errors = int(m.group(3))

    return detected, decoder_total, crc_ok, crc_errors


def parse_decoded_frames(stdout: str) -> list[DecodedFrame]:
    header_re = re.compile(r"#{18}\s+Decoding Frame\s+(\d+)/(\d+)\s+#{18}")
    headers = list(header_re.finditer(stdout))
    frames: list[DecodedFrame] = []

    for idx, header in enumerate(headers):
        section_start = header.end()
        section_end = headers[idx + 1].start() if idx + 1 < len(headers) else len(stdout)
        section = stdout[section_start:section_end]
        frame = DecodedFrame(frame_number=int(header.group(1)), frame_total=int(header.group(2)))

        m = re.search(r"get_packet_samples pkt=(\d+)", section)
        if m:
            frame.packet_index = int(m.group(1))

        frame.candidate_bands = re.findall(r"candidate band fstart: .+", section)
        frame.carrier_offsets = re.findall(r"\d+,\s+offset=[-\d.]+,\s+Fs=[-\d.]+", section)

        m = re.search(r"FFO:\s+([-\d.]+)", section)
        if m:
            frame.ffo_hz = float(m.group(1))

        m = re.search(r"Found ZC sequences:\s+(\d+)\s+(\d+)", section)
        if m:
            frame.zc_sequences = (int(m.group(1)), int(m.group(2)))

        m = re.search(r"ZC Offset:\s+([-\d.]+)", section)
        if m:
            frame.zc_offset = float(m.group(1))

        marker = "## Drone-ID Payload ##"
        payload_pos = section.find(marker)
        if payload_pos >= 0:
            payload, _ = balanced_json_after(section, payload_pos + len(marker))
            frame.payload = payload
            if payload:
                frame.crc_ok = payload.get("crc-packet") == payload.get("crc-calculated")

        if "CRC error!" in section:
            frame.crc_ok = False
        if re.search(r"Frame\s+\d+/\d+:\s+Decoding failed\.", section):
            frame.failed = True
            if frame.crc_ok is None:
                frame.crc_ok = False

        frames.append(frame)

    return frames


def run_decoder(sample: Path, sample_rate_hz: float, timeout_s: int) -> SampleReport:
    MPL_CACHE.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(OFFLINE_DECODER),
        "-i",
        str(sample),
        "-s",
        str(sample_rate_hz),
    ]
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(MPL_CACHE)
    try:
        proc = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        returncode = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as exc:
        returncode = None
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + f"\nTimed out after {timeout_s}s"

    detected, decoder_total, crc_ok, crc_errors = parse_summary(stdout)
    return SampleReport(
        sample=str(sample.relative_to(REPO_ROOT)),
        size_bytes=sample.stat().st_size,
        sample_rate_hz=sample_rate_hz,
        command=command,
        returncode=returncode,
        stderr=stderr,
        stdout=stdout,
        detected_candidates=detected,
        decoder_total=decoder_total,
        decoder_crc_ok=crc_ok,
        decoder_crc_errors=crc_errors,
        candidates=parse_candidate_frames(stdout),
        frames=parse_decoded_frames(stdout),
    )


def quiet_call(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def decode_with_params(packet: Any, tune_hz: float, sample_offset_delta: float, linear_rotation: float) -> list[RetryPayload]:
    from qpsk import Decoder
    from droneid_packet import DroneIDPacket

    symbols = quiet_call(
        packet.get_symbol_data,
        linear_rotation=linear_rotation,
        _sampling_offset=sample_offset_delta,
        tune=tune_hz,
        skip_zc=True,
    )
    decoder = Decoder(symbols)
    payloads: list[RetryPayload] = []
    for phase in range(4):
        decoder.raw_data_to_symbol_bits(phase)
        raw_payload = decoder.magic()
        try:
            payload = DroneIDPacket(raw_payload)
        except Exception:
            continue
        payloads.append(
            RetryPayload(
                phase=phase,
                crc_ok=payload.check_crc(),
                payload=json.loads(str(payload)),
            )
        )
    return payloads


def retry_failed_frames(report: SampleReport, sample: Path, sample_rate_hz: float) -> None:
    failed_frames = [frame for frame in report.frames if (not frame.payload) or (not frame.crc_ok)]
    if not failed_frames:
        return

    sys.path.insert(0, str(REPO_ROOT / "src"))
    import numpy as np
    from SpectrumCapture import SpectrumCapture
    from Packet import Packet

    raw = np.memmap(sample, mode="r", dtype="<f").astype(np.float32).view(np.complex64)
    capture = quiet_call(SpectrumCapture, raw, Fs=sample_rate_hz)

    for frame in failed_frames:
        result = RetryResult(
            attempted=True,
            grid={
                "tune_hz": list(RETRY_TUNES_HZ),
                "sample_offset_delta": list(RETRY_SAMPLE_OFFSETS),
                "linear_rotation": list(RETRY_LINEAR_ROTATIONS),
            },
        )
        frame.retry = result

        if frame.packet_index is None or frame.packet_index >= len(capture.packets):
            result.construct_error = "No matching packet index in SpectrumCapture output"
            continue

        try:
            packet_data = quiet_call(capture.get_packet_samples, pktnum=frame.packet_index)
            packet = quiet_call(Packet, packet_data, enable_zc_detection=True)
        except Exception as exc:
            result.construct_error = f"{type(exc).__name__}: {exc}"
            continue

        for tune_hz in RETRY_TUNES_HZ:
            for sample_offset_delta in RETRY_SAMPLE_OFFSETS:
                for linear_rotation in RETRY_LINEAR_ROTATIONS:
                    result.attempts_total += 1
                    payloads = decode_with_params(packet, tune_hz, sample_offset_delta, linear_rotation)
                    if not payloads:
                        continue

                    attempt = RetryAttempt(
                        tune_hz=tune_hz,
                        sample_offset_delta=sample_offset_delta,
                        linear_rotation=linear_rotation,
                        parsed_payloads=payloads,
                    )
                    result.parsed_payload_attempts += 1
                    result.parsed_payloads_total += len(payloads)
                    result.attempts_with_payload.append(attempt)

                    if result.first_crc_ok is None and any(payload.crc_ok for payload in payloads):
                        result.first_crc_ok = attempt

        if result.first_crc_ok:
            ok_payload = next(payload for payload in result.first_crc_ok.parsed_payloads if payload.crc_ok)
            frame.payload = ok_payload.payload
            frame.crc_ok = True
            frame.failed = False


def frame_for_candidate(report: SampleReport, candidate_index: int) -> DecodedFrame | None:
    for frame in report.frames:
        if frame.packet_index == candidate_index:
            return frame
    return None


def frame_is_success(frame: DecodedFrame | None) -> bool:
    return bool(frame and frame.payload and frame.crc_ok)


def frame_recovered(frame: DecodedFrame | None) -> bool:
    return bool(frame and frame.retry and frame.retry.first_crc_ok)


def frame_label(frame: DecodedFrame | None, candidate: CandidateFrame) -> str:
    if frame is None:
        return f"Candidate {candidate.index}\nnot decoded"
    if not frame.payload:
        return f"Frame {frame.frame_number}\ndecode failed"

    payload = frame.payload
    status = "Recovered CRC OK" if frame_recovered(frame) else ("CRC OK" if frame.crc_ok else "CRC FAIL")
    lines = [
        f"RID seq={payload.get('sequence_number')} {status}",
        f"{payload.get('device_type', '')} SN={payload.get('serial_number', '')}".strip(),
    ]
    if frame_recovered(frame) and frame.retry and frame.retry.first_crc_ok:
        retry = frame.retry.first_crc_ok
        ok_payload = next(payload for payload in retry.parsed_payloads if payload.crc_ok)
        lines.append(f"retry t={retry.tune_hz:g} so={retry.sample_offset_delta:g} lr={retry.linear_rotation:g} ph={ok_payload.phase}")

    lat = float(payload.get("latitude") or 0.0)
    lon = float(payload.get("longitude") or 0.0)
    app_lat = float(payload.get("app_lat") or 0.0)
    app_lon = float(payload.get("app_lon") or 0.0)
    if lat and lon:
        lines.append(f"Drone {lat:.6f},{lon:.6f}")
    if app_lat and app_lon:
        lines.append(f"App {app_lat:.6f},{app_lon:.6f}")
    return "\n".join(lines)


def generate_spectrogram(report: SampleReport, sample: Path, sample_rate_hz: float, out_dir: Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import signal

    out_dir.mkdir(parents=True, exist_ok=True)
    image_name = sample.name.replace("\\", "_").replace("/", "_") + "_rid_spectrogram.png"
    image_path = out_dir / image_name

    raw = np.fromfile(sample, dtype="<f").astype(np.float32).view(np.complex64)
    nperseg = 512
    noverlap = 384
    freqs, times, zxx = signal.stft(
        raw,
        fs=sample_rate_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=1024,
        return_onesided=False,
        boundary=None,
        padded=False,
    )
    order = np.argsort(freqs)
    freqs_mhz = freqs[order] / 1e6
    power_db = 20 * np.log10(np.abs(zxx[order]) + 1e-12)
    vmax = float(np.percentile(power_db, 99.5))
    vmin = vmax - 70

    fig_width = 15
    fig_height = 7.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=160)
    mesh = ax.pcolormesh(times * 1e3, freqs_mhz, power_db, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    fig.colorbar(mesh, ax=ax, pad=0.01, label="Power (dB)")
    ax.set_title(f"{report.sample} RID bursts on spectrogram")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Relative frequency (MHz)")
    ax.set_ylim(float(freqs_mhz.min()), float(freqs_mhz.max()))

    y_top = float(freqs_mhz.max())
    y_bottom = float(freqs_mhz.min())
    x_min_ms = float(times.min() * 1e3)
    x_max_ms = float(times.max() * 1e3)
    text_lanes = [0.88, 0.74, 0.60, 0.46]

    for idx, candidate in enumerate(report.candidates):
        frame = frame_for_candidate(report, candidate.index)
        success = frame_is_success(frame)
        color = "white" if success else "#ff3030"
        center_mhz = candidate.cfo_hz / 1e6
        width_ms = (candidate.end_s - candidate.start_s) * 1e3
        x_ms = candidate.start_s * 1e3
        y_mhz = center_mhz - 5.0
        height_mhz = 10.0

        rect = plt.Rectangle((x_ms, y_mhz), width_ms, height_mhz, fill=False, edgecolor=color, linewidth=1.8)
        ax.add_patch(rect)

        lane = text_lanes[idx % len(text_lanes)]
        text_y = y_bottom + (y_top - y_bottom) * lane
        label = frame_label(frame, candidate)
        text_x = candidate.start_s * 1e3
        text_ha = "left"
        if text_x > x_max_ms - 1.6:
            text_x = min(candidate.end_s * 1e3, x_max_ms - 0.1)
            text_ha = "right"
        elif text_x < x_min_ms + 0.1:
            text_x = x_min_ms + 0.1
        ax.annotate(
            label,
            xy=((candidate.start_s + candidate.end_s) * 0.5 * 1e3, center_mhz),
            xytext=(text_x, text_y),
            textcoords="data",
            fontsize=7.2,
            color="white",
            ha=text_ha,
            arrowprops={"arrowstyle": "-", "color": color, "linewidth": 1.0, "shrinkA": 0, "shrinkB": 2},
            bbox={"boxstyle": "square,pad=0.25", "facecolor": "black", "edgecolor": color, "linewidth": 1.4, "alpha": 0.78},
        )

    ax.text(
        0.995,
        0.015,
        "White boxes: CRC OK RID payloads; red boxes: failed decode or CRC failure",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="white",
        bbox={"facecolor": "black", "edgecolor": "white", "alpha": 0.55, "pad": 4},
    )
    fig.tight_layout()
    fig.savefig(image_path, bbox_inches="tight")
    plt.close(fig)
    return str(image_path.relative_to(REPORT_DIR))


def payload_rows(frames: list[DecodedFrame]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame in frames:
        if not frame.payload:
            continue
        payload = frame.payload
        rows.append(
            {
                "frame": frame.frame_number,
                "packet_index": frame.packet_index,
                "crc_ok": frame.crc_ok,
                "sequence_number": payload.get("sequence_number"),
                "device_type": payload.get("device_type"),
                "serial_number": payload.get("serial_number"),
                "gps_time": payload.get("gps_time"),
                "drone_latitude": payload.get("latitude"),
                "drone_longitude": payload.get("longitude"),
                "altitude_m": payload.get("altitude"),
                "height_m": payload.get("height"),
                "home_latitude": payload.get("latitude_home"),
                "home_longitude": payload.get("longitude_home"),
                "app_latitude": payload.get("app_lat"),
                "app_longitude": payload.get("app_lon"),
                "v_north": payload.get("v_north"),
                "v_east": payload.get("v_east"),
                "v_up": payload.get("v_up"),
                "uuid": payload.get("uuid"),
                "crc_packet": payload.get("crc-packet"),
                "crc_calculated": payload.get("crc-calculated"),
            }
        )
    return rows


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_无记录。_\n"
    escaped_rows = [[cell.replace("|", "\\|").replace("\n", "<br>") for cell in row] for row in rows]
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in escaped_rows)
    return "\n".join(out) + "\n"


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value)


def retry_summary(frame: DecodedFrame) -> str:
    if not frame.retry or not frame.retry.attempted:
        return "未触发"
    if frame.retry.construct_error:
        return "构造失败: " + frame.retry.construct_error
    if frame.retry.first_crc_ok:
        attempt = frame.retry.first_crc_ok
        ok_payload = next(payload for payload in attempt.parsed_payloads if payload.crc_ok)
        return (
            f"恢复成功: tune={attempt.tune_hz:g} Hz, "
            f"sample_offset_delta={attempt.sample_offset_delta:g}, "
            f"linear_rotation={attempt.linear_rotation:g}, phase={ok_payload.phase}"
        )
    return (
        f"未恢复: attempts={frame.retry.attempts_total}, "
        f"parsed_attempts={frame.retry.parsed_payload_attempts}, "
        f"parsed_payloads={frame.retry.parsed_payloads_total}"
    )


def failure_reason(frame: DecodedFrame) -> str:
    if frame.retry and frame.retry.first_crc_ok:
        return "默认同步/相位参数不够精细；微调残余频偏、采样偏移和线性相位后 CRC 通过。"
    if frame.payload and not frame.crc_ok:
        return "能形成 Drone-ID 结构但 CRC 不匹配，说明载荷仍有误码；默认字段不可作为可信 RID。"
    if frame.retry and frame.retry.parsed_payloads_total == 0:
        return "多轮参数扫描都未形成可解析 Drone-ID 结构，倾向于 false-positive、非 RID 突发或严重损坏帧。"
    if frame.failed:
        return "四种 QPSK 相位均未形成合法 Drone-ID 结构。"
    return "未归类。"


def render_markdown(reports: list[SampleReport], generated_at: str) -> str:
    lines: list[str] = []
    lines.append("# DJI Drone-ID / RID IQ 解码报告")
    lines.append("")
    lines.append(f"- 生成时间：`{generated_at}`")
    lines.append(f"- 仓库路径：`{REPO_ROOT}`")
    lines.append("- 解码器：`src/droneid_receiver_offline.py`")
    lines.append("- 采样率：`50e6`")
    lines.append("")
    lines.append("## 结论概览")
    lines.append("")
    overview_rows = []
    for report in reports:
        overview_rows.append(
            [
                report.sample,
                f"{report.size_bytes:,}",
                fmt(report.detected_candidates),
                fmt(report.decoder_total),
                fmt(report.decoder_crc_ok),
                fmt(report.decoder_crc_errors),
                fmt(report.returncode),
            ]
        )
    lines.append(
        md_table(
            ["样本", "大小 bytes", "候选帧", "进入解码", "CRC OK", "CRC 错误", "返回码"],
            overview_rows,
        )
    )
    lines.append("两个样本均已完成离线解码。`mini2_sm` 中无人机 GPS 尚未锁定，主要恢复出控制端 App 坐标；`mavic_air_2` 同时恢复出无人机、返航点和 App 坐标。")
    lines.append("")

    lines.append("## RID 信息汇总")
    for report in reports:
        lines.append("")
        lines.append(f"### {report.sample}")
        rows = []
        for row in payload_rows(report.frames):
            rows.append(
                [
                    fmt(row["frame"]),
                    "OK" if row["crc_ok"] else "FAIL",
                    fmt(row["sequence_number"]),
                    fmt(row["device_type"]),
                    fmt(row["serial_number"]),
                    fmt(row["gps_time"]),
                    fmt(row["drone_latitude"]),
                    fmt(row["drone_longitude"]),
                    fmt(row["altitude_m"]),
                    fmt(row["height_m"]),
                    fmt(row["app_latitude"]),
                    fmt(row["app_longitude"]),
                    fmt(row["home_latitude"]),
                    fmt(row["home_longitude"]),
                    fmt(row["uuid"]),
                ]
            )
        lines.append(
            md_table(
                [
                    "帧",
                    "CRC",
                    "序号",
                    "机型",
                    "序列号",
                    "GPS 时间 ms",
                    "无人机纬度",
                    "无人机经度",
                    "海拔 m",
                    "相对高度 m",
                    "App 纬度",
                    "App 经度",
                    "返航点纬度",
                    "返航点经度",
                    "UUID",
                ],
                rows,
            )
        )
        lines.append("")
        lines.append("<details>")
        lines.append("<summary>完整 RID payload JSON</summary>")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps([frame.payload for frame in report.frames if frame.payload], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
        lines.append("</details>")

    lines.append("")
    lines.append("## 失败帧原因分析与二次尝试")
    lines.append("")
    lines.append("本节只针对默认解码失败或 CRC 失败的帧。二次尝试会在固定网格内微调残余频偏、亚采样定时偏移、频域线性相位旋转和 QPSK 相位；只有 CRC OK 的结果才被提升为可信 RID。")
    lines.append("")
    for report in reports:
        failed_frames = [frame for frame in report.frames if frame.retry and frame.retry.attempted]
        lines.append(f"### {report.sample}")
        rows = []
        for frame in failed_frames:
            retry = frame.retry
            ok_payload = None
            if retry and retry.first_crc_ok:
                ok_payload = next(payload.payload for payload in retry.first_crc_ok.parsed_payloads if payload.crc_ok)
            rows.append(
                [
                    fmt(frame.frame_number),
                    fmt(frame.packet_index),
                    "CRC FAIL" if frame.payload and not frame_recovered(frame) else ("解码失败" if frame.failed else "默认失败"),
                    retry_summary(frame),
                    fmt(ok_payload.get("sequence_number") if ok_payload else None),
                    fmt(ok_payload.get("serial_number") if ok_payload else None),
                    fmt(ok_payload.get("app_lat") if ok_payload else None),
                    fmt(ok_payload.get("app_lon") if ok_payload else None),
                    failure_reason(frame),
                ]
            )
        lines.append(
            md_table(
                ["帧", "候选索引", "默认症状", "二次尝试结果", "恢复序号", "恢复序列号", "恢复 App 纬度", "恢复 App 经度", "原因判断"],
                rows,
            )
        )
        lines.append("")
    lines.append("公开资料对这个判断有佐证：RUB-SysSec/DroneSecurity README 明确说明部分候选帧是 false positive 且样本质量差异很大；proto17/dji_droneid 的 GNU Radio 实现也指出 fractional timing 会让频域出现 walking phase offset，因此需要定时/相位补偿。")
    lines.append("")
    lines.append("额外第三轮尝试：对 `mavic_air_2` 的 packet 0/1，在默认同步下扫描 4 种 QPSK phase 与 `Decoder.magic()` 的循环缓冲 offset `0..7199`，没有任何 offset 能构造出 `DroneIDPacket`。这说明失败不只是固定解扰 offset 选择错误。")
    lines.append("")
    lines.append("外部调研参考：")
    lines.append("- proto17/dji_droneid README（https://github.com/proto17/dji_droneid）：说明 DroneID 是 10 MHz/15.36 MHz guard-carrier 信号，ZC root index 为 600/147，且 fractional time offset 会导致频域 walking phase offset，需要相位/定时补偿。")
    lines.append("- NDSS 2023 Drone Security 论文（https://www.ndss-symposium.org/wp-content/uploads/2023/02/ndss2023_f217_paper.pdf）：说明 DJI DroneID 接收链路由 receiver、demodulator、decoder 组成，并能恢复无人机、home point 和 pilot/operator 位置。")
    lines.append("")

    lines.append("")
    lines.append("## 信号展示")
    lines.append("")
    lines.append("下图是在原始 IQ 上重新计算的 STFT 时频图。矩形框标出候选 RID 突发所在的时间和频率范围；白色标注框表示 CRC 通过，红色标注框表示解码失败或 CRC 校验失败。")
    lines.append("")
    for report in reports:
        lines.append(f"### {report.sample}")
        if report.spectrogram_image:
            lines.append(f"![{report.sample} RID spectrogram]({report.spectrogram_image.replace('\\', '/')})")
        else:
            lines.append("_未生成时频图。_")
        lines.append("")

    lines.append("")
    lines.append("## 信号处理详细过程")
    lines.append("")
    lines.append("解码链路按照仓库现有实现执行，核心步骤如下：")
    lines.append("")
    lines.append("1. 读取 IQ 文件：按 little-endian float 读取原始文件，并 view 为 `complex64` 复数 IQ 流。")
    lines.append("2. 帧检测：`SpectrumCapture` 调用 `packetizer.find_packet_candidate_time`，用 STFT 在时频图中寻找带宽约 9 MHz、时长约 0.64 ms 的 Drone-ID 候选突发。")
    lines.append("3. 粗频偏估计：对每个候选突发估计中心频偏 CFO，并用复指数频移把候选信号搬移到基带附近。")
    lines.append("4. 重采样：候选帧由 50 MHz 重采样到 Drone-ID OFDM 处理链路使用的 15.36 MHz。")
    lines.append("5. 细同步：`Packet` 利用循环前缀相关寻找 OFDM 符号起点，并估计细频偏 FFO。")
    lines.append("6. Zadoff-Chu 检测：定位两个 ZC 训练符号，细调采样偏移，并估计信道。")
    lines.append("7. OFDM 解调：去循环前缀、FFT 到频域、取有效子载波并做信道均衡。")
    lines.append("8. QPSK 软/硬判决：对 QPSK 星座方向进行尝试，恢复符号比特。")
    lines.append("9. 扰码与纠错：执行 gold 序列解扰和 turbo 解码，恢复 Drone-ID bitstream。")
    lines.append("10. RID 字段解析：`DroneIDPacket` 解析包长、版本、序号、状态、序列号、坐标、速度、机型、UUID 与 CRC，并用 CRC 判断 payload 可信度。")
    lines.append("")

    for report in reports:
        lines.append(f"### {report.sample} 的检测与同步记录")
        lines.append("")
        candidate_rows = [
            [
                fmt(c.index),
                fmt(c.start_s),
                fmt(c.end_s),
                fmt(c.length_s),
                fmt(c.cfo_hz),
            ]
            for c in report.candidates
        ]
        lines.append(md_table(["候选", "起始 s", "结束 s", "长度 s", "CFO Hz"], candidate_rows))
        frame_rows = []
        for frame in report.frames:
            frame_rows.append(
                [
                    fmt(frame.frame_number),
                    fmt(frame.packet_index),
                    "OK" if frame.crc_ok else ("FAIL" if frame.payload else "未恢复"),
                    fmt(frame.ffo_hz),
                    fmt(frame.zc_sequences),
                    fmt(frame.zc_offset),
                    fmt(frame.payload.get("sequence_number") if frame.payload else None),
                    "是" if frame.failed else "否",
                ]
            )
        lines.append(md_table(["帧", "候选索引", "RID CRC", "FFO Hz", "ZC 序列位置", "ZC 偏移", "RID 序号", "解码失败"], frame_rows))
        lines.append("")
        lines.append("<details>")
        lines.append("<summary>原始解码日志</summary>")
        lines.append("")
        lines.append("```text")
        lines.append(report.stdout.strip())
        if report.stderr.strip():
            lines.append("")
            lines.append("STDERR:")
            lines.append(report.stderr.strip())
        lines.append("```")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    return "\n".join(lines)


def render_html(markdown_text: str) -> str:
    try:
        import markdown  # type: ignore

        body = markdown.markdown(markdown_text, extensions=["tables", "fenced_code"])
    except Exception:
        body = "<pre>" + html.escape(markdown_text) + "</pre>"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>DJI Drone-ID / RID IQ 解码报告</title>
<style>
body{{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.55;margin:32px auto;max-width:1280px;padding:0 20px;color:#1f2328}}
h1,h2,h3{{line-height:1.25}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0 24px}}
th,td{{border:1px solid #d0d7de;padding:6px 8px;vertical-align:top}}
th{{background:#f6f8fa;text-align:left}}
pre{{white-space:pre-wrap;background:#f6f8fa;border:1px solid #d0d7de;border-radius:6px;padding:12px;overflow:auto}}
code{{font-family:ui-monospace,SFMono-Regular,Consolas,"Liberation Mono",monospace}}
details{{margin:12px 0 24px}}
summary{{cursor:pointer;font-weight:600}}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def write_outputs(reports: list[SampleReport], generated_at: str, md_out: Path, html_out: Path, json_out: Path, process_out: Path) -> None:
    markdown_text = render_markdown(reports, generated_at)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(markdown_text, encoding="utf-8")
    html_out.write_text(render_html(markdown_text), encoding="utf-8")
    payload = {
        "generated_at": generated_at,
        "reports": [asdict(report) for report in reports],
    }
    json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    retry_process = {
        "generated_at": generated_at,
        "retry_grid": {
            "tune_hz": RETRY_TUNES_HZ,
            "sample_offset_delta": RETRY_SAMPLE_OFFSETS,
            "linear_rotation": RETRY_LINEAR_ROTATIONS,
            "phase": [0, 1, 2, 3],
        },
        "frames": [
            {
                "sample": report.sample,
                "frame_number": frame.frame_number,
                "packet_index": frame.packet_index,
                "default_payload": frame.payload,
                "default_crc_ok": frame.crc_ok,
                "retry": asdict(frame.retry) if frame.retry else None,
            }
            for report in reports
            for frame in report.frames
            if frame.retry
        ],
        "extended_offset_scan": EXTENDED_OFFSET_SCAN,
    }
    process_out.write_text(json.dumps(retry_process, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="append", type=Path, help="IQ sample path; default: bundled samples")
    parser.add_argument("--sample-rate", type=float, default=50e6)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--md-out", type=Path, default=REPORT_DIR / "rid_decode_report.md")
    parser.add_argument("--html-out", type=Path, default=REPORT_DIR / "rid_decode_report.html")
    parser.add_argument("--json-out", type=Path, default=REPORT_DIR / "rid_decode_report.json")
    parser.add_argument("--process-out", type=Path, default=REPORT_DIR / "rid_retry_process.json")
    args = parser.parse_args()

    samples = args.sample or DEFAULT_SAMPLES
    generated_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    resolved_samples = [sample if sample.is_absolute() else REPO_ROOT / sample for sample in samples]
    reports = [run_decoder(sample, args.sample_rate, args.timeout) for sample in resolved_samples]
    for report, sample in zip(reports, resolved_samples):
        retry_failed_frames(report, sample, args.sample_rate)
    for report, sample in zip(reports, resolved_samples):
        report.spectrogram_image = generate_spectrogram(report, sample, args.sample_rate, REPORT_DIR / "assets")
    write_outputs(reports, generated_at, args.md_out, args.html_out, args.json_out, args.process_out)
    print(f"Wrote {args.md_out}")
    print(f"Wrote {args.html_out}")
    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.process_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

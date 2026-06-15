#!/usr/bin/env python3
"""Generate detailed signal-processing HTML reports for Drone-ID IQ samples."""

from __future__ import annotations

import contextlib
import html
import io
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
REPORTS_DIR = REPO_ROOT / "reports"
MPL_CACHE = REPO_ROOT / ".matplotlib-cache"
SAMPLE_RATE = 50e6
RESAMPLED_RATE = 15.36e6

SAMPLES = [
    REPO_ROOT / "samples" / "mini2_sm",
    REPO_ROOT / "samples" / "mavic_air_2",
]

KEY_PACKETS = {
    "mini2_sm": [0, 6, 7, 9],
    "mavic_air_2": [2, 0, 1],
}

STEP_DESCRIPTIONS = [
    {
        "name": "1. 原始 IQ 读取",
        "purpose": "将 little-endian float32 I/Q 交错数据恢复成 complex64 基带采样流。",
        "params": "dtype='<f'，view(complex64)，采样率 50 MHz。",
        "performance": "文件级读取适合离线分析；工程实现中建议使用环形缓冲和零拷贝 DMA/SDR buffer。",
        "optimization": "保留复数流式接口，避免重复 astype/view；长文件按窗口处理。",
    },
    {
        "name": "2. 粗帧检测",
        "purpose": "用短时频谱能量门限寻找约 0.64 ms、约 9 MHz 带宽的 Drone-ID 突发。",
        "params": "STFT nfft=64/nperseg=64；能量阈值为 1.15 倍平均噪声底；帧长窗口 630-665 us。",
        "performance": "可流式运行，成本低；当前 Python/SciPy 足够离线，实时工程建议固定窗口 FFT 批处理。",
        "optimization": "用滑动能量积分 + 粗 FFT bank 替代全量 STFT 绘图级计算。",
    },
    {
        "name": "3. 粗 CFO 估计与频移",
        "purpose": "用 Welch PSD 找到占用频带中心，把 9 MHz 信号搬移到基带附近。",
        "params": "Welch nfft=2048；选择 8-11 MHz 连续高能频带；复指数频移校正。",
        "performance": "单帧级 PSD 成本可接受；实时工程可用峰值/边缘跟踪减少 FFT 次数。",
        "optimization": "结合帧检测的频带边界缓存 CFO；用 NCO 批量复乘实现频移。",
    },
    {
        "name": "4. 重采样",
        "purpose": "将候选帧从 50 MHz 转换到 OFDM 解调链路使用的 15.36 MHz。",
        "params": "当前 helpers.resample 使用线性插值 np.interp。",
        "performance": "线性插值快但频域保真一般；工程上建议 polyphase FIR。",
        "optimization": "改为 scipy.signal.resample_poly 或硬件/SDR DSP 中的 polyphase resampler。",
    },
    {
        "name": "5. CP 相关细同步",
        "purpose": "利用循环前缀与符号尾部重复关系寻找 OFDM 起点，并估计细频偏 FFO。",
        "params": "NFFT=1024，首符号 CP=80；逐点相关并选 prominence > 1 的峰。",
        "performance": "当前 Python for 循环是慢点；每帧仍可离线跑完，但不适合高并发实时。",
        "optimization": "用向量化滑窗、FFT 卷积或 C/Numba 实现 CP 相关。",
    },
    {
        "name": "6. ZC 检测与信道估计",
        "purpose": "定位 Zadoff-Chu 同步符号，估计信道响应和采样偏移。",
        "params": "Drone-ID 使用 ZC root 600/147；find_zc_offset 在 -15..15 sample 内扫 1000 点。",
        "performance": "1000 点搜索是明显慢点；但提供了很好的离线诊断能力。",
        "optimization": "用 ZC 相位差线性拟合一次估计 fractional timing，替代密集搜索。",
    },
    {
        "name": "7. 采样偏移/线性相位纠偏",
        "purpose": "修正 fractional timing 造成的频域 walking phase offset。",
        "params": "本次 retry 成功参数为 tune=-250 Hz、sample_offset_delta=-2、linear_rotation=0.002。",
        "performance": "693 组网格搜索不适合工程实时；它证明了失败帧可通过同步微调恢复。",
        "optimization": "由 ZC phase slope 直接估计线性相位，再只做极小范围 CRC 验证。",
    },
    {
        "name": "8. OFDM/QPSK 解调",
        "purpose": "去 CP、FFT、信道均衡，硬判决 QPSK 子载波并尝试 4 种星座旋转。",
        "params": "跳过 ZC 符号 3/5；保留 7 个数据 OFDM 符号；QPSK phase=0..3。",
        "performance": "4 相位尝试成本很低，可保留在实时链路中。",
        "optimization": "用已知 Gold 序列或 CRC 早停，避免无效相位继续后处理。",
    },
    {
        "name": "9. 解扰、CRC 与 RID 解析",
        "purpose": "Gold 解扰、rate matching 反交织，解析 DroneIDPacket 并用 CRC 判断可信度。",
        "params": "Gold seed=0x12345678；Decoder.magic 固定循环 offset=4148。",
        "performance": "比前端 DSP 成本低；工程风险主要在同步误差传导到硬判决误码。",
        "optimization": "引入软判决/置信度，或在 CRC 失败时只做有依据的小范围同步回退。",
    },
]


@dataclass
class StepTrace:
    sample: str
    packet_index: int | None
    step: str
    elapsed_ms: float
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)


def setup_imports() -> None:
    os.environ["MPLCONFIGDIR"] = str(MPL_CACHE)
    MPL_CACHE.mkdir(parents=True, exist_ok=True)
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))


def quiet_call(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def rel(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_report_sample(report_data: dict[str, Any], sample_name: str) -> dict[str, Any]:
    for report in report_data["reports"]:
        if Path(report["sample"]).name == sample_name:
            return report
    raise KeyError(sample_name)


def find_frame(report: dict[str, Any], packet_index: int) -> dict[str, Any] | None:
    for frame in report["frames"]:
        if frame.get("packet_index") == packet_index:
            return frame
    return None


def retry_for(retry_data: dict[str, Any], sample_name: str, packet_index: int) -> dict[str, Any] | None:
    for item in retry_data.get("frames", []):
        if Path(item["sample"]).name == sample_name and item["packet_index"] == packet_index:
            return item
    return None


def record(trace: list[StepTrace], sample: str, packet: int | None, step: str, start: float, metrics: dict[str, Any], artifacts: list[Path], out_dir: Path) -> None:
    trace.append(
        StepTrace(
            sample=sample,
            packet_index=packet,
            step=step,
            elapsed_ms=(time.perf_counter() - start) * 1000.0,
            metrics=metrics,
            artifacts=[rel(path, out_dir) for path in artifacts],
        )
    )


def decimate(values: np.ndarray, limit: int = 120_000) -> np.ndarray:
    if len(values) <= limit:
        return values
    return values[:: max(1, len(values) // limit)]


def plot_raw_overview(raw: np.ndarray, sample_name: str, assets: Path, out_dir: Path) -> list[Path]:
    t = np.arange(len(raw)) / SAMPLE_RATE * 1e3
    idx = np.linspace(0, len(raw) - 1, min(25_000, len(raw))).astype(int)
    artifacts: list[Path] = []

    path = assets / f"{sample_name}_raw_amplitude_phase.png"
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    axes[0].plot(t[idx], np.abs(raw[idx]), linewidth=0.6)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"{sample_name} raw IQ amplitude")
    axes[1].plot(t[idx], np.angle(raw[idx]), linewidth=0.4)
    axes[1].set_ylabel("Phase (rad)")
    axes[1].set_xlabel("Time (ms)")
    savefig(path)
    artifacts.append(path)

    path = assets / f"{sample_name}_raw_iq_scatter.png"
    plt.figure(figsize=(6, 6))
    plt.scatter(raw[idx].real, raw[idx].imag, s=1, alpha=0.25)
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.title(f"{sample_name} raw IQ scatter")
    plt.axis("equal")
    savefig(path)
    artifacts.append(path)

    path = assets / f"{sample_name}_full_spectrogram.png"
    plot_spectrogram(raw, SAMPLE_RATE, path, f"{sample_name} full spectrogram")
    artifacts.append(path)
    return artifacts


def plot_spectrogram(data: np.ndarray, fs: float, path: Path, title: str, boxes: list[dict[str, float]] | None = None) -> None:
    freqs, times, zxx = signal.stft(
        data,
        fs=fs,
        window="hann",
        nperseg=512 if len(data) > 2048 else 128,
        noverlap=384 if len(data) > 2048 else 96,
        nfft=1024 if len(data) > 2048 else 256,
        return_onesided=False,
        boundary=None,
        padded=False,
    )
    order = np.argsort(freqs)
    pwr = 20 * np.log10(np.abs(zxx[order]) + 1e-12)
    vmax = float(np.percentile(pwr, 99.5))
    vmin = vmax - 70
    plt.figure(figsize=(13, 6))
    plt.pcolormesh(times * 1e3, freqs[order] / 1e6, pwr, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    plt.colorbar(label="Power (dB)")
    if boxes:
        ax = plt.gca()
        for box in boxes:
            rect = plt.Rectangle(
                (box["start_ms"], box["center_mhz"] - box["bw_mhz"] / 2),
                box["width_ms"],
                box["bw_mhz"],
                fill=False,
                edgecolor=box.get("color", "white"),
                linewidth=1.8,
            )
            ax.add_patch(rect)
            ax.text(box["start_ms"], box["center_mhz"] + box["bw_mhz"] / 2 + 0.5, box.get("label", ""), color="white", fontsize=8, bbox={"facecolor": "black", "alpha": 0.6})
    plt.title(title)
    plt.xlabel("Time (ms)")
    plt.ylabel("Relative frequency (MHz)")
    savefig(path)


def plot_psd(before: np.ndarray, after: np.ndarray, fs: float, cfo_hz: float, path: Path, title: str) -> None:
    f0, p0 = signal.welch(before, fs, nfft=2048, return_onesided=False)
    f1, p1 = signal.welch(after, fs, nfft=2048, return_onesided=False)
    order0 = np.argsort(f0)
    order1 = np.argsort(f1)
    plt.figure(figsize=(12, 5))
    plt.plot(f0[order0] / 1e6, 10 * np.log10(p0[order0] + 1e-20), label="before CFO")
    plt.plot(f1[order1] / 1e6, 10 * np.log10(p1[order1] + 1e-20), label="after CFO")
    plt.axvline(cfo_hz / 1e6, color="r", linestyle="--", linewidth=1, label=f"CFO {cfo_hz/1e6:.3f} MHz")
    plt.xlabel("Relative frequency (MHz)")
    plt.ylabel("PSD (dB)")
    plt.title(title)
    plt.legend()
    savefig(path)


def cp_correlation(samples: np.ndarray, cp_lengths: list[int], nfft: int, fs: float) -> tuple[np.ndarray, np.ndarray, int, float]:
    cpl = cp_lengths[0]
    res = []
    for n in range(nfft, len(samples) - cpl):
        res.append(np.sum(samples[n:n + cpl] * np.conj(samples[n - nfft:n - nfft + cpl])))
    res = np.asarray(res)
    res_abs = np.abs(res)
    peaks, _ = signal.find_peaks(res_abs, distance=1000)
    prominences, _, _ = signal.peak_prominences(res_abs, peaks)
    if len(peaks):
        strong = np.where(prominences > 1.0)[0]
        if len(strong):
            selected = peaks[strong[0]]
        else:
            selected = peaks[int(np.argmax(prominences))]
    else:
        selected = int(np.argmax(res_abs)) if len(res_abs) else 0
    ffo = fs / (2 * np.pi * nfft) * np.angle(res[selected]) if len(res) else 0.0
    return res_abs, peaks, int(selected), float(ffo)


def plot_cp(res_abs: np.ndarray, peaks: np.ndarray, selected: int, path: Path, title: str) -> None:
    plt.figure(figsize=(12, 4))
    x = np.arange(len(res_abs))
    plt.plot(x, res_abs, linewidth=0.7)
    if len(peaks):
        plt.scatter(peaks, res_abs[peaks], s=18, marker="x", label="peaks")
    plt.axvline(selected, color="r", linestyle="--", label=f"selected {selected}")
    plt.xlabel("Sample index")
    plt.ylabel("|CP correlation|")
    plt.title(title)
    plt.legend()
    savefig(path)


def zc_phase_slope(packet: Any, symbol_idx: int, seq: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    from zcsequence import zcsequence_t
    from helpers import NCARRIERS

    ref = zcsequence_t(seq, NCARRIERS)
    rx = packet.symbols_freq_domain[symbol_idx].copy()
    rx[rx == 0] = 1
    phase = np.unwrap(np.angle(ref / rx))
    phase[NCARRIERS // 2] = phase[NCARRIERS // 2 + 1]
    x = np.arange(len(phase))
    slope, intercept = np.polyfit(x, phase, 1)
    return x, phase, float(slope), float(intercept)


def plot_zc_phase(x: np.ndarray, phase: np.ndarray, slope: float, intercept: float, path: Path, title: str) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(x, phase, linewidth=0.8, label="unwrapped phase error")
    plt.plot(x, slope * x + intercept, "--", label=f"linear fit slope={slope:.4g}")
    plt.xlabel("Subcarrier index")
    plt.ylabel("Phase error (rad)")
    plt.title(title)
    plt.legend()
    savefig(path)


def flatten_symbols(symbols: list[np.ndarray]) -> np.ndarray:
    if not symbols:
        return np.array([], dtype=np.complex64)
    return np.concatenate([np.asarray(sym) for sym in symbols])


def plot_constellation(default_symbols: list[np.ndarray], retry_symbols: list[np.ndarray] | None, path: Path, title: str) -> None:
    default = flatten_symbols(default_symbols)
    retry = flatten_symbols(retry_symbols or [])
    fig, axes = plt.subplots(1, 2 if len(retry) else 1, figsize=(12 if len(retry) else 6, 5))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes[0].scatter(default.real, default.imag, s=2, alpha=0.25)
    axes[0].set_title("default")
    axes[0].set_xlabel("I")
    axes[0].set_ylabel("Q")
    axes[0].axis("equal")
    if len(retry):
        axes[1].scatter(retry.real, retry.imag, s=2, alpha=0.25, color="C1")
        axes[1].set_title("retry/recovered")
        axes[1].set_xlabel("I")
        axes[1].set_ylabel("Q")
        axes[1].axis("equal")
    fig.suptitle(title)
    savefig(path)


def trace_selected_packets(sample_path: Path, report: dict[str, Any], retry_data: dict[str, Any], assets: Path, out_dir: Path, trace: list[StepTrace]) -> list[dict[str, Any]]:
    from SpectrumCapture import SpectrumCapture
    from Packet import Packet
    from helpers import CP_LENGTHS, NFFT, estimate_offset, fshift, resample

    sample_name = sample_path.name
    start = time.perf_counter()
    raw = np.fromfile(sample_path, dtype="<f").astype(np.float32).view(np.complex64)
    overview_artifacts = plot_raw_overview(raw, sample_name, assets, out_dir)
    record(trace, sample_name, None, "raw_iq_read", start, {"samples": int(len(raw)), "duration_ms": len(raw) / SAMPLE_RATE * 1e3}, overview_artifacts, out_dir)

    start = time.perf_counter()
    capture = quiet_call(SpectrumCapture, raw, Fs=SAMPLE_RATE)
    boxes = []
    for candidate in report.get("candidates", []):
        frame = find_frame(report, candidate["index"])
        label = f"pkt {candidate['index']}"
        color = "white" if frame and frame.get("payload") and frame.get("crc_ok") else "#ff4040"
        boxes.append(
            {
                "start_ms": candidate["start_s"] * 1e3,
                "width_ms": (candidate["end_s"] - candidate["start_s"]) * 1e3,
                "center_mhz": candidate["cfo_hz"] / 1e6,
                "bw_mhz": 10.0,
                "label": label,
                "color": color,
            }
        )
    path = assets / f"{sample_name}_detected_candidates.png"
    plot_spectrogram(raw, SAMPLE_RATE, path, f"{sample_name} detected candidates", boxes)
    record(trace, sample_name, None, "coarse_frame_detection", start, {"candidate_count": len(capture.packets)}, [path], out_dir)

    packet_summaries: list[dict[str, Any]] = []
    for packet_index in KEY_PACKETS[sample_name]:
        frame = find_frame(report, packet_index) or {}
        retry = retry_for(retry_data, sample_name, packet_index)
        packet_data = capture.packets[packet_index].copy()
        packet_prefix = f"{sample_name}_pkt{packet_index}"

        packet_info: dict[str, Any] = {
            "sample": sample_name,
            "packet_index": packet_index,
            "frame": frame,
            "retry": retry,
            "artifacts": [],
        }

        start = time.perf_counter()
        path = assets / f"{packet_prefix}_candidate_spectrogram.png"
        plot_spectrogram(packet_data, SAMPLE_RATE, path, f"{packet_prefix} raw candidate spectrogram")
        iq_path = assets / f"{packet_prefix}_raw_iq_scatter.png"
        idx = np.linspace(0, len(packet_data) - 1, min(15_000, len(packet_data))).astype(int)
        plt.figure(figsize=(6, 6))
        plt.scatter(packet_data[idx].real, packet_data[idx].imag, s=2, alpha=0.25)
        plt.xlabel("I")
        plt.ylabel("Q")
        plt.title(f"{packet_prefix} raw candidate IQ")
        plt.axis("equal")
        savefig(iq_path)
        record(trace, sample_name, packet_index, "candidate_view", start, {"candidate_samples": int(len(packet_data))}, [path, iq_path], out_dir)
        packet_info["artifacts"].extend([rel(path, out_dir), rel(iq_path, out_dir)])

        start = time.perf_counter()
        cfo_hz, cfo_ok = quiet_call(estimate_offset, packet_data, SAMPLE_RATE)
        shifted = quiet_call(fshift, packet_data.copy(), -1.0 * cfo_hz, SAMPLE_RATE) if cfo_ok else packet_data.copy()
        psd_path = assets / f"{packet_prefix}_psd_cfo_before_after.png"
        plot_psd(packet_data, shifted, SAMPLE_RATE, cfo_hz or 0.0, psd_path, f"{packet_prefix} PSD before/after CFO")
        record(trace, sample_name, packet_index, "coarse_cfo", start, {"cfo_hz": cfo_hz, "success": cfo_ok}, [psd_path], out_dir)
        packet_info["artifacts"].append(rel(psd_path, out_dir))

        start = time.perf_counter()
        resampled = resample(shifted, SAMPLE_RATE, RESAMPLED_RATE)
        resampled_path = assets / f"{packet_prefix}_resampled_spectrogram.png"
        plot_spectrogram(resampled, RESAMPLED_RATE, resampled_path, f"{packet_prefix} after CFO + resample")
        record(trace, sample_name, packet_index, "resample", start, {"before_samples": int(len(shifted)), "after_samples": int(len(resampled)), "from_hz": SAMPLE_RATE, "to_hz": RESAMPLED_RATE}, [resampled_path], out_dir)
        packet_info["artifacts"].append(rel(resampled_path, out_dir))

        start = time.perf_counter()
        cp, peaks, selected, ffo = cp_correlation(resampled, CP_LENGTHS, NFFT, RESAMPLED_RATE)
        cp_path = assets / f"{packet_prefix}_cp_correlation.png"
        plot_cp(cp, peaks, selected, cp_path, f"{packet_prefix} CP correlation")
        record(trace, sample_name, packet_index, "cp_fine_sync", start, {"selected_start": selected, "ffo_hz": ffo, "peak_count": int(len(peaks))}, [cp_path], out_dir)
        packet_info["artifacts"].append(rel(cp_path, out_dir))

        try:
            start = time.perf_counter()
            packet = quiet_call(Packet, resampled, enable_zc_detection=True)
            x, zc_phase, slope, intercept = zc_phase_slope(packet, packet.ZC_SYMBOL_IDX[0], 600)
            zc_path = assets / f"{packet_prefix}_zc_phase_slope.png"
            plot_zc_phase(x, zc_phase, slope, intercept, zc_path, f"{packet_prefix} ZC phase slope")
            record(
                trace,
                sample_name,
                packet_index,
                "zc_phase_channel",
                start,
                {
                    "detected_ffo_hz": float(packet.detected_ffo),
                    "sampling_offset": float(packet.sampling_offset),
                    "zc_phase_slope": slope,
                    "zc_phase_intercept": intercept,
                },
                [zc_path],
                out_dir,
            )
            packet_info["artifacts"].append(rel(zc_path, out_dir))

            start = time.perf_counter()
            default_symbols = quiet_call(packet.get_symbol_data, skip_zc=True)
            retry_symbols = None
            retry_params = None
            if retry and retry.get("retry", {}).get("first_crc_ok"):
                first = retry["retry"]["first_crc_ok"]
                retry_params = {
                    "tune_hz": first["tune_hz"],
                    "sample_offset_delta": first["sample_offset_delta"],
                    "linear_rotation": first["linear_rotation"],
                }
                retry_symbols = quiet_call(
                    packet.get_symbol_data,
                    skip_zc=True,
                    tune=first["tune_hz"],
                    _sampling_offset=first["sample_offset_delta"],
                    linear_rotation=first["linear_rotation"],
                )
            const_path = assets / f"{packet_prefix}_constellation_default_retry.png"
            plot_constellation(default_symbols, retry_symbols, const_path, f"{packet_prefix} QPSK constellation")
            record(trace, sample_name, packet_index, "ofdm_qpsk", start, {"retry_params": retry_params, "data_symbol_count": len(default_symbols), "subcarriers_per_symbol": int(len(default_symbols[0])) if default_symbols else 0}, [const_path], out_dir)
            packet_info["artifacts"].append(rel(const_path, out_dir))
        except Exception as exc:
            record(trace, sample_name, packet_index, "packet_analysis_error", time.perf_counter(), {"error": f"{type(exc).__name__}: {exc}"}, [], out_dir)

        packet_summaries.append(packet_info)
    return packet_summaries


def html_page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body{{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.6;margin:32px auto;max-width:1320px;padding:0 22px;color:#1f2328}}
h1,h2,h3{{line-height:1.25}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0 24px}}
th,td{{border:1px solid #d0d7de;padding:6px 8px;vertical-align:top}}
th{{background:#f6f8fa;text-align:left}}
img{{max-width:100%;height:auto;border:1px solid #d0d7de;border-radius:4px;background:#fff}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}}
.card{{border:1px solid #d0d7de;border-radius:6px;padding:12px;margin:12px 0;background:#fff}}
.note{{background:#f6f8fa;border-left:4px solid #0969da;padding:10px 12px}}
code{{font-family:ui-monospace,SFMono-Regular,Consolas,"Liberation Mono",monospace}}
pre{{white-space:pre-wrap;background:#f6f8fa;border:1px solid #d0d7de;border-radius:6px;padding:12px;overflow:auto}}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def table(headers: list[str], rows: list[list[Any]]) -> str:
    parts = ["<table><thead><tr>"]
    parts.extend(f"<th>{html.escape(str(header))}</th>" for header in headers)
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        parts.extend(f"<td>{html.escape(str(cell))}</td>" for cell in row)
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def render_index(out_dir: Path, trace: list[StepTrace], packet_summaries: list[dict[str, Any]], report_data: dict[str, Any]) -> str:
    overview_rows = []
    for report in report_data["reports"]:
        ok = sum(1 for frame in report["frames"] if frame.get("payload") and frame.get("crc_ok"))
        retry_ok = sum(1 for frame in report["frames"] if frame.get("retry") and frame["retry"].get("first_crc_ok"))
        overview_rows.append([report["sample"], len(report["frames"]), ok, retry_ok])

    step_rows = [[step["name"], step["purpose"], step["params"], step["performance"], step["optimization"]] for step in STEP_DESCRIPTIONS]
    timing_rows = [[item.sample, "" if item.packet_index is None else item.packet_index, item.step, f"{item.elapsed_ms:.2f}", json.dumps(item.metrics, ensure_ascii=False)[:220]] for item in trace]

    visual_cards = []
    for sample_name in [path.name for path in SAMPLES]:
        imgs = [item for item in trace if item.sample == sample_name and item.packet_index is None]
        for item in imgs:
            for artifact in item.artifacts:
                visual_cards.append(f"<div class='card'><h3>{html.escape(sample_name)} / {html.escape(item.step)}</h3><img src='{html.escape(artifact)}'></div>")

    body = f"""
<h1>DJI Drone-ID 精细化信号处理流程报告</h1>
<p class="note">本报告重新计算关键中间信号图，覆盖原始 IQ、帧检测、CFO、重采样、CP 同步、ZC 相位、QPSK 星座与 RID/CRC 结果。频率均为相对采样中心频率。</p>
<p><a href="frames.html">查看逐帧细节 frames.html</a> | <a href="data/processing_trace.json">下载 processing_trace.json</a></p>
<h2>解码概览</h2>
{table(["样本", "候选帧", "最终 CRC OK", "retry recovered"], overview_rows)}
<h2>完整信号处理流程、参数与工程评价</h2>
{table(["步骤", "目的", "参数设置", "速度/工程适配性", "下一步优化"], step_rows)}
<h2>样本级中间信号图</h2>
<div class="grid">{''.join(visual_cards)}</div>
<h2>处理耗时与关键指标</h2>
<p class="note">这里的耗时是报告生成路径的端到端耗时，包含 PNG 绘图和 Python instrumentation 开销；实时工程评估应剥离绘图，并用向量化或原生实现重新 profile。</p>
{table(["样本", "packet", "步骤", "耗时 ms", "关键指标"], timing_rows)}
<h2>工程优化建议</h2>
<ul>
<li>用 ZC 相位差 unwrap + 线性拟合估计 fractional timing 和频域线性相位，替代 693 组 retry 网格。</li>
<li>用相邻 ZC/OFDM 符号公共相位漂移估计 residual CFO，将频偏搜索收敛到 ±100 Hz 以内。</li>
<li>将 CP 相关改为向量化滑窗或 FFT 卷积；将重采样改为 polyphase FIR。</li>
<li>保留 4 种 QPSK phase 尝试，因为成本极低；用 CRC 或 Gold 序列早停。</li>
<li>高分辨率绘图只作为离线报告，不进入实时链路。</li>
</ul>
"""
    return html_page("DJI Drone-ID 精细化信号处理流程报告", body)


def render_frames(out_dir: Path, packet_summaries: list[dict[str, Any]]) -> str:
    cards = []
    for packet in packet_summaries:
        frame = packet.get("frame") or {}
        retry = packet.get("retry")
        payload = frame.get("payload") or {}
        status = "CRC OK" if frame.get("payload") and frame.get("crc_ok") else "未恢复"
        if retry and retry.get("retry", {}).get("first_crc_ok"):
            status = "retry recovered CRC OK"
        retry_text = json.dumps(retry.get("retry", {}).get("first_crc_ok"), ensure_ascii=False, indent=2) if retry else "无 retry"
        images = "".join(f"<div class='card'><img src='{html.escape(path)}'></div>" for path in packet["artifacts"])
        cards.append(
            f"""
<section class="card">
<h2>{html.escape(packet['sample'])} packet {packet['packet_index']} - {html.escape(status)}</h2>
<p><strong>RID:</strong> seq={html.escape(str(payload.get('sequence_number', '')))}，
serial={html.escape(str(payload.get('serial_number', '')))}，
device={html.escape(str(payload.get('device_type', '')))}，
app=({html.escape(str(payload.get('app_lat', '')))}, {html.escape(str(payload.get('app_lon', '')))})</p>
<h3>Retry / failure evidence</h3>
<pre>{html.escape(retry_text)}</pre>
<div class="grid">{images}</div>
</section>
"""
        )
    body = f"""
<h1>逐帧信号处理中间过程</h1>
<p><a href="index.html">返回主报告</a></p>
{''.join(cards)}
"""
    return html_page("逐帧信号处理中间过程", body)


def main() -> int:
    setup_imports()
    report_data = load_json(REPORTS_DIR / "rid_decode_report.json")
    retry_data = load_json(REPORTS_DIR / "rid_retry_process.json")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPORTS_DIR / f"{stamp}_signal_processing_detail"
    assets = out_dir / "assets"
    data_dir = out_dir / "data"
    assets.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    trace: list[StepTrace] = []
    packet_summaries: list[dict[str, Any]] = []
    sample_overview: list[dict[str, Any]] = []
    for sample in SAMPLES:
        report = find_report_sample(report_data, sample.name)
        retry_recovered = sum(1 for frame in report["frames"] if frame.get("retry") and frame["retry"].get("first_crc_ok"))
        crc_ok = sum(1 for frame in report["frames"] if frame.get("payload") and frame.get("crc_ok"))
        unrecovered = sum(
            1
            for frame in report["frames"]
            if frame.get("failed") and not (frame.get("retry") and frame["retry"].get("first_crc_ok"))
        )
        sample_overview.append(
            {
                "sample": report["sample"],
                "candidates": len(report["frames"]),
                "final_crc_ok": crc_ok,
                "retry_recovered": retry_recovered,
                "unrecovered": unrecovered,
                "key_packets": KEY_PACKETS.get(sample.name, []),
            }
        )
        packet_summaries.extend(trace_selected_packets(sample, report, retry_data, assets, out_dir, trace))

    trace_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(out_dir),
        "samples": [str(path.relative_to(REPO_ROOT)) for path in SAMPLES],
        "sample_overview": sample_overview,
        "key_packets": KEY_PACKETS,
        "step_descriptions": STEP_DESCRIPTIONS,
        "trace": [asdict(item) for item in trace],
        "packet_summaries": packet_summaries,
    }
    (data_dir / "processing_trace.json").write_text(json.dumps(trace_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "index.html").write_text(render_index(out_dir, trace, packet_summaries, report_data), encoding="utf-8")
    (out_dir / "frames.html").write_text(render_frames(out_dir, packet_summaries), encoding="utf-8")
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

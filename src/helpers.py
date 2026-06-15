import numpy as np
import scipy.signal as signal
from fractions import Fraction
import matplotlib.pyplot as plt

# Drone-ID 使用中心 601 个有效子载波。
NCARRIERS = 601
NCARRIERS_c2 = 73
MAXNCARRIERS = NCARRIERS
MAXNCARRIERS_c2 = NCARRIERS_c2
NFFT = 1024  # OFDM FFT 点数

# 循环前缀长度: 第一个 OFDM 符号较长, 其余符号较短, 形式接近 LTE normal CP。
CP_LENGTHS = [
    72 + 8,  # 符号 0
    72,  # 符号 1
    72,  # 符号 2
    72,  # 符号 3
    72,  # 符号 4
    72,  # 符号 5
    72,  # 符号 6
    72,  # 符号 7
    72 + 8,  # 符号 8
]

CP_LENGTHS_legacy = [
    72 + 8,  # 符号 0
    72,  # 符号 1
    72,  # 符号 2
    72,  # 符号 3
    72,  # 符号 4
    72,  # 符号 5
    72,  # 符号 6
    72 + 8,  # 符号 7
]

CP_LENGTHS_C2 = [
    72 + 8,  # 符号 0
    72,  # 符号 1
    72,  # 符号 2
    72,  # 符号 3
    72,  # 符号 4
    72,  # 符号 5
    72 + 8,  # 符号 6
]

# 这些 OFDM 符号承载 Zadoff-Chu 同步序列。
ZC_SYMBOL_IDX = [3, 5]
ZC_SYMBOL_IDX_legacy = [2, 4]
ZC_SYMBOL_IDX_c2 = [0, 6]


def corr(x, y=None):
    """返回从零延迟开始的一侧相关结果。"""
    if y is None:
        y = x
    result = np.correlate(x, y, mode="full")
    return result[result.size//2:]


def fshift(y, offset, Fs):
    """通过乘复指数把信号频谱平移 offset Hz。"""
    print(f"{len(y)}, offset={offset}, Fs={Fs}")
    x = np.linspace(0.0, len(y)/Fs, len(y))
    return y * np.exp(x * 2j * np.pi * offset)


def fshift_rad(y, offset, Fs):
    """按弧度形式的频率偏移做频移, 主要用于调试/实验。"""
    x = np.linspace(0.0, len(y)/Fs, len(y))
    return y * np.exp(x * 1j * np.pi * offset)


def with_sample_offset(data, offset):
    """用线性插值实现非整数采样点偏移。"""
    return np.interp(np.arange(offset, offset+len(data), 1), np.arange(0, len(data)), data)


def resample(pkt_fullrate, Fs: float, Fsnew: float):
    """用线性插值把采样流从 Fs 重采样到 Fsnew。"""
    # 这里保留了几种曾尝试的 scipy 重采样方法, 当前实现采用插值。
    # fr = Fraction(int(Fsnew), int(Fs)).limit_denominator(1000)
    # return signal.resample_poly(pkt_fullrate, fr.numerator, fr.denominator)
    # return signal.resample(pkt_fullrate, int(len(pkt_fullrate)/Fs*Fsnew))

    return np.interp(np.arange(0, len(pkt_fullrate), Fs/Fsnew),
            np.arange(0, len(pkt_fullrate)), pkt_fullrate)


def consecutive(data, stepsize=1):
    """把连续递增的索引分组。"""
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


def tfft(sy):
    """对一个去 CP 后的 OFDM 符号做 FFT, 并取中心有效子载波。"""
    fft = np.fft.fft(sy, n=NFFT)
    half_carriers = NCARRIERS//2
    new_fft = np.concatenate((fft[-half_carriers:], fft[:half_carriers+1]))
    return new_fft


def itfft(c):
    """把中心有效子载波放回 1024 点频域栅格并 IFFT 回时域。"""
    half_carriers = NCARRIERS//2
    c_full = np.zeros((NFFT), dtype=np.complex64)
    c_full[-half_carriers:] = c[:half_carriers]
    c_full[:half_carriers+1] = c[half_carriers:]

    return np.fft.ifft(c_full)


def estimate_offset(y, Fs, debug=False, packet_type="droneid"):
    """从功率谱密度中估计候选帧的中心频率偏移。"""
    nfft_welch = 2048

    if len(y) < nfft_welch:
        return None, False

    # 用 Welch 方法估计功率谱密度。
    f, Pxx_den = signal.welch(
        y, Fs, nfft=nfft_welch, return_onesided=False)

    Pxx_den = np.fft.fftshift(Pxx_den)
    f = np.fft.fftshift(f)

    if debug:
        # 调试时画出功率谱密度和平均噪声底。
        plt.semilogy(f, Pxx_den)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.plot(f, [Pxx_den.mean(), ]*len(f))
        plt.show()
        # plt.plot(Pxx_den > Pxx_den.mean())

    # 抬高 DC 附近的功率, 避免真实信号被 DC 缺口切成两个候选频带。
    Pxx_den[nfft_welch//2-10:nfft_welch//2+10] = 1.1*Pxx_den.mean()

    # 找出功率高于平均值的连续 FFT bin, 作为候选占用频带。
    candidate_bands = consecutive(np.where(Pxx_den > Pxx_den.mean())[0])

    band_found = False
    offset = 0.0

    for band in candidate_bands:
        start = band[0]-nfft_welch/2
        end = band[-1]-nfft_welch/2

        bw = (end - start) * (Fs/nfft_welch)

        fend = start * Fs/nfft_welch
        fstart = end * Fs/nfft_welch

        if debug:
            print("candidate band fstart: %3.2f, fend: %3.2f, bw: %3.2f MHz" % (fstart, fend, bw/1e6))
        print("candidate band fstart: %3.2f, fend: %3.2f, bw: %3.2f MHz" % (fstart, fend, bw/1e6))

        # 不同包类型具有不同占用带宽, 据此筛掉误检频带。
        if packet_type == "droneid" and (bw > 8e6 and bw < 11e6):
            offset = fstart - 0.5*bw
            band_found = True
            break
        elif packet_type == "c2" and (bw > 1.2e6 and bw < 1.95e6):
            offset = fstart - 0.5*bw
            band_found = True
            break
        elif packet_type == "video" and (bw > 18e6 and bw < 22e6):
            # 注意: 如果同时存在多个 Drone-ID 广播, 这种简单带宽筛选可能失效。
            # 这里得到的是相对当前采样中心的频偏, 不是绝对射频频点。
            offset = fstart - 0.5*bw
            band_found = True
            break

    if debug:
        print("Offset found: %.2fkHz" % (offset/1000))
    return offset, band_found

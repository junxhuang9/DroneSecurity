#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import signal
from zcsequence import zcsequence_f, zcsequence_t
from helpers import corr, fshift, tfft, itfft, with_sample_offset, NFFT, MAXNCARRIERS, NCARRIERS, MAXNCARRIERS_c2, NCARRIERS_c2, CP_LENGTHS_legacy, ZC_SYMBOL_IDX_legacy, CP_LENGTHS, CP_LENGTHS_C2, ZC_SYMBOL_IDX, ZC_SYMBOL_IDX_c2


class Packet:
    """把单个 Drone-ID 时域帧解调为频域 QPSK 数据。"""
    def __init__(self, raw_samples, Fs=15.36e6, enable_zc_detection=True, debug=False, legacy = False, packet_type = "droneid"):
        self.debug = debug
        self.NCARRIERS = NCARRIERS
        self.MAXNCARRIERS = MAXNCARRIERS

        if legacy and packet_type == "droneid":
            self.CP_LENGTHS = CP_LENGTHS_legacy
            self.ZC_SYMBOL_IDX = ZC_SYMBOL_IDX_legacy
        elif packet_type == "droneid":
            self.CP_LENGTHS = CP_LENGTHS
            self.ZC_SYMBOL_IDX = ZC_SYMBOL_IDX
        elif packet_type == "c2":
            self.CP_LENGTHS = CP_LENGTHS_C2
            self.ZC_SYMBOL_IDX = ZC_SYMBOL_IDX_c2
            self.NCARRIERS = NCARRIERS_c2
            self.MAXNCARRIERS = MAXNCARRIERS_c2          

        # 当前解调链路假定输入已经重采样到 15.36 MHz。
        self.Fs = Fs

        # 细频偏：由循环前缀相关相位估计得到。
        self.detected_ffo = 0

        # 第一个 OFDM 符号在当前帧内的起始采样点。
        self.start = 0

        # 幅度归一化只影响数值尺度，不改变星座相位关系。
        self.raw_samples = raw_samples
        self.raw_samples /= np.max(np.abs(raw_samples))

        # 保留归一化后的原始采样，后面会用不同的偏移/相位参数重新切符号。
        self.raw_samples_orig = self.raw_samples

        # 利用循环前缀和符号尾部的重复关系，估计第一个符号起点和细频偏。
        self.start, self.detected_ffo = self.find_fine_start(self.raw_samples)
        if self.debug:
            print("First Symbol at Sample %i, FFO %f" % (self.start, self.detected_ffo))

        # 第一次粗切 OFDM 符号，用于寻找 Zadoff-Chu 参考序列。
        self.symbols_time_domain, self.symbols_freq_domain = self.raw_data_to_symbols(self.raw_samples_orig, self.start, ffo=self.detected_ffo)

        if enable_zc_detection:
            # 检查两个同步符号里出现的 ZC 根序列，确认当前候选帧确实像 Drone-ID。
            zc_seq_1 = self.find_zc_seq(self.symbols_freq_domain[self.ZC_SYMBOL_IDX[0]])
            zc_seq_2 = self.find_zc_seq(self.symbols_freq_domain[self.ZC_SYMBOL_IDX[1]])
        else:
            zc_seq_1 = 600
            zc_seq_2 = 147

        # 第一个 ZC 序列用于粗同步，可能变化；第二个 ZC 用于细同步，Drone-ID 中应为 147。
        if not (zc_seq_2 == 147) and packet_type == "droneid":
            # print("未找到 ZC 序列。期望: 600 和 147, 实际: %i 和 %i" % (zc_seq_1, zc_seq_2))
            raise ValueError("ZC Sequence not found. Expected: 600 and 147, Found: %i and %i" % (zc_seq_1, zc_seq_2))

        print("Found ZC sequences:",zc_seq_1, zc_seq_2)
        # 用两个 ZC 参考符号分别估计频域信道响应，再取平均作为均衡器。
        self.channel = self.estimate_channel(self.ZC_SYMBOL_IDX[0], zc_seq_1)
        self.channel += self.estimate_channel(self.ZC_SYMBOL_IDX[1], zc_seq_2)
        self.channel *= 0.5

        # zc_cyc = self.find_zc_shift(self.symbol_equalized(ZC_SYMBOL_IDX[0], self.channel), 600)
        # print("ZC 循环移位: %i" % zc_cyc)

        zc_cyc = 0

        # 通过 ZC 相位斜率搜索亚采样级定时偏移；当前实现只用第一个 ZC 做该校正。
        self.sampling_offset = self.find_zc_offset(self.ZC_SYMBOL_IDX[0], 600, zc_cyc)
        print("ZC Offset: %f" % self.sampling_offset)

        # 带上亚采样定时偏移后重新切符号。
        self.symbols_time_domain, self.symbols_freq_domain = self.raw_data_to_symbols(self.raw_samples, self.start, ffo=self.detected_ffo, sampling_offset=self.sampling_offset)

        # 再根据 ZC 符号估计整体相位旋转，并第三次切符号得到最终星座。
        angle = self.find_zc_angle(self.symbols_freq_domain[self.ZC_SYMBOL_IDX[0]], 600)

        self.symbols_time_domain, self.symbols_freq_domain = self.raw_data_to_symbols(self.raw_samples, self.start, ffo=self.detected_ffo, sampling_offset=self.sampling_offset, angle=angle)

        # 调试用：把均衡后的频域符号 IFFT 回时域，便于观察校正后的包频谱。
        yfake = np.zeros(len(self.CP_LENGTHS)*NFFT, dtype=np.complex64)
        for i, symbol_f in enumerate(self.symbols_freq_domain):
            yfake[i*NFFT:(i+1)*NFFT] = itfft(self.symbol_equalized(symbol_f, self.channel))

        if self.debug:
            plt.title("Channel-Equalized Packet")
            plt.specgram(yfake, Fs=Fs)
            plt.show()

    def raw_data_to_symbols(self, samples, first_symbol_offset, ffo = None, sampling_offset = None, angle = None, linear_rotation=None):
        """把连续时域采样切成 OFDM 符号，并转换到频域子载波。"""

        samples = samples[first_symbol_offset:]

        if ffo != None:
            # 细频偏校正：对时域采样乘以复指数，抵消残余载波偏移。
            samples = fshift(samples, -ffo, self.Fs)

        symbols_time_domain = []
        symbols_freq_domain = []

        if sampling_offset != None:
            # 亚采样定时校正：用插值实现非整数采样点平移。
            samples = with_sample_offset(samples, sampling_offset)

        if angle != None:
            # 整体相位校正：把星座整体旋回参考方向。
            samples *= np.exp(-1j * angle)
        
        sample_offset = 0
        for i, cp_len in enumerate(CP_LENGTHS):
            # 每个 OFDM 符号由循环前缀 CP 和 NFFT 个有效采样组成。
            symbols_time_domain.append(samples[sample_offset:sample_offset+NFFT+cp_len])
            sample_offset = sample_offset + NFFT + cp_len

        for i, _ in enumerate(symbols_time_domain):
            # 去掉循环前缀后做 FFT，只保留中心的有效子载波。
            sym = symbols_time_domain[i][CP_LENGTHS[i]:]
            symbols_freq_domain.append(tfft(sym))

        if linear_rotation != None:
            # 供手工调试使用：给频域子载波施加线性相位旋转。
            for i, symbol in enumerate(symbols_freq_domain):
                x = np.linspace(-.5 * linear_rotation * len(symbol), .5 *
                            linear_rotation * len(symbol), len(symbol))
                symbols_freq_domain[i] *= np.exp(x * 2j * np.pi)

        return symbols_time_domain, symbols_freq_domain

    def estimate_channel(self, sym_index, zc_seq):
        """用已知 ZC 参考序列估计每个有效子载波上的复信道响应。"""
        if sym_index not in self.ZC_SYMBOL_IDX:
            raise ValueError("Bad ZC Symbol Index")

        # 如果 sym_index == ZC_SYMBOL_IDX[0]:
        #     zc_seq = 600
        # 如果 sym_index == ZC_SYMBOL_IDX[1]:
        #     zc_seq = 147

        expected_signal = zcsequence_f(zc_seq, NCARRIERS)
        received_signal = self.symbols_freq_domain[sym_index]

        expected_signal[NCARRIERS//2] = 1
        channel = np.divide(received_signal, expected_signal)

        if self.debug:
            plt.title("Channel Estimation")
            plt.plot(np.abs(channel))
            plt.show()

        return channel

    def symbol_equalized(self, symbol_f, channel):
        """用频域信道响应对一个 OFDM 符号做一拍均衡。"""
        return np.divide(symbol_f, channel)

    def find_zc_angle(self, symbol_f, zc_seq):
        """根据 ZC 符号中心载波估计整包的公共相位旋转。"""
        a = zcsequence_t(zc_seq, NCARRIERS)

        if (symbol_f == 0).any():
            symbol_f += 1

        adiff = np.angle(a / symbol_f)
        adiff[NCARRIERS//2] = adiff[NCARRIERS//2+1]
        adiff = np.unwrap(adiff)

        slope = np.max(adiff) - np.min(adiff)
        slope = (adiff - np.mean(adiff))
        slope = np.sqrt(np.mean(slope**2))

        if self.debug:
            print("slope", slope)
            print("phase 0", np.angle(symbol_f[NCARRIERS//2]))
            plt.plot(adiff)
            plt.title("Phase diff of ZC Seq")
            plt.show()

        return np.angle(symbol_f[NCARRIERS//2])

    def find_fine_start(self, samples):
        """利用循环前缀相关细调符号起点，并估计残余频偏。"""
        res = []
        cpl = self.CP_LENGTHS[0]

        for n in range(NFFT, len(samples) - cpl):
            # CP 是当前符号尾部的拷贝；当 n 对齐到符号尾部时，两段相关峰值最大。
            ac = np.sum(samples[n:n+cpl] * np.conj(samples[n-NFFT:n-NFFT+cpl]))
            #ac = np.max(corr(samples[n:n+cpl], samples[n-NFFT:n-NFFT+cpl]))
            res.append(ac)

        res_abs = np.abs(res)
        # 峰之间的距离大约是一个 OFDM 符号长度，避免把同一个峰重复检测。
        peaks, _ = signal.find_peaks(res_abs, distance = 1000)
        peak_prominences, _, _ = signal.peak_prominences(res_abs, peaks)
        # 丢掉不够明显的小峰，保留第一个可靠的符号边界。
        peak_index = np.where(peak_prominences > 1.0)[0][0]
        
        if self.debug:
            x = np.linspace(0, len(samples) / (NFFT + cpl), len(res_abs))
            plt.plot(x, np.array(res_abs)*300)
            # 调试图：把 CP 相关峰叠加在原始频谱上，观察粗定时是否合理。
            plt.scatter(x[peaks], abs(res_abs[peaks])*300, marker='x')
            plt.scatter(x[peaks-cpl//2], abs(res_abs[peaks])*300, marker='x')
            plt.specgram(samples, Fs=NFFT + cpl, NFFT=NFFT//64,
                         window=matplotlib.mlab.window_none, noverlap=0)
            plt.title("Raw Spectrum + Rough Packet Peak Estimation")
            plt.show()

        start = peaks[peak_index]

        # 相关值的相位来自符号前后重复片段之间的相位差，可换算为细频偏。
        ffo = self.Fs / (2 * np.pi * NFFT) * np.angle(res[start])
        print("FFO: %f" % ffo)
        return start, ffo

    def find_zc_seq(self, symbol_f):
        """遍历可能的 ZC 根序列，找出与接收符号相关性最高的序列号。"""
        res = []
        for r in range(1, NCARRIERS):
            a = zcsequence_t(r, NCARRIERS)
            res.append(np.max(np.abs(corr(symbol_f, a))))

        best = np.argmax(res) + 1
        if self.debug:
            print("best zc seq", best)

            # 调试时画出所有根序列的相关峰；理想情况下应只有一个明显峰值。
            plt.plot(res)
            plt.title("Correlation for best ZC Sequence (seq=" +str(best) +")")
            plt.show()

        return best

    def find_zc_offset(self, symbol_idx, seq, cyc):
        """通过最小化 ZC 相位斜率，搜索最佳亚采样定时偏移。"""
        a = zcsequence_t(seq, NCARRIERS)

        resx = []
        resy = []

        # 固定粗起点和细频偏，在小范围内尝试不同的非整数采样偏移。
        samples = self.raw_samples_orig[self.start:]
        samples = fshift(samples, -self.detected_ffo, self.Fs)

        for i in np.linspace(-15, 15, 1000):
            _, symbols_f = self.raw_data_to_symbols(samples, 0, ffo=None, sampling_offset=i)

            zc_sym_f = symbols_f[symbol_idx]

            # 防止除零导致相位差计算出现异常值。
            if (zc_sym_f == 0).any():
                zc_sym_f += 1

            adiff = np.angle(a / zc_sym_f)
            # DC 子载波不承载有效参考信息，用相邻值替代后再展开相位。
            adiff[NCARRIERS//2] = adiff[NCARRIERS//2+1]
            adiff = np.unwrap(adiff)

            slope = np.max(adiff) - np.min(adiff)
            slope = (adiff - np.mean(adiff))
            slope = np.sqrt(np.mean(slope**2))

            # x = np.linspace(0, len(adiff)-1, len(adiff))
            # npslope, npoffset = np.polyfit(x, adiff, 1)

            resx.append(i)
            resy.append(slope)

        if self.debug:
            plt.title("RMS for ZC sequence")
            plt.xlabel("Sample Offset Correction")
            plt.ylabel("")
            plt.plot(resx, resy)
            plt.plot(resx[np.argmin(resy)], np.min(resy), marker='X')
            plt.show()
    
        return resx[np.argmin(resy)]

    def find_zc_shift(self, symbol_f, seq: int, cyc=0):
        """寻找 ZC 循环移位。"""
        a = np.zeros(NFFT, dtype=np.complex64)

        a = zcsequence_f(seq, MAXNCARRIERS)
        rx_symbol_f = self.symbol_equalized(symbol_f, self.channel)
    
        am = np.argmax(np.abs(corr(rx_symbol_f, a)))
        return (cyc - am) % (NCARRIERS)
    
    def get_symbol_data(self, linear_rotation=0, _sampling_offset=0, tune=0, skip_zc=False):
        """返回最终用于 QPSK 解码的频域 OFDM 符号。"""
        sampling_offset = self.sampling_offset+_sampling_offset
        ffo = self.detected_ffo+tune

        # 按最终校正参数重新生成所有频域符号。
        _, all_symbols_f = self.raw_data_to_symbols(self.raw_samples_orig, self.start, ffo = ffo, sampling_offset=sampling_offset, linear_rotation=linear_rotation)
        
        symbols_f = []
        for i, symbol in enumerate(all_symbols_f):
            if skip_zc and i in self.ZC_SYMBOL_IDX:
                continue
            symbols_f.append(symbol)
        return symbols_f

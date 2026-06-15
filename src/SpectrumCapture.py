import numpy as np
import matplotlib.pyplot as plt
from packetizer import find_packet_candidate_time
from helpers import estimate_offset, fshift, resample

class SpectrumCapture:
    """保存原始采样，并提供粗切分后的 Drone-ID 候选帧。

    这个类不做完整解调，只负责从连续采样中分离疑似 RF 突发帧，
    并在进入 Packet 的 OFDM 符号同步前完成必要的粗射频校正。
    """
    raw_data: np.array
    sampling_rate: float
    packets: list
    debug: bool

    def __init__(self, raw_data=None, skip_detection=False, Fs=50e6, debug=False, p_type = "droneid", legacy = False):
        """保存输入采样，并按需执行粗帧检测。"""
        self.legacy = legacy
        self.raw_data = raw_data
        self.debug = debug
        self.sampling_rate = Fs
        self.packet_type = p_type
        if skip_detection:
            self.packets = [self.raw_data, ]
        else:
            self._packetize_coarse()

        if debug:
            print(f"SpectrumCapture: found {len(self.packets)} packets")

    def _packetize_coarse(self):
        """在输入采样窗口中寻找亚毫秒级 Drone-ID 突发帧。"""
        droneid_found = False

        # packetizer 会寻找能量持续时间和占用带宽都符合 Drone-ID 特征的时间片段。
        # 它还会返回最后一个有效候选帧的中心频偏估计值。
        self.packets, cfo = find_packet_candidate_time(self.raw_data, self.sampling_rate, debug = self.debug, packet_type=self.packet_type, legacy = self.legacy)

        if self.debug:
            # 调试时画出所有检测到的候选帧频谱。
            for p in self.packets:
                plt.specgram(p,Fs=self.sampling_rate)
                plt.show()

        if len(self.packets) > 0:
            droneid_found = True

        if not droneid_found:
            if self.debug:
                print("Could not verify DroneID packet!")

        #self.packets = droneid_pkt

    def get_packet_samples(self, pktnum=0, debug=False):
        """返回完成中心频偏粗校正、并重采样到 15.36 MHz 的 Drone-ID 帧。"""
        if pktnum >= len(self.packets):
            raise ValueError("Only %i packets available but you requested packet %i" % (len(self.packets), pktnum))

        packet_data = self.packets[pktnum].copy()

        # 根据功率谱密度估计突发帧中心频率，再乘以反向频移因子，
        # 把有效子载波搬回基带中心，方便后续 FFT 解调。
        print(f"get_packet_samples pkt={pktnum}")
        offset, success = estimate_offset(packet_data, self.sampling_rate)
        if success:
            packet_data = fshift(packet_data, -1.0*offset, self.sampling_rate)
        else:
            return ValueError("Cannot estimate carrier offset for packet %i" % (pktnum))

        if self.packet_type == "droneid" or self.packet_type == "beacon":
            resample_rate = 15.36e6
        elif self.packet_type == "c2":
            resample_rate = 1.92e6

        # Drone-ID 后续解调使用类似 LTE 的 OFDM 参数：
        # 1024 点 FFT、601 个有效子载波、15.36 MHz 采样时钟。
        # 这里先重采样，后面的循环前缀长度和 FFT 子载波间隔才对应得上。
        if self.sampling_rate > resample_rate + .1e6:
            if debug:
                print("Resampling from %i MHz to %f MHz" % ((self.sampling_rate / 1e6),resample_rate))
            packet_data = resample(packet_data, self.sampling_rate, resample_rate)
        elif self.sampling_rate < resample_rate - .1e6:
            raise ValueError("Your sampling rate is too low")
        else:
            if debug:
                print("Sampling rate matches, not resampling.")
        
        if self.debug:
            plt.specgram(packet_data, Fs=self.sampling_rate)
            plt.show()
        
        return packet_data

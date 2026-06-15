#!/usr/bin/env python3

import argparse
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from helpers import estimate_offset

def find_packet_candidate_time(raw_data, Fs, debug=False, packet_type = "droneid", legacy = False):
    """根据突发能量持续时间寻找长度符合要求的候选帧。"""
    # Mavic 2 等旧机型的帧约为 576 us，缺少符号 0。
    # 新机型 Drone-ID 帧更长，约为 630 到 665 us。

    if packet_type == "droneid": 
        if legacy:
            min_packet_len_t = 565e-6
            max_packet_len_t = 600e-6
        else:
            min_packet_len_t = 630e-6
            max_packet_len_t = 665e-6
    elif packet_type == "c2":
        min_packet_len_t = 500e-6
        max_packet_len_t = 520e-6
    elif packet_type == "beacon":
        min_packet_len_t = 490e-6
        max_packet_len_t = 540e-6
    elif packet_type == "pairing":
        min_packet_len_t = 490e-6
        max_packet_len_t = 540e-6
    elif packet_type == "video":
        min_packet_len_t = 630e-6
        max_packet_len_t = 665e-6


    print("Packet Type:",packet_type)

    start_offset = 3*15e-6
    end_offset = 3*15e-6

    f, t, Zxx = signal.stft(raw_data, Fs, nfft=64, nperseg=64)
    res_abs = np.max(np.abs(Zxx), axis=0)
    noise_floor = np.mean(np.abs(Zxx))


    # 对短时频谱每个时间片取最大幅度，超过平均噪声底一定比例就认为有信号。
    above_level = res_abs > 1.15*noise_floor

    # 在“高于噪声底”的布尔序列中寻找宽度符合帧时长的连续区域。
    signal_length_min_samples = int(min_packet_len_t/(t[1]-t[0])) # 帧时长换算成 STFT 时间片数量
    signal_length_max_samples = int(max_packet_len_t/(t[1]-t[0])) # 帧时长换算成 STFT 时间片数量
    peaks, properties = signal.find_peaks(above_level, width=[signal_length_min_samples, signal_length_max_samples],wlen=100*signal_length_max_samples)

        
    packets = []
    center_freq_offset = 0
    start = 0
    end = 0
    length = 0

    for i, _ in enumerate(peaks):
        start = properties["left_bases"][i] * (t[1]-t[0]) # STFT 时间片换算成秒
        end = properties["right_bases"][i] * (t[1]-t[0])
        length =  properties["widths"][i] * (t[1]-t[0])

        packet_data = raw_data[int((start-start_offset)*Fs):int((end+end_offset)*Fs)]

        # 估计中心频偏；只有候选帧带宽符合预期时才会认为估计成功。
        center_freq_offset, found = estimate_offset(packet_data, Fs)

        if not found:
            if debug:
                print("Packet #%i, start %f, end %f, length %f, cfo MISMATCH" % (i, start, end, length))
            continue

        print(center_freq_offset)
        print("Packet #%i, start %f, end %f, length %f, cfo %f" % (i, start, end, length, center_freq_offset))
        packets.append(packet_data)

    if debug:
        print("legacy")

        plt.plot(t, above_level)
        plt.scatter(t[peaks], abs(above_level[peaks]), marker="x", color="C5")
        plt.show()

    return packets, center_freq_offset

def main(args):
    data = np.fromfile(args.input_file, dtype="<f").astype(np.float32).view(np.complex64)
    find_packet_candidate_time(data, args.sample_rate, args.debug)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', help="Binary Sample Input")
    parser.add_argument('-s', '--sample-rate', default="50e6", type=float, help="Sample Rate")
    parser.add_argument('-d', '--debug', default=False, action="store_true", help="Enable debug output")
    args = parser.parse_args()

    main(args)

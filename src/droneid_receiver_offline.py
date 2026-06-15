#!/usr/bin/env python3

import argparse
import numpy as np

from SpectrumCapture import SpectrumCapture
from Packet import Packet
from qpsk import Decoder
from droneid_packet import DroneIDPacket
from gui import interactive

def main(_args):
    """把离线 IQ 采样文件解码成 Drone-ID 载荷。

    输入文件按 float32 交错保存 I/Q 数据。这里先按 float32 读取，再把
    相邻两个浮点数组成一个 complex64，从而恢复后续检测和解调使用的
    复数基带采样流。
    """
    raw = np.memmap(_args.input_file, mode='r', dtype="<f").astype(np.float32).view(np.complex64)

    packets_decoded = 0
    crc_error = 0

    drone_coords = []
    app_coords = []

    # 按 500 ms 分块处理，避免对长采样文件一次性做很大的 STFT。
    # 每个分块内部仍会继续检测单个亚毫秒级 Drone-ID 突发帧。
    chunk_samples = int(500e-3 * _args.sample_rate) # 单位：秒
    chunks = len(raw) // chunk_samples +1

    for i in range(chunks):
        print("Drone-ID Frame Detection")

        # SpectrumCapture 负责类似射频前端的粗处理：
        # 粗帧检测、粗中心频偏估计、搬频回基带，以及重采样到解调器使用的采样率。
        capture = SpectrumCapture(raw[i*chunk_samples:(i+1)*chunk_samples], skip_detection = args.skip_detection, Fs=_args.sample_rate, debug=args.debug, legacy=args.legacy)
        print(f"Found {len(capture.packets)} Drone-ID RF frames in spectrum capture.")

        for packet_num, _ in enumerate(capture.packets):
            payload = None

            print(f"################## Decoding Frame {packet_num+1}/{len(capture.packets)} ##################")

            # 获取一个已经完成重采样和粗中心频偏校正的 Drone-ID 帧。
            # 更细的定时、频偏和相位校正会在下面的 Packet 类中完成。
            packet_data = capture.get_packet_samples(pktnum=packet_num)

            try:
                # Packet 把时域帧转换成校正后的 OFDM 符号：
                # 先用循环前缀相关找第一个符号和细频偏，再用 Zadoff-Chu 参考符号
                # 做同步确认、信道估计和相位校正，最后通过 FFT 恢复有效子载波。
                packet = Packet(packet_data, debug=args.debug, enable_zc_detection=not args.disable_zc_detection, legacy=args.legacy)
            except Exception as error:
                print(f"Demodulation FAILED (Frame {packet_num+1}): {error}")
                continue
            
            # 打开 GUI, 便于手工检查 RF 帧和中间解调结果。
            if _args.gui:
                interactive(packet)

            # 取出已经完成定时、频偏和信道校正的频域 OFDM 符号。
            # skip_zc=True 会去掉两个同步用的 ZC 符号，只保留承载 QPSK 数据的符号。
            symbols = packet.get_symbol_data(skip_zc=True)
            decoder = Decoder(symbols)
    
            # RF 同步后，QPSK 星座的绝对旋转仍可能差 0/90/180/270 度。
            # 因此暴力尝试 4 种相位映射，能解析出合法 Drone-ID 包的就是正确方向。
            for phase_corr in range(4):
                decoder.raw_data_to_symbol_bits(phase_corr)
                droneid_duml = decoder.magic()

                try:
                    payload = DroneIDPacket(droneid_duml)
                except:
                    continue

                print(f"## Drone-ID Payload ##")
                print(payload)

                if not payload.check_crc():
                    print("CRC error!")

                    # CRC 校验失败, 说明当前相位方向可解析但载荷有误码。
                    crc_error += 1
                    break

                drone_lat, drone_lon, app_lat, app_lon, height = DroneIDPacket(droneid_duml).get_coords()

                # CRC 正确, 说明收到了一个有效 Drone-ID 包。
                packets_decoded += 1

                if drone_lat != 0.0 and drone_lon != 0.0:
                    drone_coords.append((drone_lat, drone_lon, height))

                if app_lat != 0.0 and app_lon != 0.0:
                    app_coords.append((app_lat,app_lon))
    
                # 当前帧已经解码成功, 不再尝试其他 QPSK 相位方向。
                break

            if not payload:
                print(f"Frame {packet_num}/{len(capture.packets)}: Decoding failed.")

    print("\n\n")
    print(f"Frame detection: {len(capture.packets)} candidates")
    print(f"Decoder: {packets_decoded+crc_error} total, CRC OK: {packets_decoded} ({crc_error} CRC errors)")
    
    print("Drone Coordinates:")
    for coords in drone_coords:
        print(coords)

    print("App Coordinates:")
    for coords in app_coords:
        print(coords)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gui', default=False, action="store_true", help="Show interactive")
    parser.add_argument('-i', '--input-file', default="../samples/mini2_sm", help="Binary Sample Input")
    parser.add_argument('-s', '--sample-rate', default="50e6", type=float, help="Sample Rate")
    parser.add_argument('-l', '--legacy', default=False, action="store_true", help="Support of legacy drones (Mavic Pro, Mavic 2)")
    parser.add_argument('-d', '--debug', default=False, action="store_true", help="Enable debug output")
    parser.add_argument('-z', '--disable-zc-detection', default=True, action="store_false", help="Disable per-symbol ZC sequence detection (faster)")
    parser.add_argument('-f', '--skip-detection', default=False, action="store_true", help="Skip packet detection and enforce decoding of input file")
    args = parser.parse_args()

    main(args)

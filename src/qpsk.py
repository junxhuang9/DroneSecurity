#!/usr/bin/env python3

import argparse
import bitarray

import numpy as np
from goldgen import gold
from droneid_packet import DroneIDPacket

# QPSK 象限到 2 bit 符号的映射表。
# 由于接收端无法直接知道绝对星座方向，这里准备了 0/90/180/270 度四种旋转。
qpsk_to_bits = [[2, 3, 1, 0],
                [0, 2, 3, 1], # +90 度
                [1, 0, 2, 3], # +180 度
                [3, 1, 0, 2]]  # +270 度

# Drone-ID 帧中承载数据的 OFDM 符号编号。
sym = [0, 1, 2, 4, 6, 7, 8] # 符号 3 和 5 故意跳过，
                            # 因为它们承载 ZC 同步序列，不承载载荷信息。

# 3GPP turbo rate matching 使用的列交织排列。
RM_PERM_TURBO = [0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30, 1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31]

def rm_turbo_rx(bits_in):
    """撤销发送端的 turbo rate matching 交织，恢复系统比特流。"""
    ncols = 32
    nrows = (len(bits_in) + 31) // ncols
    n_dummy = (ncols * nrows) - len(bits_in)

    bits = np.zeros((nrows, ncols), dtype=int)

    p = 0
    for col in range(ncols):
        if RM_PERM_TURBO[col] < n_dummy:
            bits[1:,RM_PERM_TURBO[col]] = bits_in[p:p + nrows - 1]
            bits[0,RM_PERM_TURBO[col]] = -1
            p += nrows - 1
        else:
            bits[:,RM_PERM_TURBO[col]] = bits_in[p:p + nrows]
            p += nrows
    assert p == len(bits_in)

    bits_out = bits.flatten()
    assert (bits_out[:n_dummy] == -1).all()
    return bits_out[n_dummy:]

def get_symbol_bits(symbol: np.complex, phase_correction: int=0) -> int:
    """按象限对单个 QPSK 星座点做硬判决，输出 0..3 的 2 bit 符号值。"""
    if phase_correction < 0 or phase_correction >= len(qpsk_to_bits):
        raise ValueError("Invalid phase correction")

    if symbol.real >= 0 and symbol.imag >= 0:
        return qpsk_to_bits[phase_correction][0]
    elif symbol.real >= 0 and symbol.imag < 0:
        return qpsk_to_bits[phase_correction][1]
    elif symbol.real < 0 and  symbol.imag< 0:
        return qpsk_to_bits[phase_correction][2]
    elif symbol.real < 0 and  symbol.imag> 0:
        return qpsk_to_bits[phase_correction][3]

class Decoder:
    def __init__(self, raw_data=None):
        # raw_data 是二维列表：7 个数据 OFDM 符号，每个符号包含 601 个 QPSK 子载波。

        self.raw_data = []
        self.sym_bits = []

        if raw_data != None:
            self.raw_data = raw_data

    def raw_data_to_symbol_bits(self, phase_correction):
        """把频域 QPSK 子载波硬判决为每个子载波的 2 bit 符号值。"""
        demod = []

        for frame_symbol in self.raw_data:
            frame_symbol_demod = []
            for qpsk_symbol in frame_symbol:
                frame_symbol_demod.append(get_symbol_bits(qpsk_symbol, phase_correction))
            demod.append(frame_symbol_demod)

        self.sym_bits = demod

    def read_file(self, path=None):
        """从 pkt_sym_*.txt 调试文件读取已经导出的 QPSK 星座点。"""
        raw_data = []

        for i, s in enumerate(sym):
            f = "pkt_sym_" + str(s) + ".txt"
            qbits = open(f).readlines()

            raw_data.append([])
            for qval_ in qbits:
                qval_ = qval_.split(" ")
                qval = np.complex(float(qval_[0]), float(qval_[1]))
                raw_data[i].append(qval)

        self.raw_data = raw_data

    def magic(self):
        """完成 Drone-ID 载荷恢复：去 DC、解扰、抽取系统位并打包成字节。"""
        sym_bits = self.sym_bits
        bits = np.array(sym_bits)
        # 删除中心 DC 子载波，只保留 600 个真正承载 QPSK 数据的子载波。
        bits = np.delete(bits, 300,1)

        # 每个 0..3 符号拆成两个布尔 bit。
        bits = np.repeat(bits, 2, axis=1)
        bits &= np.tile([1,2], 600)
        bits = bits > 0
        bits = bits.astype(bool)

        if len(np.concatenate((bits[0:]))) > 7200:

            goldseq = gold(1600, 1200, 0x12345678)
            # print("Gold 序列长度: ", len(goldseq))
            # print("符号 0 的 Gold 序列")
            gold_err = 0
            for i,j in enumerate(bits[0]):
                if j != goldseq[i]:
                    gold_err += 1

            if not np.all(goldseq == bits[0]):
                # print("符号 0 与 Gold 序列不匹配")
                # print("Gold 错误数:", gold_err, "\n")
                pass
                # print("Gold 序列不匹配")
                #return False

            all_bits = np.concatenate((bits[1:]))
        else:
            # 兼容旧机型：旧帧缺少符号 0，因此不跳过第一组符号。
            all_bits = np.concatenate((bits[0:]))

        # 用 Gold 序列解扰。
        plo = gold(1600, len(all_bits), 0x12345678) ^ all_bits

        # 从循环缓冲中抽取 turbo 编码后的系统位；此处忽略校验流。
        plo = np.concatenate((plo, plo))
        plo = plo.astype(int)
        offset = 4148

        systematic_stream = plo[offset:offset + 1412]

        # 撤销 rate matching，得到载荷比特。
        p_decoded = rm_turbo_rx(systematic_stream)

        # 按大端 bit 顺序打包成字节，交给 DroneIDPacket 解析结构字段。
        ba = bitarray.bitarray(list(p_decoded), endian='big')
        return ba.tobytes()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--phase-shift', type=int, default=0, help="Phase Shift (0..3)")
    args = parser.parse_args()
    d = Decoder()
    d.read_file()
    for phase_corr in range(4):
        d.raw_data_to_symbol_bits(phase_corr)
        droneid_pack = d.magic()
        if droneid_pack:
            try:
                payload = DroneIDPacket(droneid_pack)
                print(payload)
                break
            except:
                continue

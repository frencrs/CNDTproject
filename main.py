import tensorflow as tf
import numpy as np
import math as m
import matplotlib.pyplot as plt
from collections import Counter
from scapy.all import *

count = 0
reader1 = PcapReader("TestCap1.pcapng")

stats_of_pcap = {
    'packet_count': 0,
    'avg_len': 0,
    'len_sum': 0,
    'port_list': [], 'unique_port_count': 0, 'num_unique_ports': 0,
    'dst_list': [], 'unique_dst_count': 0, 'num_unique_dst': 0,
    'src_list': [], 'unique_src_count': 0, 'num_unique_src': 0,
    'ttl_list': [], 'unique_ttl_count': 0, 'num_unique_ttl': 0,
    'chksum_list': [], 'unique_chksum_count': 0, 'num_unique_chksum': 0,
}


for (pkt_data) in reader1.read_all():
    try:
        print(collections.ChainMap(pkt_data))

        stats_of_pcap['len_sum'] += pkt_data.len
        stats_of_pcap['port_list'].append(pkt_data.dport)
        stats_of_pcap['dst_list'].append(pkt_data['IP'].dst)
        stats_of_pcap['src_list'].append(pkt_data['IP'].src)
        stats_of_pcap['ttl_list'].append(pkt_data['IP'].ttl)
        stats_of_pcap['chksum_list'].append(pkt_data['IP'].chksum)

        count += 1

    except(ValueError, AttributeError, IndexError):
        print("Packet Error")
        pass


stats_of_pcap['unique_port_count'] = Counter(stats_of_pcap['port_list'])
stats_of_pcap['num_unique_ports'] = len(stats_of_pcap['unique_port_count'])
stats_of_pcap['unique_dst_count'] = Counter(stats_of_pcap['dst_list'])
stats_of_pcap['num_unique_dst'] = len(stats_of_pcap['unique_dst_count'])
stats_of_pcap['unique_src_count'] = Counter(stats_of_pcap['src_list'])
stats_of_pcap['num_unique_src'] = len(stats_of_pcap['unique_src_count'])
stats_of_pcap['unique_ttl_count'] = Counter(stats_of_pcap['ttl_list'])
stats_of_pcap['num_unique_ttl'] = len(stats_of_pcap['unique_ttl_count'])
stats_of_pcap['unique_chksum_count'] = Counter(stats_of_pcap['chksum_list'])
stats_of_pcap['num_unique_chksum'] = len(stats_of_pcap['unique_chksum_count'])
stats_of_pcap['packet_count'] = count
stats_of_pcap['avg_len'] = stats_of_pcap['len_sum']/count

print(stats_of_pcap['avg_len'],
      stats_of_pcap['num_unique_ports'],
      stats_of_pcap['num_unique_dst'],
      stats_of_pcap['num_unique_src'],
      stats_of_pcap['num_unique_ttl'],
      stats_of_pcap['num_unique_chksum'],
      stats_of_pcap['packet_count']
      )

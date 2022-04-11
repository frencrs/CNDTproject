import tensorflow as tf
import numpy as np
import math as m
import matplotlib.pyplot as plt
from scapy.all import *

count = 0
reader1 = PcapReader("TestCap1.pcapng")




for (pkt_data) in reader1.read_all():
    print(pkt_data.dst)
    print(pkt_data.time)
    print(pkt_data.len)
    print(pkt_data.dport)
    print(pkt_data.sent_time)
    #print(pkt_data)
    #print(pkt_data.summary())
    count += 1

print(count)





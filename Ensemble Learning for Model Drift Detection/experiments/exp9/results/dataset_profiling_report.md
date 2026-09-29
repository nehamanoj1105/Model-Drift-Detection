# Dataset Profiling Report: 5G Campus Network QoS Dataset (Zenodo 13754300)

## 1. Overview & File Inventory

The dataset contains 6 CSV files capturing packet-level and throughput-level telemetry across NTNU and WUE testbeds.

### Throughput Summaries

- **ntnu_tput_all_Throughput.csv**: 26,971 bytes, 142 rows, 142 scenarios
  - gNBs: ['oai', 'srs'], SDRs: ['b200'], Bandwidths: [20, 40]
- **wue_tput_all_Throughput.csv**: 29,260 bytes, 160 rows, 160 scenarios
  - gNBs: ['oai', 'srs'], SDRs: ['b200'], Bandwidths: [20, 40]

### Packet Telemetry Profiling (Sampled / Streamed)

#### File: `ntnu_owd_Packets_with_IATs.csv`
- **File Size**: 385,728,512 bytes
- **Inspected Rows**: 1,700,000
- **Timestamp Range**: 1720178589.58873 to 1720212213.27376 (Duration: 33623.69 s)
- **Unique Scenarios**: 43
- **Traffic Directions**: {'uplink': 850000, 'downlink': 850000}
- **gNB Implementations**: {'oai': 1700000}
- **SDR Hardware**: {'b200': 1700000}
- **Bandwidths (MHz)**: {20: 1700000}
- **Duplicate Packets (in sample)**: 0
- **Missing Values**: {'Unnamed: 0': 0, 'src': 0, 'Timestamp': 0, 'SourceIPOuter': 859914, 'DestinationIPOuter': 859914, 'SourceIPInner': 0, 'DestinationIPInner': 0, 'PacketSize': 0, 'SeqNum': 0, 'iat': 0, 'trel': 0, 'gnb': 0, 'sdr': 0, 'bw': 0, 'slots': 0, 'ratio': 0, 'pdist': 0, 'piat': 0, 'psize': 0, 'pnpak': 0, 'rep': 0, 'scenario': 0, 'direction': 0}

**Top Scenarios by Packet Count:**
  - `oai_b200_20mhz_10slots_ratio2_det_01700_small_10000_rep1`: 39,996 packets
  - `oai_b200_20mhz_10slots_ratio2_det_01700_small_10000_rep2`: 39,996 packets
  - `oai_b200_20mhz_10slots_ratio2_det_01700_small_10000_rep3`: 39,996 packets
  - `oai_b200_20mhz_10slots_ratio2_det_01700_small_10000_rep4`: 39,996 packets
  - `oai_b200_20mhz_10slots_ratio2_det_01700_small_10000_rep5`: 39,996 packets
  - `oai_b200_20mhz_10slots_ratio2_exp_01700_small_10000_rep1`: 39,996 packets
  - `oai_b200_20mhz_10slots_ratio2_exp_01700_small_10000_rep2`: 39,996 packets
  - `oai_b200_20mhz_10slots_ratio2_exp_01700_small_10000_rep3`: 39,996 packets
  - `oai_b200_20mhz_10slots_ratio2_exp_01700_small_10000_rep4`: 39,996 packets
  - `oai_b200_20mhz_10slots_ratio2_exp_01700_small_10000_rep5`: 39,996 packets


#### File: `wue_owd_Packets_with_IATs.csv`
- **File Size**: 178,315,264 bytes
- **Inspected Rows**: 700,000
- **Timestamp Range**: 1720829348.63332 to 1721053297.33205 (Duration: 223948.70 s)
- **Unique Scenarios**: 18
- **Traffic Directions**: {'uplink': 350002, 'downlink': 349998}
- **gNB Implementations**: {'oai': 700000}
- **SDR Hardware**: {'b200': 700000}
- **Bandwidths (MHz)**: {20: 700000}
- **Duplicate Packets (in sample)**: 0
- **Missing Values**: {'Unnamed: 0': 0, 'src': 0, 'Timestamp': 0, 'SourceIPOuter': 359964, 'DestinationIPOuter': 359964, 'SourceIPInner': 0, 'DestinationIPInner': 0, 'PacketSize': 0, 'SeqNum': 0, 'iat': 0, 'trel': 0, 'gnb': 0, 'sdr': 0, 'bw': 0, 'slots': 0, 'ratio': 0, 'pdist': 0, 'piat': 0, 'psize': 0, 'pnpak': 0, 'rep': 0, 'scenario': 0, 'direction': 0}

**Top Scenarios by Packet Count:**
  - `oai_b200_20mhz_10slots_ratio2_det_01700_small_10000_rep13`: 39,996 packets
  - `oai_b200_20mhz_10slots_ratio2_det_01700_small_10000_rep22`: 39,996 packets
  - `oai_b200_20mhz_10slots_ratio2_det_01700_small_10000_rep23`: 39,996 packets
  - `oai_b200_20mhz_10slots_ratio2_det_01700_small_10000_rep24`: 39,996 packets
  - `oai_b200_20mhz_10slots_ratio2_det_01700_small_10000_rep31`: 39,996 packets
  - `oai_b200_20mhz_10slots_ratio2_det_01700_small_10000_rep32`: 39,996 packets
  - `oai_b200_20mhz_10slots_ratio2_det_01700_small_10000_rep33`: 39,996 packets
  - `oai_b200_20mhz_10slots_ratio2_exp_01700_small_10000_rep13`: 39,996 packets
  - `oai_b200_20mhz_10slots_ratio2_exp_01700_small_10000_rep15`: 39,996 packets
  - `oai_b200_20mhz_10slots_ratio2_exp_01700_small_10000_rep4`: 39,996 packets


## 2. Testbed & File Combination Rationale

- **NTNU vs WUE Testbeds**: The two testbeds differ in radio environment, hardware (SDR type), and gNB software configurations.
- **Independent vs Combined Stream**: For controlled streaming drift experiments, NTNU and WUE represent distinct operational environments. Combining them directly without domain alignment would inject static hardware differences. We construct recurring regime streams within each testbed domain to strictly evaluate policy transfer under recurring network operating regimes.
- **Regime Isolation**: Each scenario string uniquely specifies gNB, SDR, bandwidth, slot allocation, traffic pattern, packet size mode, and offered load. Scenarios with identical gNB/SDR/BW configuration constitute recurring operating regimes.

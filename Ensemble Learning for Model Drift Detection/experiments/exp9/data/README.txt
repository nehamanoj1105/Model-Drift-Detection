This 5G campus network dataset contains 6 distinct CSV files.
Three for each of the testbeds located at either the Norwegian University of Science and Technology(NTNU) or the University of Wuerzburg(WUE).
all_Packets: contains all One-Way-Delay measurement packets. Columns:  src, Timestamp, SourceIPOuter, DestinationIPOuter, SourceIPInner, DestinationIPInner, PacketSize, SeqNum, iat, trel, gnb, sdr, bw, slots, ratio, pdist, piat, psize, pnpak, rep, scenario, direction
Packets_with_IATs: contains all One-Way-Delay measurement packets, with the Interarrival-times already calculated and added as column. Columns: src, Timestamp, SourceIPOuter, DestinationIPOuter, SourceIPInner, DestinationIPInner, PacketSize, SeqNum, iat, trel, gnb, sdr, bw, slots, ratio, pdist, piat, psize, pnpak, rep, scenario, direction
all_Throughput: contains all output information from the throughput measurements conducted with iperf3. Columns:  mbpsactual_uplink, mbpsactual_downlink, meanjitterms_uplink, meanjitterms_downlink, meanloss_uplink, meanloss_downlink, gnb, sdr, bw, slots, ratio, mbpsoffered_uplink, mbpsoffered_downlink, rep, scenario

src: Source Device
Timestamp: Timestamp from wireshark
SourceIPOuter: Source IP of the outer GTP packet
DestinationIPOuter: Destination IP of the outer GTP packet
SourceIPInner: Source IP of the inner GTP packet
DestinationIPInner: Destination IP of the inner GTP packet
PacketSize: Size of the packet
SeqNum: Sequence number injected during traffic generation
iat: Interarrival-time of the packet
trel: relative timestamp
gnb: gNB implementation used
sdr: SDR used
bw: configured bandwidth
slots: configured DL slots
ratio: configured DL:UL ratio
pdist: chosen distribution of the packet generation (deterministic/negative-exponential time between transmission of the generated packets)
piat: chosen interarrival -time during packet generation
psize: chosen packet size mode
pnpak: total amount of packets generated for this measurement run
rep: repetition of measurement
scenario: measurement scenario
direction: direction of traffic

mbpsactual_uplink: actual uplink throughput in MBps
mbpsactual_downlink: actual downlink throughput in MBps
meanjitterms_uplink: actual uplink jitter in ms
meanjitterms_downlink: actual downlink jitter in ms
meanloss_uplink: mean loss in uplink
meanloss_downlink: mean loss in downlink
mbpsoffered_uplink: configured uplink throughput in MBps
mbpsoffered_downlink: configured downlink throughput in MBps

This dataset stems from the Paper "Parameterizing 5G New Radio: A Comparative Measurement Study on Throughput and Delay" by Simon Raffeck, Sebastian G. Grøsvik, Stanislav Lange, Tobias Hossfeld, Thomas Zinner, Stefan Geissler of the CNSM conference 2024.
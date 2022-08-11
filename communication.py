import math
import numpy as np
# Configure the network
# Initialize the wireless environment and some other variables.
# Calculate the maximum data rate carried by a single Resource Block into one slot to estimate 5G Throughput .
Number_Sub_carriers = 12
Number_Symbols = 14
Symbols_Data = 11  # 14- (2 for DMRS: demodulation reference signal  and 1 for Physical Downlink Control Channel)
Data_Resource_Elements = Number_Sub_carriers * Symbols_Data
d_slot = 1600  # Number of slots per second for downlink if  DL/UL Ratio is 4:1
path_loss_exponent = 3
C = 3e8  # Speed of light
CSI_mu = 1
cell_radius = 3e3  # In terms of meters
freq = 900e6
channel_wavelength = C/freq
N_0 = math.pow(10, ((-169 / 3) / 10))  # Noise power spectrum density is -169dBm/Hz;
eta_los = 1  # Loss corresponding to the LoS connections
eta_nlos = 20  # Loss corresponding to the NLoS connections
path_loss_factor = eta_los - eta_nlos  #
C = 20 * np.log10(4 * np.pi * 9 / 3) + eta_nlos  #  \carrier frequncy is # 900Mhz=900*10^6,
# and light speed is c=3*10^8; then one has f/c=9/3;
transmission_power = 5 * math.pow(10, 5)  # maximum uplink transimission power of one GT is 5mW;
Number_MIMO_layers = 4
channel_bandwidth = 100  # Channel bandwidth: 100 MHZ


def wireless_network(number_vehicles, rb_vehicle, MCS_spectrum_efficiency, distance_vehicle_ORU):
    TB_size = Data_Resource_Elements * MCS_spectrum_efficiency  # transmission block size
    TB_size = math.floor(TB_size)
    throughput_service = TB_size * math.floor(rb_vehicle) * math.floor(d_slot) * Number_MIMO_layers
    # This is in bits per second. Now we need to convert it to mbps
    throughput_service = throughput_service / 1000 / 1000
    sigma = 7  # Standard  deviation[dB]
    mu = 0  # Zero mean
    tempx = np.random.normal(mu, sigma, number_vehicles)
    x_com = np.mean(tempx)  # In term of dBm
    PL_0_dBm = 34  # In terms of dBm;
    PL_dBm = PL_0_dBm + 10 * path_loss_factor * math.log10(1+distance_vehicle_ORU) + x_com
    path_loss = 10 ** (PL_dBm / 10)  # [milli - Watts]
    channel_gain = transmission_power - path_loss
    channel_gain = float(channel_gain)
    SNR = (transmission_power * channel_gain ** 2 * distance_vehicle_ORU)/N_0
    achievable_data_rate = channel_bandwidth * math.log(1 + SNR)
    return throughput_service,  achievable_data_rate


def fronthaul_network():
    fronthaul_capicity = int(np.random.uniform(low=2000, high=4000))  # In terms of megabytes
    return fronthaul_capicity

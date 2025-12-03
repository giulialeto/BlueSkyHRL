import pandas as pd
from bluesky.tools.aero import ft, kts
from data_analysis.opensky_tools.plot_trajectory import plot_trajectory, plot_trajectory_line, plot_trajectory_clickable2
import numpy as np

## HAVE TO FILTER FOR INVALID FLIGHTS

select_flights = False
gen_color_dict = False
slice_data = True
slice_min = 0
slice_max = 500
alpha = 0.9

data = pd.read_csv("output/july_2024_direct/flight_output.csv")


if select_flights:
    # List of flight codes you want to include
    # flights = ['KL8757', 'KL8832', 'KL8834','KL9963','KL9962','KL9961']

    # HOLDING MUCH
    # flights = ['KL2070','KL850','KL1547','KL2964','KL535',
    #            'KL2324','KL2076','KL171','KL2197','KL2405','KL758','KL840',
    #            'KL119','KL1584','KL2214','KL48','KL854',
    #            'KL962','KL926','KL844','KL647','KL920']
    
    # HOLDING LITTLE
    # flights = ['KL2070','KL535',
    #            'KL2324','KL840',
    #            'KL119','KL854',
    #            'KL962','KL920']

    # TROMBONING
    flights = ['KL361',
               'KL83',
               'KL50',
               'KL51'
               ]
    
    flights = ['KL220',
               'KL221',
               'KL222',
               'KL223',
               'KL224',
               'KL225',
               'KL226',
               'KL227',
               'KL228',
               'KL229',
               'KL230',
               'KL231',
               'KL361',
               'KL83',
               'KL50'
                ]
    
    # POINTMERGE 

    # flights = ['KL192','KL181']
    data = data[data['icao24'].isin(flights)]

if gen_color_dict:
    color_dict = {'KL220':'red',
               'KL221':'blue',
               'KL222':'black',
               'KL223':'green',
               'KL224':'green',
               'KL225':'red',
               'KL226':'blue',
               'KL227':'black',
               'KL228':'black',
               'KL229':'blue',
               'KL230':'red',
               'KL231':'red',
               'KL361':'blue',
               'KL83':'red',
               'KL50':'red'
    }
else:
    color_dict = None

if slice_data:
    id = data["icao24"].unique()
    flights = id[slice_min:slice_max]
    data = data[data['icao24'].isin(flights)]

data = data.rename(
        columns={
            "time": "timestamp",
            "lat": "latitude",
            "lon": "longitude",
            "heading": "track",
        }
    ).drop(columns=["geoaltitude"])

data = data.assign(
        altitude=data.baroaltitude, # / ft,
        vertical_rate=data.vertrate / ft * 60,
        groundspeed=data.velocity # / kts,
    )


# Constants (SI)
gamma = 1.4               # ratio of specific heats
R = 287.05287             # specific gas constant for air, J/(kg·K)
g = 9.80665               # gravity, m/s^2
p0 = 101325.0             # sea-level standard pressure, Pa
T0 = 288.15               # sea-level standard temp, K
L = 0.0065                # tropospheric lapse rate, K/m
KTS_TO_MPS = 0.514444444  # 1 kt = 0.514444... m/s
MPS_TO_KTS = 1.0 / KTS_TO_MPS

# helper: ISA pressure & temperature given geometric/baro altitude in meters
def isa_pressure_temperature(h):
    """
    Return (p, T) at altitude h (m) using a simple ISA model:
      - troposphere (h <= 11000 m): linear lapse
      - stratosphere (h > 11000 m): isothermal layer approx
    """
    h = np.asarray(h)
    # troposphere mask
    mask_trop = h <= 11000.0
    T = np.empty_like(h, dtype=float)
    p = np.empty_like(h, dtype=float)

    # troposphere
    T[mask_trop] = T0 - L * h[mask_trop]
    p[mask_trop] = p0 * (T[mask_trop] / T0) ** (g / (R * L))

    # above troposphere (simple isothermal approximation)
    if np.any(~mask_trop):
        h1 = 11000.0
        T1 = T0 - L * h1
        p1 = p0 * (T1 / T0) ** (g / (R * L))
        T[~mask_trop] = T1
        p[~mask_trop] = p1 * np.exp(-g * (h[~mask_trop] - h1) / (R * T1))

    return p, T

# main calculation
# assume data.velocity is TAS in m/s and data.baroaltitude in meters
tas_ms = data.velocity.copy()  # true airspeed (m/s) under zero-wind assumption
h_m = data.baroaltitude.copy() # barometric altitude in meters

# compute local static pressure and temperature
p_local, T_local = isa_pressure_temperature(h_m)

# local speed of sound and Mach number
a_local = np.sqrt(gamma * R * T_local)
M_local = np.where(a_local > 0, tas_ms / a_local, np.nan)

# impact (stagnation minus static) pressure qc at flight conditions
# qc = p * ( (1 + (gamma-1)/2 * M^2)^(gamma/(gamma-1)) - 1 )
qc = p_local * ( (1.0 + 0.5 * (gamma - 1.0) * M_local**2) ** (gamma / (gamma - 1.0)) - 1.0 )

# invert to find equivalent Mach at sea level (M_cas) that produces same qc
# (1 + qc/p0)^{(gamma-1)/gamma} - 1 = (gamma-1)/2 * M_cas^2
term = (1.0 + qc / p0) ** ((gamma - 1.0) / gamma) - 1.0
M_cas = np.where(term >= 0, np.sqrt(2.0 * term / (gamma - 1.0)), np.nan)

# sea-level speed of sound and CAS (m/s)
a0 = np.sqrt(gamma * R * T0)
cas_ms = M_cas * a0
cas_kts = cas_ms * MPS_TO_KTS

# attach to dataframe
data = data.assign(
    tas_ms=tas_ms,
    cas_ms=cas_ms,
    cas_kts=cas_kts
)

plot_trajectory_clickable2(traffic_data=data, c_feature='cas_kts', label = "callibrated airspeed (kts)",color_dict=color_dict, 
                           alpha=alpha, vmin_plot=100, vmax_plot=350)
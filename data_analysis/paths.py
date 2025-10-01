import pandas as pd
from bluesky.tools.aero import ft, kts
from data_analysis.opensky_tools.plot_trajectory import plot_trajectory, plot_trajectory_line, plot_trajectory_clickable
import numpy as np

data = pd.read_csv("output/65ac_1_direct/flight_output.csv")

data = data.rename(
        columns={
            "time": "timestamp",
            "lat": "latitude",
            "lon": "longitude",
            "heading": "track",
        }
    ).drop(columns=["geoaltitude"])

data = data.assign(
        altitude=data.baroaltitude / ft,
        vertical_rate=data.vertrate / ft * 60,
        groundspeed=data.velocity / kts,
    )

plot_trajectory_clickable(traffic_data=data)
import pandas as pd
import numpy as np
import openap
import seaborn as sns
import matplotlib.pyplot as plt

from data_analysis.opensky_tools.statebased import detect

pd.set_option('future.no_silent_downcasting', True)

class BlueSky_Traffic:
    def __init__(self, index, lat, lon, gs, alt, vs, trk):
        self.id = index
        self.ntraf = len(lat)
        self.lat = lat
        self.lon = lon
        self.gs = gs
        self.alt = alt
        self.vs = vs
        self.trk = trk

def get_conflict_data(data, summary=None, r=9260, vzh=300, tlook=1000, alt_cutoff=1220, save=False, file_name='conflict.csv'):
    df = data
    df = df.sort_values('timestamp')

    id_list = []
    cs_list = []

    lat_array = np.array([])
    lon_array = np.array([])
    gs_array = np.array([])
    alt_array = np.array([])
    vs_array = np.array([])
    trk_array = np.array([])

    dcpa_array = np.array([])
    tcpa_array = np.array([])
    dist_array = np.array([])
    dalt_array = np.array([])
    

    for t in df['timestamp'].unique():
        print(t)
        df_t = df[df['timestamp']==t]

        icao24 = df_t['icao24'].to_list()
        callsign = df_t['callsign'].to_list()
        lat = df_t['latitude'].to_numpy()
        lon = df_t['longitude'].to_numpy()
        gs = df_t['velocity'].to_numpy()
        alt = df_t['baroaltitude'].to_numpy()
        vs = df_t['vertrate'].to_numpy()
        trk = df_t['track'].to_numpy()

        if len(icao24) == 0:
            continue

        ac = BlueSky_Traffic(icao24,lat,lon,gs,alt,vs,trk)

        dcpa, dist, dalt, swconfl, tcpa = detect(ac, ac, r, vzh, tlook)

        id_list.append([icao24[i] for i, j in zip(*np.where(swconfl))])
        cs_list.append([callsign[i] for i, j in zip(*np.where(swconfl))])
        
        lat_array = np.append(lat_array, np.array([ac.lat[i] for i, j in zip(*np.where(swconfl))]))
        lon_array = np.append(lon_array, np.array([ac.lon[i] for i, j in zip(*np.where(swconfl))]))
        gs_array = np.append(gs_array, np.array([ac.gs[i] for i, j in zip(*np.where(swconfl))]))
        alt_array = np.append(alt_array, np.array([ac.alt[i] for i, j in zip(*np.where(swconfl))]))
        vs_array = np.append(vs_array, np.array([ac.vs[i] for i, j in zip(*np.where(swconfl))]))
        trk_array = np.append(trk_array, np.array([ac.trk[i] for i, j in zip(*np.where(swconfl))]))
            
        dcpa_array = np.append(dcpa_array, dcpa[swconfl])
        tcpa_array = np.append(tcpa_array, tcpa[swconfl])
        dist_array = np.append(dist_array, dist[swconfl])
        dalt_array = np.append(dalt_array, dalt[swconfl])

    id_list = [item for sublist in id_list for item in sublist]
    cs_list = [item for sublist in cs_list for item in sublist]

    conf_df = pd.DataFrame({
        'icao24': id_list,
        'callsign': cs_list,
        'lat': lat_array,
        'lon': lon_array,
        'gs': gs_array,
        'alt': alt_array,
        'vs': vs_array,
        'trk': trk_array,
        'dcpa':dcpa_array,
        'tcpa':tcpa_array,
        'dist':dist_array,
        'dalt':dalt_array
    })

    conflict_counts = conf_df[
        (conf_df['dalt'] < vzh) & (conf_df['alt'] > alt_cutoff)
        ].groupby(['icao24', 'callsign']).size().reset_index(name='conflicts')
    intrusion_counts = conf_df[
        (conf_df['dist'] < r) & (conf_df['dalt'] < vzh) & (conf_df['alt'] > alt_cutoff)
    ].groupby(['icao24', 'callsign']).size().reset_index(name='intrusions')

    # summary = summary.merge(conflict_counts, on=['icao24', 'callsign'], how='left')
    # summary = summary.merge(intrusion_counts, on=['icao24', 'callsign'], how='left')
    # summary[['conflicts', 'intrusions']] = summary[['conflicts', 'intrusions']].fillna(0).astype(int)

    if save:
        conf_df.to_csv(file_name)

    return summary, conf_df, conflict_counts, intrusion_counts

def update_conflict_summary(summary, conf_df, r=5000, vzh=300):
    summary = summary[['icao24','callsign']]
    conflict_counts = conf_df.groupby(['icao24', 'callsign']).size().reset_index(name='conflicts')
    intrusion_counts = conf_df[
        (conf_df['dist'] < r) & (conf_df['dalt'] < vzh)
    ].groupby(['icao24', 'callsign']).size().reset_index(name='intrusions')

    summary = summary.merge(conflict_counts, on=['icao24', 'callsign'], how='left')
    summary = summary.merge(intrusion_counts, on=['icao24', 'callsign'], how='left')
    summary[['conflicts', 'intrusions']] = summary[['conflicts', 'intrusions']].fillna(0).astype(int)

    return summary, conf_df
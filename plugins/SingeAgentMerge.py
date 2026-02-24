from bluesky import core, stack, traf, tools, settings 
from stable_baselines3 import SAC
import plugins.SingleAgentCRTools as SACR
import numpy as np
import plugins.MergeTools as MT
import plugins.CommonTools.functions as fn
from plugins.CommonTools.common import runways_schiphol_faf, NM2KM, schiphol, MpS2Kt
import torch

FAF_DISTANCE = 25 #km
STRAIGHT_RWY = True

D_HEADING = 22.5 # deg
D_VELOCITY = 20/3 # kts

M2FEET = 3.2808
GLIDE_SLOPE = 3 #degree
ALT_PER_KM = (np.tan(np.radians(GLIDE_SLOPE))*1000)*M2FEET #feet

DISTANCE_MARGIN_FAF = 2 # km
DISTANCE_MARGIN_RWY = 0.2 # km

DIRECT_MERGE = False

def init_plugin():
    sa_merge = SA_Merge()
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'SA_MERGE',
        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }
    # init_plugin() should always return a configuration dict.
    return config

class SA_Merge(core.Entity):  
    def __init__(self):
        super().__init__()
        
        self.model = SAC.load(f"plugins/SingleAgentMergeTools/model", env=None)

        with self.settrafarrays():
            self.wpt_reach = np.array([])
            self.waypoint_dist = np.array([])

    @core.timed_function(name='SA_Merge', dt=MT.constants.TIMESTEP)
    def update(self):

        rwys = list(set(traf.merge_rwy))

        if 0 in rwys:
            rwys.remove(0)
            
        for runway in rwys:
            ids = [ids for rwy, ids in zip(traf.merge_rwy, traf.id) if rwy == runway] 
            for id in ids:
                self._update_wpt_dist(id,runway)
                if len(ids)>1 and not DIRECT_MERGE:
                    observation = self._get_obs(id,runway,ids)
                    action, _ = self.model.predict(observation, deterministic=True)
                    idx = traf.id2idx(id)
                    self._set_action(action,idx, runway)

                else:
                    self._set_zero_drift(id, runway)
                    req_altitude = self._get_altitude_command(id,runway)
                    idx = traf.id2idx(id)
                    speed = fn.get_speed_at_altitude(traf.alt[idx]) * MpS2Kt
                    stack.stack(f'SPD {id} {speed}')
                    stack.stack(f'ALT {id} {req_altitude}')
        
            for id in ids:
                idx = traf.id2idx(id)
                self._check_waypoint(idx)
            
    
    def create(self, n=1):
        super().create(n)
        self.wpt_reach[-n:] = np.zeros(n)
        self.waypoint_dist[-n:] = np.ones(n)*100_000_000

    def _update_wpt_dist(self,id,runway):
        ac_idx = traf.id2idx(id)
        if self.wpt_reach[ac_idx] == 0: # pre-faf check
            faf_lat, faf_lon = fn.get_point_at_distance(runways_schiphol_faf[runway]['lat'],
                                                runways_schiphol_faf[runway]['lon'],
                                                FAF_DISTANCE,
                                                runways_schiphol_faf[runway]['track']-180)
            wpt_qdr, wpt_dist  = tools.geo.kwikqdrdist(traf.lat[ac_idx], traf.lon[ac_idx], faf_lat, faf_lon)

        else: # post-faf check
            wpt_qdr, wpt_dist  = tools.geo.kwikqdrdist(traf.lat[ac_idx], traf.lon[ac_idx], runways_schiphol_faf[runway]['lat'], runways_schiphol_faf[runway]['lon'])

        self.waypoint_dist[ac_idx] = wpt_dist

    def _get_obs(self, id, runway, ids):
        """
        Observation is the normalized x and y coordinate of the aircraft
        """
        x_r = np.array([])
        y_r = np.array([])
        vx_r = np.array([])
        vy_r = np.array([])
        cos_track = np.array([])
        sin_track = np.array([])
        distances = np.array([])

        idx = traf.id2idx(id)
        idxs = traf.id2idx(ids)
        ac_idx = idx
        ac_hdg = traf.hdg[ac_idx]
        
        if self.wpt_reach[ac_idx] == 0: # pre-faf check
            faf_lat, faf_lon = fn.get_point_at_distance(runways_schiphol_faf[runway]['lat'],
                                                runways_schiphol_faf[runway]['lon'],
                                                FAF_DISTANCE,
                                                runways_schiphol_faf[runway]['track']-180)
            wpt_qdr, wpt_dist  = tools.geo.kwikqdrdist(traf.lat[ac_idx], traf.lon[ac_idx], faf_lat, faf_lon)

        else: # post-faf check
            wpt_qdr, wpt_dist  = tools.geo.kwikqdrdist(traf.lat[ac_idx], traf.lon[ac_idx], runways_schiphol_faf[runway]['lat'], runways_schiphol_faf[runway]['lon'])

        # Get and decompose agent aircaft drift
        drift = ac_hdg - wpt_qdr
        drift = fn.bound_angle_positive_negative_180(drift)
        cos_drift = np.cos(np.deg2rad(drift))
        sin_drift = np.sin(np.deg2rad(drift))

        self.waypoint_dist[ac_idx] = wpt_dist

        # Get agent aircraft airspeed, m/s
        airspeed_cas = traf.cas[ac_idx]
        airspeed_gs = traf.gs[ac_idx]

        # Get speedlimit information
        altitude = traf.alt[ac_idx]
        cas_min, cas_max = fn.get_speed_limits_at_altitude(altitude)

        vx = np.cos(np.deg2rad(ac_hdg)) * traf.gs[ac_idx]
        vy = np.sin(np.deg2rad(ac_hdg)) * traf.gs[ac_idx]

        ac_loc = fn.latlong_to_nm(schiphol, np.array([traf.lat[ac_idx], traf.lon[ac_idx]])) * NM2KM * 1000 # Two-step conversion lat/long -> NM -> m
        x = ac_loc[0]
        y = ac_loc[1]

        ac_loc = SACR.functions.latlong_to_nm(SACR.constants.CENTER, np.array([traf.lat[idx], traf.lon[idx]])) * SACR.constants.NM2KM * 1000 # Two-step conversion lat/long -> NM -> m
        # dist = [SACR.functions.euclidean_distance(ac_loc, SACR.functions.latlong_to_nm(SACR.constants.CENTER, np.array([traf.lat[i], traf.lon[i]])) * SACR.constants.NM2KM * 1000) for i in idxs]
        
        distances = tools.geo.kwikdist_matrix(traf.lat[idx], traf.lon[idx], traf.lat[idxs],traf.lon[idxs])
        dist = distances
        
        ac_idx_by_dist = np.argsort(dist)
        dist_idxs = ac_idx_by_dist
        ac_idx_by_dist = [idxs[i] for i in ac_idx_by_dist]

        for i, dist_idx in zip(range(len(idxs)),dist_idxs):
            int_idx = ac_idx_by_dist[i]
            if int_idx == idx:
                continue
            int_hdg = traf.hdg[int_idx]
            
            # Intruder AC relative position, m
            int_loc = SACR.functions.latlong_to_nm(SACR.constants.CENTER, np.array([traf.lat[int_idx], traf.lon[int_idx]])) * SACR.constants.NM2KM * 1000
            x_r = np.append(x_r, int_loc[0] - ac_loc[0])
            y_r = np.append(y_r, int_loc[1] - ac_loc[1])
            # Intruder AC relative velocity, m/s
            vx_int = np.cos(np.deg2rad(int_hdg)) * traf.gs[int_idx]
            vy_int = np.sin(np.deg2rad(int_hdg)) * traf.gs[int_idx]
            vx_r = np.append(vx_r, vx_int - vx)
            vy_r = np.append(vy_r, vy_int - vy)

            # Intruder AC relative track, rad
            track = np.arctan2(vy_int - vy, vx_int - vx)
            cos_track = np.append(cos_track, np.cos(track))
            sin_track = np.append(sin_track, np.sin(track))

            distances = np.append(distances, dist[dist_idx])

            # very crude normalization for the observation vectors
        observation = {
            "cos(drift)": np.array([cos_drift]),
            "sin(drift)": np.array([sin_drift]),
            "airspeed": np.array([airspeed_cas]),
            "waypoint_dist": np.array([self.waypoint_dist[ac_idx]/250]),
            "faf_reached": np.array([self.wpt_reach[ac_idx]]),
            "x_r": self._pad_arr(x_r, MT.constants.NUM_AC_STATE)/1000000,
            "y_r": self._pad_arr(y_r, MT.constants.NUM_AC_STATE)/1000000,
            "vx_r": self._pad_arr(vx_r, MT.constants.NUM_AC_STATE)/150,
            "vy_r": self._pad_arr(vy_r, MT.constants.NUM_AC_STATE)/150,
            "cos(track)": self._pad_arr(cos_track, MT.constants.NUM_AC_STATE),
            "sin(track)": self._pad_arr(sin_track, MT.constants.NUM_AC_STATE),
            "distances": self._pad_arr(distances, MT.constants.NUM_AC_STATE)/250
        }

        return observation
    
    def _pad_arr(self, arr, target_length):
        arr = np.asarray(arr)
        if len(arr) < target_length:
            pad_len = target_length - len(arr)
            arr = np.concatenate([arr, np.full(pad_len, arr[0])])
        return arr[:target_length]  # ensures max length
        
    def _set_action(self, action, idx, runway):
        dh = action[0] * D_HEADING
        dv = action[1] * D_VELOCITY
        heading_new = fn.bound_angle_positive_negative_180(traf.hdg[idx] + dh)
        speed_new = (traf.cas[idx] + dv) 

        if STRAIGHT_RWY and self.wpt_reach[idx]:
            runway = traf.merge_rwy[idx]
            self._set_zero_drift(traf.id[idx],runway)
        else:
            stack.stack(f"HDG {traf.id[idx]} {heading_new}")

        # limit speed based on altitude
        altitude = traf.alt[idx]
        speed_new = fn.get_speed_at_altitude(altitude,speed_new) * MpS2Kt

        stack.stack(f"SPD {traf.id[idx]} {speed_new}")

        req_altitude = self._get_altitude_command(traf.id[idx],runway)
        
        stack.stack(f'ALT {traf.id[idx]} {req_altitude}')

    def _get_altitude_command(self,agent,runway,total_dist=None):
        """
        returns the altitude[feet] requirement for an aircraft 
        input: total_dist[km]
        """
        if not total_dist:
            ac_idx = traf.id2idx(agent)
            if self.wpt_reach[ac_idx] == 0: # pre-faf check
                faf_lat, faf_lon = fn.get_point_at_distance(runways_schiphol_faf[runway]['lat'],
                                                    runways_schiphol_faf[runway]['lon'],
                                                    FAF_DISTANCE,
                                                    runways_schiphol_faf[runway]['track']-180)
                wpt_qdr, wpt_dist  = tools.geo.kwikqdrdist(traf.lat[ac_idx], traf.lon[ac_idx], faf_lat, faf_lon)
                total_dist = wpt_dist * NM2KM + FAF_DISTANCE
            else: # post-faf check
                _, wpt_dist  = tools.geo.kwikqdrdist(traf.lat[ac_idx], traf.lon[ac_idx], runways_schiphol_faf[runway]['lat'], runways_schiphol_faf[runway]['lon'])
                total_dist = wpt_dist * NM2KM
        
        altitude = total_dist * ALT_PER_KM
        return altitude
    
    def _check_waypoint(self, idx):
        if self.waypoint_dist[idx] < DISTANCE_MARGIN_FAF and self.wpt_reach[idx] != 1:
            self.wpt_reach[idx] = 1
        elif self.waypoint_dist[idx] < DISTANCE_MARGIN_RWY and self.wpt_reach[idx] == 1:
            stack.stack(f"DEL {traf.id[idx]}")

    def _set_zero_drift(self, id, runway):
        ac_idx = traf.id2idx(id)
        ac_hdg = traf.hdg[ac_idx]
        
        if self.wpt_reach[ac_idx] == 0: # pre-faf check
            faf_lat, faf_lon = fn.get_point_at_distance(runways_schiphol_faf[runway]['lat'],
                                                runways_schiphol_faf[runway]['lon'],
                                                FAF_DISTANCE,
                                                runways_schiphol_faf[runway]['track']-180)
            wpt_qdr, wpt_dist  = tools.geo.kwikqdrdist(traf.lat[ac_idx], traf.lon[ac_idx], faf_lat, faf_lon)
            bearing_faf, _ = tools.geo.kwikqdrdist(faf_lat, faf_lon, traf.lat[ac_idx], traf.lon[ac_idx])
            bearing_faf = fn.bound_angle_positive_negative_180(bearing_faf)

        else: # post-faf check
            wpt_qdr, wpt_dist  = tools.geo.kwikqdrdist(traf.lat[ac_idx], traf.lon[ac_idx], runways_schiphol_faf[runway]['lat'], runways_schiphol_faf[runway]['lon'])
            bearing_faf = 0

        stack.stack(f'HDG {id} {wpt_qdr}')
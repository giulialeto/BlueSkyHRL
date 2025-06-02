from bluesky import core, stack, traf, tools, settings 
from stable_baselines3 import SAC
import numpy as np
import plugins.MergeTools as MT
import plugins.CommonTools.functions as fn
from plugins.CommonTools.common import runways_schiphol_faf, NM2KM, schiphol, MpS2Kt
import torch

FAF_DISTANCE = 25 #km

D_HEADING = 22.5 # deg
D_VELOCITY = 20/3 # kts

M2FEET = 3.2808
GLIDE_SLOPE = 3 #degree
ALT_PER_KM = (np.tan(np.radians(GLIDE_SLOPE))*1000)*M2FEET #feet

DISTANCE_MARGIN_FAF = 1 # km
DISTANCE_MARGIN_RWY = 5 # km

def init_plugin():
    merge = Merge()
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'MERGE',
        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }
    # init_plugin() should always return a configuration dict.
    return config

class Merge(core.Entity):  
    def __init__(self):
        super().__init__()
        
        self.model = MT.actor.MultiHeadAdditiveActorBasic(q_dim = 9,
                                                            kv_dim = 7,
                                                            out_dim = 2,
                                                            num_heads = 5,
                                                            dropout_rate=0)
        self.model.load_state_dict(torch.load("plugins/MergeTools/weights_v1/actor.pt"))
        self.model.set_test(True)

        with self.settrafarrays():
            self.wpt_reach = np.array([])
            self.waypoint_dist = np.array([])

    @core.timed_function(name='Merge', dt=MT.constants.TIMESTEP)
    def update(self):

        rwys = list(set(traf.merge_rwy))

        if 0 in rwys:
            rwys.remove(0)
            
        for runway in rwys:
            ids = [ids for rwy, ids in zip(traf.merge_rwy, traf.id) if rwy == runway] 
            observations = self._get_obs(ids,runway)
            obs_array = np.array(list(observations.values()))

            if len(ids)>1:
                action = self.model(torch.FloatTensor(np.array([obs_array])))
                action = np.array(action[0].detach().numpy())
                act_array = np.clip(action, -1, 1)

                for id, action in zip(ids,act_array[0]):
                    idx = traf.id2idx(id)
                    self._set_action(action,idx, runway)
                
                # import code
                # code.interact(local=locals())

            else:
                self._set_zero_drift(ids[0], runway)
        
            for id in ids:
                idx = traf.id2idx(id)
                self._check_waypoint(idx)
            
        
        # here I should write a check for all the runways and corresponding aircraft
        # then follow a similar logic to CR -> see below

        # if len(traf.id) > 0:
        #     observations = self._get_obs()
        #     obs_array = np.array(list(observations.values()))
        #     # obs_array = np.clip(obs_array,-5,5)
        #     action = self.model(torch.FloatTensor(np.array([obs_array])))
        #     action = np.array(action[0].detach().numpy())
        #     act_array = np.clip(action, -1, 1)

        #     for id, action in zip(traf.id,act_array[0]):
        #         idx = traf.id2idx(id)
        #         self._set_action(action,idx)
        # else:
        #     pass
    
    def create(self, n=1):
        super().create(n)
        self.wpt_reach[-n:] = np.zeros(n)
        self.waypoint_dist[-n:] = np.ones(n)*100_000_000

    def _get_obs(self, ids, runway):
        """
        Observation is the normalized x and y coordinate of the aircraft
        """

        # need to find a way to filter this for each merge environment
        # probably add the runway identifier to the traffic object
        # then get the IDs / indices for traffic in rwy a, b etc...
        # for rwy in rwys:
        #   ids = traf.id[traf.rwy == rwy]?
        #   for id in ids:
        #       start obs
        # id = [id for rwy, id in zip(traf.merge_rwy, traf.id) if rwy == '18R']

        obs = []

        for id in ids:
            ac_idx = traf.id2idx(id)
            ac_hdg = traf.hdg[ac_idx]
            
            if self.wpt_reach[ac_idx] == 0: # pre-faf check
                faf_lat, faf_lon = fn.get_point_at_distance(runways_schiphol_faf[runway]['lat'],
                                                    runways_schiphol_faf[runway]['lon'],
                                                    FAF_DISTANCE,
                                                    runways_schiphol_faf[runway]['track']-180)
                wpt_qdr, wpt_dist  = tools.geo.kwikqdrdist(traf.lat[ac_idx], traf.lon[ac_idx], faf_lat, faf_lon)
                bearing_faf, _ = tools.geo.kwikqdrdist(faf_lat, faf_lon, traf.lat[ac_idx], traf.lon[ac_idx])
                bearing_faf = fn.bound_angle_positive_negative_180(bearing_faf-runways_schiphol_faf[runway]['track']-180)

            else: # post-faf check
                wpt_qdr, wpt_dist  = tools.geo.kwikqdrdist(traf.lat[ac_idx], traf.lon[ac_idx], runways_schiphol_faf[runway]['lat'], runways_schiphol_faf[runway]['lon'])
                bearing_faf = 0

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

            observation = {
                "cos(drift)": np.array([cos_drift]),
                "sin(drift)": np.array([sin_drift]),
                "airspeed_cas": np.array([(airspeed_cas-150)/50]),
                "airspeed_gs": np.array([(airspeed_gs-150)/50]),
                "altitude": np.array([(altitude-1500)/3000]),
                "cas_min": np.array([(cas_min-150)/50]),
                "cas_max": np.array([(cas_max-150)/50]),
                "faf_dist": np.array([(wpt_dist-15)/20]),
                "bearing_faf": np.array([(bearing_faf)/30]),
                "x": np.array([x/50000]),
                "y": np.array([y/50000]),
                "vx": np.array([vx/150]),
                "vy": np.array([vy/150])
            }

            obs.append(np.concatenate(list(observation.values())))

        observations = {
            a: o
            for a, o in zip(ids, obs)
        }

        return observations
    
    def _set_action(self, action, idx, runway):
        dh = action[0] * D_HEADING
        dv = action[1] * D_VELOCITY
        heading_new = fn.bound_angle_positive_negative_180(traf.hdg[idx] + dh)
        speed_new = (traf.cas[idx] + dv) 

        # limit speed based on altitude
        altitude = traf.alt[idx]
        speed_new = fn.get_speed_at_altitude(altitude,speed_new) * MpS2Kt

        # print(speed_new)
        stack.stack(f"HDG {traf.id[idx]} {heading_new}")
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
from bluesky import core, stack, traf, tools, settings 
from bluesky.tools.aero import Rearth, ft, fpm, vcas2tas
from stable_baselines3 import SAC
import numpy as np
import math
from matplotlib.path import Path
import plugins.SingleAgentPathPlanningTools as SAPP
from plugins.Sink import Sink
from plugins.CommonTools.functions import get_speed_at_altitude
from plugins.CommonTools.common import MpS2Kt

# PROJECTION_DISTANCE = 25 #km
GLIDE_SLOPE = np.deg2rad(3) #degrees
TARGET_ALTITUDE_PM = 2900 #meters, target altitude at point merge start
ALT_CONTROL_TIMESTEP = 15

SET_TARGET_HEADING = False #if True, commands CR module, otherwise executes heading command directly

def init_plugin():
    singleagentpathplanning = SingleAgentPathPlanning()
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'SINGLEAGENTPATHPLANNING',
        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }
    # init_plugin() should always return a configuration dict.
    return config

class SingleAgentPathPlanning(core.Entity):  
    def __init__(self, altitude=True):
        super().__init__()
        self.model = SAC.load(f"plugins/SingleAgentPathPlanningTools/model", env=None)
        self.sink = Sink()
        self.altitude = altitude
        with traf.settrafarrays():
            traf.target_heading = np.array([])
            traf.distance_remaining = np.array([])

    @core.timed_function(name='SingleAgentPathPlanning', dt=SAPP.constants.TIMESTEP)
    def update(self):
        self.sink.init_sinks()
        for id in traf.id:
            idx = traf.id2idx(id)
            obs = self._get_obs(idx)
            action, _ = self.model.predict(obs, deterministic=True)
            self._set_action(action,idx)
    
    @core.timed_function(dt=ALT_CONTROL_TIMESTEP)
    def update_altitude(self):
        if self.altitude:
            for id in traf.id:
                idx = traf.id2idx(id)
                gs = traf.gs[idx]
                self._get_remaining_distance(idx,traf)
                self._set_altitude(id,idx)

    def create(self, n=1):
        super().create(n)
        self.update()
        traf.target_heading[-n:] = traf.hdg[-n:]

    def _get_obs(self, idx=None, lat=None, lon=None):
        """
        Observation is the normalized x and y coordinate of the aircraft
        """
        if idx is not None:
            brg, dis = tools.geo.kwikqdrdist(SAPP.constants.SCHIPHOL[0], SAPP.constants.SCHIPHOL[1], traf.lat[idx], traf.lon[idx])
        elif lat is not None:
            brg, dis = tools.geo.kwikqdrdist(SAPP.constants.SCHIPHOL[0], SAPP.constants.SCHIPHOL[1], lat, lon)
        else:
            print('Either idx, or [lat lon] pair should be given as input, not nothing.')
        
        x = np.sin(np.radians(brg))*dis*SAPP.constants.NM2KM / SAPP.constants.MAX_DISTANCE
        y = np.cos(np.radians(brg))*dis*SAPP.constants.NM2KM / SAPP.constants.MAX_DISTANCE

        observation = {
            "x" : np.array([x]),
            "y" : np.array([y])
        }
        return observation
    
    def _set_action(self, action, idx):
        bearing = np.rad2deg(np.arctan2(action[0],action[1]))
        if SET_TARGET_HEADING:
            traf.target_heading[idx] = bearing
        elif traf.merge_rwy[idx] == 0:
            speed = get_speed_at_altitude(traf.alt[idx]) * MpS2Kt
            stack.stack(f'SPD {traf.id[idx]} {speed}')
            stack.stack(f'HDG {traf.id[idx]} {bearing}')
        # if traf.merge_rwy[idx] == 0:
        #     traf.ap.selhdgcmd(idx,bearing) # could consider HDG stack command here

    def _project_path(self, action, lat, lon, idx):
        distance = traf.gs[idx]*SAPP.constants.TIMESTEP/1000
        bearing = math.atan2(action[0],action[1])

        ac_lat = np.deg2rad(lat)
        ac_lon = np.deg2rad(lon)

        new_lat = np.rad2deg(self.get_new_latitude(bearing,ac_lat,distance))
        new_lon = np.rad2deg(self.get_new_longitude(bearing,ac_lon,ac_lat,new_lat,distance))
        
        return new_lat, new_lon
    
    def get_new_latitude(self,bearing,lat,radius):
        R = Rearth/1000.
        return math.asin( math.sin(lat)*math.cos(radius/R) +\
                math.cos(lat)*math.sin(radius/R)*math.cos(bearing))
        
    def get_new_longitude(self,bearing,lon,lat1,lat2,radius):
        R = Rearth/1000.
        return lon + math.atan2(math.sin(bearing)*math.sin(radius/R)*\
                        math.cos(lat1),math.cos(radius/R)-math.sin(lat1)*math.sin(lat2))

    @stack.command
    def print_remaining_distance(self, acid: 'acid'):
        print(traf.distance_remaining[acid])

    def _get_remaining_distance(self, idx, traf):
        finished = False
        obs = self._get_obs(idx)
        lat = traf.lat[idx]
        lon = traf.lon[idx]
        distance = 0
        while not finished:
            action, _ = self.model.predict(obs, deterministic=True)
            _lat, _lon = self._project_path(action,lat,lon,idx)
            line_ac = Path(np.array([[lat,lon],[_lat,_lon]]))
            for line_sink in self.sink.line_sinks:
                if line_sink.intersects_path(line_ac):
                    finished = True
                    ls = line_sink.vertices
                    dis_orig = 1000
                    for l in ls:
                        _, dis = tools.geo.kwikqdrdist(l[0], l[1], lat, lon)
                        if dis < dis_orig:
                            dis_orig = dis
                    distance += dis_orig*SAPP.constants.NM2KM

            if distance > 1000:
                finished = True

            if not finished:
                distance += traf.gs[idx]*SAPP.constants.TIMESTEP/1000
                lat, lon = _lat, _lon
                obs = self._get_obs(lat=lat,lon=lon)
            
        traf.distance_remaining[idx] = distance
        # if idx == 0:
        #     print(traf.distance_remaining[0], traf.alt[0])

    def _set_altitude(self,id,idx):
        if traf.distance_remaining.any():
            distance_remaining = max(0,traf.distance_remaining[idx] - (traf.gs[idx]*ALT_CONTROL_TIMESTEP)/1000)
            # distance_remaining = max(0,traf.distance_remaining[idx] - (traf.gs[idx]*SAPP.constants.TIMESTEP)/1000)
            target_altitude = TARGET_ALTITUDE_PM + distance_remaining*np.tan(GLIDE_SLOPE)*1000 # in meters
            vert_speed = np.tan(GLIDE_SLOPE)*traf.gs[idx] # in meters/sec
            if target_altitude < traf.alt[idx]:
                stack.stack(f"ALT {id} {target_altitude/ft} {100*vert_speed/fpm}")
            
            ## block that check how long before PM ac at target altitude
            # if traf.alt[idx] == TARGET_ALTITUDE_PM:
            #     dis_orig = 1000
            #     for line_sink in self.sink.line_sinks:
            #         ls = line_sink.vertices
            #         for l in ls:
            #             _, dis = tools.geo.kwikqdrdist(l[0], l[1], traf.lat[idx], traf.lon[idx])
            #             if dis < dis_orig:
            #                 dis_orig = dis
            #     print(dis_orig*SAPP.constants.NM2KM)
            # if idx == 0:
            #     print(traf.distance_remaining[0], traf.alt[0], target_altitude)
        else:
            print('altitude control plugin only works when traf.distance_remaining exists')
            print('try including a pathplanning plugin')




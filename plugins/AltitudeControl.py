from bluesky import core, stack, traf, tools, settings 
from bluesky.tools.aero import ft, fpm
from stable_baselines3 import SAC
import numpy as np
import math
from matplotlib.path import Path
import plugins.SingleAgentPathPlanningTools as SAPP
from plugins.Sink import Sink

def init_plugin():
    altitudecontrol = AltitudeControl()
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'ALTITUDECONTROL',
        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
        }
    # init_plugin() should always return a configuration dict.
    return config

class AltitudeControl(core.Entity):  
    def __init__(self, glide_slope=3):
        super().__init__()
        self.glide_slope = np.deg2rad(glide_slope) # angle in radians

    @core.timed_function(name='AltitudeControl', dt=SAPP.constants.TIMESTEP)
    def update(self):
        for id in traf.id:
            idx = traf.id2idx(id)
            if traf.distance_remaining.any():
                distance_remaining = traf.distance_remaining[idx] - (traf.gs[idx]*SAPP.constants.TIMESTEP)/1000
                target_altitude = distance_remaining*np.tan(self.glide_slope)*1000 # in meters
                vert_speed = np.tan(self.glide_slope)*traf.gs[idx] # in meters/sec
                if target_altitude < traf.alt[idx]:
                    stack.stack(f"ALT {id} {target_altitude/ft} {100*vert_speed/fpm}")
                if idx == 0:
                    print(traf.distance_remaining[0], traf.alt[0], target_altitude)
            else:
                print('altitude control plugin only works when traf.distance_remaining exists')
                print('try including a pathplanning plugin')

    def create(self, n=1):
        super().create(n)



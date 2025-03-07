import numpy as np


def schedule(endtime):
    # T_supply profile
    tevap = 15 * np.ones(int(endtime))
    tevap[round(endtime / 6):round(endtime / 6) * 2] = 10
    tevap[round(endtime / 6) * 2 + 1:round(endtime / 6) * 3] = 20
    tevap[round(endtime / 6) * 3 + 1:round(endtime / 6) * 4] = 12.5
    tevap[round(endtime / 6) * 4 + 1:round(endtime / 6) * 5] = 17.5
    # T_set profile
    t_set = 22 * np.ones(int(endtime))
    # vdot_air profile
    cfmtoms = 0.00047194745
    vdot_air = 424 * cfmtoms * np.ones(int(endtime))  # M
    # vdot_air[round(endtime/4):round(endtime/4)*3] = 442*cfmtoms  # H
    # vdot_air[round(endtime/4)*3+1:] = 399*cfmtoms  # L
    # Occupancy schedule
    n_people = 5 * np.ones(int(endtime))
    return tevap, t_set, vdot_air, n_people


class BuildingModel:
    def __init__(self, dt, l_room=3, w_room=3, h_room=3, rho_a=1.29, cp_a=1029, u_wall=2.35, shg=73, lhg=59):
        self.l_room = l_room
        self.w_room = w_room
        self.h_room = h_room
        # air properties
        self.rho_a = rho_a  # air density [kg/m^3]
        self.cp_a = cp_a  # specific heat of air [J/kg]
        # room properties
        self.u_wall = u_wall  # [W/m^2-K]
        # occupancy
        self.shg = shg  # sensible heat gain [W]
        self.lhg = lhg  # latent heat gain [W]
        # time interval
        self.dt = dt

    def datagen(self, n_people, tevap, t_set, vdot_air, t_amb):
        # room dimension
        vol_room = self.l_room * self.w_room * self.h_room
        area_wall = 2 * self.h_room * (self.l_room + self.w_room)
        # Internal heat gain
        hg = n_people * (self.shg + self.lhg)
        # time setup
        endtime = 8*60*60/self.dt
        # Initialize
        ac_onoff = np.zeros(int(endtime)) # initial AC on
        power = np.zeros(int(endtime)) # initial AC PC
        t_room = np.zeros(int(endtime+1))
        t_room[0] = t_amb[0]
        #  baseline: on/off control
        for i in range(int(endtime)):
            if i > 0:
                ac_onoff[i] = ac_onoff[i-1]
            if (t_room[i] >= (t_set[i]+2)).any():
                ac_onoff[i] = 1 # AC on
            if (t_room[i] <= (t_set[i]-2)).any():
                ac_onoff[i] = 0  # AC off
            if ac_onoff[i] == 0:
                vdot_air[i] = 0
            t_room[i+1] = t_room[i]+self.dt/(self.rho_a*vol_room*self.cp_a)*(self.u_wall*area_wall*(t_amb[i]-t_room[i])+
                                                                      hg[i]-self.rho_a*self.cp_a*vdot_air[i]*(t_room[i]-tevap[i]))
            power[i] = ac_onoff[i] * 1200/15 * (25 - tevap[i])
        return t_room, power, ac_onoff

    def testbed(self, n_people, tevap, t_room, vdot_air, t_amb):
        # Room dimension
        vol_room = self.l_room * self.w_room * self.h_room
        area_wall = 2 * self.h_room * (self.l_room + self.w_room)
        # Internal heat gain
        hg = n_people * (self.shg + self.lhg)
        t_room_n = t_room + self.dt / (self.rho_a * vol_room * self.cp_a) * (self.u_wall * area_wall * (t_amb - t_room)
                                                                             + hg - self.rho_a * self.cp_a * vdot_air * (t_room - tevap))
        return t_room_n

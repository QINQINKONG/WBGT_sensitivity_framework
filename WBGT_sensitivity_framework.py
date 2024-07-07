"""
This python file describes how to use the sensitivity framework of wet-bulb globe temperature (WBGT) developed by Kong and Huber (2024). WBGT is defined as:
WBGT=0.7*Tnw + 0.2*Tg + 0.1*Ta
where Tnw, Tg and Ta represent natural wet-bulb temperature, black globe temperature, and dry-bulb temperature respectively.
Starting from the energy balance equations of the black globe and wet-bulb sensor, we derive the sensitivities of Tg and Tnw to changes in each meterological variable (please refer to Kong and Huber (2024) for details):

dTg = alpha_Ta*dTa + alpha_SR*dSRg + alpha_h*dhcg
dTnw = beta_Ta*dTa + beta_q*dq + beta_SR*dSRw + beta_kappa * dkappa + beta_h*dhcw + beta_Ps*dPs

The differential terms in both equations are defined as:
- dTg: changes in Tg
- dTnw: changes in Tnw
- dTa: changes in Ta
- dSRg: changes in solar radiative heating effect on the black globe; note that SRg contains multiple parameters including downward (both direct and diffuse) and surface reflected solar radiation, solar zenith angle, globe albedo, etc. Please refer to Liljegren et al.(2008) for its formulation. SRg can be calculated by the function "calSRg" provided within this file.
- dhcg: changes in convective heat transfer coefficient of the black globe. Since hcg is primarily affected by wind speed, this term evaluates the effects of wind speed changes. hcg can be calculated by the function "calhcg" provided within this file.
- dq: changes in specific humidity
- dSRw: changes in solar radiative heating effect on the wet-bulb sensor. Please refer to Liljegren et al.(2008) for its formulation. SRw can be calculated by the function "calSRw" provided within this file.
- dkappa: changes in convective mass tansfer coeffcient. Since kappa is primarily affected by wind speed, this term evaluates the effects of wind speed changes. kappa can be calculated by the function "calkappa" provided within this file.
- dhcw: changes in convective heat transfer coefficient of the wick cylinder. Since hcw is primarily affected by wind speed, this term also evaluates the effects of wind speed changes. hcw can be calculated by the function "calhcw" provided within this file.
- dPs: changes in surface pressure.

The sensitivity coefficients can be calculated using corresponding functions ("calX" where X is the name of the coefficient) provided within this file, except alpha_Ta which is a constant 1.

Since dWBGT=0.7*dTnw + 0.2*dTg + 0.1*dTa, we have

dWBGT = (0.7*beta_Ta + 0.2*alpha_Ta + 0.1) * dTa                      # the effects of temperature changes
        + 0.7*beta_q*dq                                               # the effects of specific humidity changes
        + 0.7*beta_SR*dSRw+0.2*alpha_SR*dSRg                          # the effects of solar radiation changes
        + 0.7*beta_kappa*dkappa + 0.7*beta_h*dhcw + 0.2*alpha_h*dhcg  # the effects of wind speed changes
        + 0.7*beta_Ps*dPs                                             # the effects of surface pressure changes
which evaluates the response of WBGT to changes in each meteorological variable.

The framework is originally nonlinear with state-dependent sensitivity coefficients. However, it can be linearized by calculating the sensitivity coefficients against certain reference point. Since linearization introduces biases, it needs to be validated in practial application.


References:

Kong, Q. & Huber, M. A linear sensitivity framework to understand the drivers of the wet-bulb globe temperature changes. Under review at Journal of Climate.

Liljegren, J. C., Carhart, R. A., Lawday, P., Tschopp, S. & Sharp, R. Modeling the Wet Bulb Globe Temperature Using Standard Meteorological Measurements. Journal of Occupational and Environmental Hygiene 5, 645–655 (2008).

"""

import xarray as xr
import numpy as np
from numba import vectorize

# define constants

# globe constants
diamglobe = 0.0508 # diameter of globe (m)
emisglobe = 0.95 # emissivity of globe
albglobe = 0.05 # albedo of globe
albsfc=0.45 # albedo of ground surface

#wick constants
emiswick = 0.95 # emissivity of the wick
albwick = 0.4 # albedo of the wick
diamwick = 0.007 # diameter of the wick
lenwick = 0.0254 # length of the wick

# physical constants:
stefanb = 0.000000056696  # stefan-boltzmann constant
mair = 28.97 # molecular weight of dry air (grams per mole)
mh2o = 18.015 # molecular weight of water vapor (grams per mole)
rgas = 8314.34 # ideal gas constant (J/kg mol Â· K)
rair = rgas * (mair**(-1))
cp = 1003.5 # Specific heat capacity of air at constant pressure (JÂ·kg-1Â·K-1)
Pr = cp * ((cp + 1.25 * rair)**(-1)) # Prandtl number 

@vectorize
def esat(tas,ps):
    # tas: 2m air temperature (K)
    # ps: surface pressure (Pa)
    # return saturation vapor pressure (Pa)
    if tas>273.15:
        es=(1.0007 + (3.46*10**(-6) * ps/100))*(611.21 * np.exp(17.502 * (tas - 273.15) *((tas - 32.18)**(-1))))
    else:
        es=(1.0003 + (4.18*10**(-6) * ps/100))*(611.15 * np.exp(22.452 * (tas - 273.15) * ((tas - 0.6)**(-1))))
    return es

@vectorize
def desat_dT(tas,ps):
    # tas: 2m air temperature (K)
    # ps: surface pressure (Pa)
    # return derivative of saturation vapor pressure wrt to temperature 
    if tas>273.15:
        desdT=esat(tas,ps)*17.502*(273.15-32.18)/((tas-32.18)**2)
    else:
        desdT=esat(tas,ps)*22.452*(273.15-0.6)/((tas-0.6)**2)
    return desdT

def dqs_dT(tas,ps):
    # tas: 2m air temperature (K)
    # ps: surface pressure (Pa)
    # return derivative of saturated specific humidity wrt to temperature 
    es=esat(tas,ps)
    desdT=desat_dT(tas,ps)
    result=0.622/(ps-es+0.622*es)*desdT-(0.622*es)/((ps-es+0.622*es)**2)*(0.622-1)*desdT
    return result
def dqs_dps(tas,ps):
    es=esat(tas,ps)
    result=-0.622*es/((ps-es+0.622*es)**2)
    return result
def qsat(tas,ps):
    es=esat(tas,ps)
    qs=0.622*es/(ps-es+0.622*es)
    return qs

def h_evap(tas):
    # tas: air temperature (K)
    # return heat of evaporation (J/(kg K))
    return ((313.15 - tas)/30. * (-71100.) + 2.4073e6 )

def viscosity(tas):
    # tas: air temperature (K)
    # return air viscosity (kg/(m s))
    omega=1.2945-tas/1141.176470588
    visc = 0.0000026693 * (np.sqrt(28.97 * tas)) * ((13.082689 * omega)**(-1))
    return visc
def thermcond(tas):
    # tas: air temperature (K)
    # return thermal conductivity of air (W/(m K))
    tc = (cp + 1.25 * rair) * viscosity(tas)
    return tc
def diffusivity(tas,ps):
    # tas: air temperature (K)
    # ps: surface pressure (Pa)
    # return diffusivity of water vapor in air (m2/s)
    return 2.471773765165648e-05 * ((tas *0.0034210563748421257) ** 2.334) * ((ps / 101325)**(-1))

def calhcw(Ta,Tnw,ps,sfcwind):
    # Tnw: natural wet-bulb temperature (K)
    # Ta: 2-meter air temperature (K)
    # ps: surface pressure (Pa)
    # sfcwind: 2 meter wind (m/s)
    # return convective heat transfer coefficient for a long cylinder (W/(m2 K))
    Tf = (Ta+Tnw)/2  # film temperature
    thermcon = thermcond(Tf) # thermal conductivity
    density = ps * ((rair * Tf)**(-1)) # air density
    Re = sfcwind * density * diamwick * ((viscosity(Tf))**(-1)) # Reynolds number
    Nu = 0.281 * (Re ** 0.6) * (Pr ** 0.44) # Nusselt number
    hcw = Nu * thermcon * (diamwick**(-1))
    return hcw

def calhcg(Ta, Tg, ps, sfcwind):
    # Tg: black globe temperature (K)
    # Ta: 2-meter air temperature (K)
    # ps: surface pressure (Pa)
    # sfcwind: 2 meter wind (m/s)
    # return convective heat tranfer coefficient for flow around a sphere (W/(m2 K))
    Tf = (Ta+Tg)/2 # film temperature
    thermcon = thermcond(Tf) # thermal conductivity
    density = ps * ((rair * Tf)**(-1)) # density
    Re = sfcwind * density * diamglobe * ((viscosity(Tf))**(-1)) # Reynolds number
    Nu = 2 + 0.6 * np.sqrt(Re) * (Pr**0.3333) # Nusselt number
    hcg = Nu * thermcon * (diamglobe**(-1))
    return hcg

def conv_mass(Ta,Tnw,ps,sfcwind):
    # Tnw: natural wet-bulb temperature (K)
    # Ta: 2-meter air temperature (K)
    # ps: surface pressure (Pa)
    # sfcwind: 2 meter wind (m/s)
    # return convective mass transfer coefficient for flow around a cylinder
    Tf=(Ta+Tnw)/2 # film temperature
    hcw=calhcw(Ta,Tnw, ps, sfcwind) # convective heat transfer coefficient for the wick cylinder
    density=ps/(Tf*rair) # air density
    Sc=viscosity(Tf)*((density*diffusivity(Tf,ps))**(-1)) # Schmidt number
    kx=hcw/(cp*mair)*((Pr/Sc)**0.56) # convective mass transfer coefficient
    return kx

def calkappa(Ta,Tnw,ps,sfcwind):
    # Tnw: natural wet-bulb temperature (K)
    # Ta: 2-meter air temperature (K)
    # ps: surface pressure (Pa)
    # sfcwind: 2 meter wind (m/s)
    # return scaled convective mass transfer coefficient for flow around a cylinder
    Tf=(Ta+Tnw)/2 # film temperature
    kx=conv_mass(Ta,Tnw,ps,sfcwind) # convective mass transfer coefficient 
    kappa=kx*mh2o*h_evap(Tf)/0.622 # scaled convective mass transfer coeffcient
    return kappa

def calhrg(Ta,Tg):
    # Tg: black globe temperature (K)
    # Ta: 2-meter air temperature (K)
    # return thermal radiative heat transfer coefficient for the black globe
    hrg=stefanb*emisglobe*(Tg**2+Ta**2)*(Tg+Ta)
    return hrg

def calhrw(Ta,Tnw):
    # Tnw: natural wet-bulb temperature (K)
    # Ta: 2-meter air temperature (K)
    # return thermal radiative heat transfer coefficient for the wet bulb
    hrw=stefanb*emiswick*(Tnw**2+Ta**2)*(Tnw+Ta)
    return hrw


def calalpha_SR(Ta, Tg, ps, sfcwind):
    # Tg: black globe temperature (K)
    # Ta: 2-meter air temperature (K)
    # ps: surface pressure (Pa)
    # sfcwind: 2 meter wind (m/s)
    # return the sensitivity coefficient of Tg to solar radiation variations
    hcg=calhcg(Ta, Tg, ps, sfcwind) # convective heat transfer coefficient for the black globe
    hrg=calhrg(Ta,Tg) # thermal radiative heat transfer coefficient for the black globe
    alpha_SR=1/(hcg+hrg)
    return alpha_SR

def calalpha_h(Ta,Tg,ps,sfcwind):
    # Tg: black globe temperature (K)
    # Ta: 2-meter air temperature (K)
    # ps: surface pressure (Pa)
    # sfcwind: 2 meter wind (m/s)
    # return the sensitivity coefficient of Tg to variations in convective heat transfer coefficient of the globe
    hcg=calhcg(Ta, Tg, ps, sfcwind) # convective heat transfer coefficient for the black globe
    hrg=calhrg(Ta,Tg) # thermal radiative heat transfer coefficient for the black globe
    alpha_h=(Ta-Tg)/(hcg+hrg)
    return alpha_h

def calbeta_Ta(Ta,Tnw,ps,sfcwind):
    # Tnw: natural wet-bulb temperature (K)
    # Ta: 2-meter air temperature (K)
    # ps: surface pressure (Pa)
    # sfcwind: 2 meter wind (m/s)
    # return the sensitivity coefficient of Tnw to temperature variations
    hrw=calhrw(Ta,Tnw) # thermal radiative heat transfer coefficient for the wick
    hcw=calhcw(Ta,Tnw,ps,sfcwind) # convective heat transfer coefficient for the wick
    kappa=calkappa(Ta,Tnw,ps,sfcwind) # scaled convective mass transfer coefficient for the wick
    xi=hcw+hrw+kappa*dqs_dT(Tnw,ps)
    beta_Ta=(hcw+hrw)/xi
    return beta_Ta

def calbeta_q(Ta,Tnw,ps,sfcwind):
    # Tnw: natural wet-bulb temperature (K)
    # Ta: 2-meter air temperature (K)
    # ps: surface pressure (Pa)
    # sfcwind: 2 meter wind (m/s)
    # return the sensitivity coefficient of Tnw to specific humidity variations
    
    hrw=calhrw(Ta,Tnw) # thermal radiative heat transfer coefficient for the wick
    hcw=calhcw(Ta,Tnw,ps,sfcwind) # convective heat transfer coefficient for the wick
    kappa=calkappa(Ta,Tnw,ps,sfcwind) # scaled convective mass transfer coefficient for the wick
    xi=hcw+hrw+kappa*dqs_dT(Tnw,ps)
    beta_q=kappa/xi
    return beta_q

def calbeta_SR(Ta,Tnw,ps,sfcwind):
    # Tnw: natural wet-bulb temperature (K)
    # Ta: 2-meter air temperature (K)
    # ps: surface pressure (Pa)
    # sfcwind: 2 meter wind (m/s)
    # return the sensitivity coefficient of Tnw to solar radiation variations
    hcw=calhcw(Ta,Tnw,ps,sfcwind) # convective heat transfer coefficient for the wick
    hrw=calhrw(Ta,Tnw) # thermal radiative heat transfer coefficient for the wick
    kappa=calkappa(Ta,Tnw,ps,sfcwind) # scaled convective mass transfer coefficient for the wick
    xi=hcw+hrw+kappa*dqs_dT(Tnw,ps)
    beta_SR=1/xi
    return beta_SR

def calbeta_h(Ta,Tnw,ps,sfcwind):
    # Tnw: natural wet-bulb temperature (K)
    # Ta: 2-meter air temperature (K)
    # ps: surface pressure (Pa)
    # sfcwind: 2 meter wind (m/s)
    # return the sensitivity coefficient of Tnw to variations in convective heat transfer coefficient of the wick
    hrw=calhrw(Ta,Tnw) # thermal radiative heat transfer coefficient for the wick
    hcw=calhcw(Ta,Tnw,ps,sfcwind) # convective heat transfer coefficient for the wick
    kappa=calkappa(Ta,Tnw,ps,sfcwind) # scaled convective mass transfer coefficient for the wick
    xi=hcw+hrw+kappa*dqs_dT(Tnw,ps)    
    beta_h=(Ta-Tnw)/xi
    return beta_h

def calbeta_kappa(Ta,Tnw,q,ps,sfcwind):
    # Tnw: natural wet-bulb temperature (K)
    # Ta: 2-meter air temperature (K)
    # q: specific humidity (kg/kg)
    # ps: surface pressure (Pa)
    # sfcwind: 2 meter wind (m/s)
    # return the sensitivity coefficient of Tnw to variations in kappa (scaled mass transfer coefficient)
    
    hrw=calhrw(Ta,Tnw) # thermal radiative heat transfer coefficient for the wick
    hcw=calhcw(Ta,Tnw,ps,sfcwind) # convective heat transfer coefficient for the wick
    kappa=calkappa(Ta,Tnw,ps,sfcwind) # scaled convective mass transfer coefficient for the wick
    xi=hcw+hrw+kappa*dqs_dT(Tnw,ps)
    qw=qsat(Tnw,ps)
    beta_kappa=(q-qw)/xi
    return beta_kappa


def calbeta_Ps(Ta,Tnw,ps,sfcwind):
    # Tnw: natural wet-bulb temperature (K)
    # Ta: 2-meter air temperature (K)
    # ps: surface pressure (Pa)
    # sfcwind: 2 meter wind (m/s)
    # return the sensitivity coefficient of Tnw to variations in surface pressure
    
    hrw=calhrw(Ta,Tnw) # thermal radiative heat transfer coefficient for the wick
    hcw=calhcw(Ta,Tnw,ps,sfcwind) # convective heat transfer coefficient for the wick
    kappa=calkappa(Ta,Tnw,ps,sfcwind) # scaled convective mass transfer coefficient for the wick
    xi=hcw+hrw+kappa*dqs_dT(Tnw,ps)    
    beta_Ps=-1*kappa*dqs_dps(Tnw,ps)/xi
    return beta_Ps

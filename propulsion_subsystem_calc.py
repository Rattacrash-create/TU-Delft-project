import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Constants & vehicle data
# --------------------------
dt = 0.1  # s integration increment
T = 25.0*2  # [N] two Aerojet GR-22 engines (used in the final design)
I_sp = 257  # [s] specific impulse (final monoprop design)
v_exhaust = I_sp * 9.81  # [m/s]
m_dot = T / v_exhaust    # [kg/s]

# MTM mass
m_solar = 7.25 * 41  # [kg]
m_ion = 8.3 * 4 + 48 * 2 # [kg]
m_fix_t = 0.1974 * 250 + 25.513 # [kg]
m_fix_mtm = m_solar + m_ion + m_fix_t # [kg]
c_res = 0.2 # residual propellant/total propellant fraction
c_t = 0.2 # tank structure fraction from previous report
delta_v_mtm = 5100 # [m/s] delta v required from kickstage
v_exhaust_mtm = 9.81 * 4300 # [m/s]

# orbiter
m_po = 512.32 # [kg] empty mass orbiter excluding tanks
c_tO = 0.13 # tank mass/propellant mass fraction

# orbit geometry
R_m = 2240e3   # central body radius (m)
h = 750e3      # altitude (m)
mu = 2.20319e13 # gravitational paramter Mercury
a0 = 91615e3   # initial semi-major axis (m)

# burn assumptions
v_range = 5 # true anomaly range

# tank pressurerization
rho = 1470 # kg/m^3  density AF-M315E
P_0 = 310 * 10**5 # [Pa] initial tank pressure 
P_f = 27.6 * 10**5 # [Pa] min prop fed pressure 
R = 8.31446 # universal gas constant
M_N2 = 28.0134e-3 # molar mass nitrogen

# tank structure 
sigma_ult_carbon = 5.49e9 # ultimate tensile strenght T800 carbon fiber 
SF = 1.5 # safety factor 
V_f = 0.6 # fiber volume fraction
eta = 0.7 # translation efficiency
rho_carbon = 1810 # [kg/m^3] density T800 carbon fibers
rho_resin = 1250 # [kg/m^3] density resin
rho_Ti = 4470 # [kg/m^3] Ti6AlV titanium alloy density
sigma_yield_Ti = 1170e6 # [Pa] Ti6AlV max yield strengh
t_Ti = 2.6e-3 # [m] titanium liner thickness (taken from space shuttle gas tanks)

# maximum allowable stresses
sigma_allow = (sigma_ult_carbon * V_f * eta) / SF 
sigma_allow_Ti = sigma_yield_Ti / SF 

# -------------------------------------------------------
# Functions
# -------------------------------------------------------
def integrate_burn(m0, T, v_exhaust, v_req, max_time=None):
    """Integrate a burn with thrust T until Δv ≥ v_req or until max_time."""
    m = m0
    v_accum = 0.0
    t = 0.0
    while v_accum < v_req and (max_time is None or t < max_time):
        a = T / m
        v_accum += a * dt
        m -= m_dot * dt
        t += dt
        if m <= 0:
            raise RuntimeError("Mass became non-physical during integration.")
    return v_accum, m, t

def orbit_from_speed_at_r(v, r):
    """Return semi-major axis a and eccentricity e given speed v at radius r."""
    eps = v**2 / 2.0 - mu / r
    if eps >= 0:  # hyperbolic/parabolic
        return None, None
    a = - mu / (2.0 * eps)
    e = 1.0 - r / a  # assume periapsis at radius r
    return a, e

def plot_orbits_periapsis_down(a_list, e_list, n_points=600):
    """Plot ellipses with periapsis at bottom of the figure."""
    fig, ax = plt.subplots(figsize=(8, 8))
    R = np.array([[0, 1], [-1, 0]])  # rotation matrix for -90° rotation

    for i, (a, e) in enumerate(zip(a_list, e_list)):
        if a is None or e is None:
            continue
        theta = np.linspace(0, 2*np.pi, n_points)
        r = (a * (1 - e**2)) / (1 + e * np.cos(theta))
        coords = np.vstack([r*np.cos(theta), r*np.sin(theta)])
        rotated = R @ coords
        x_r, y_r = rotated[0]/1e3, rotated[1]/1e3  # km for clarity
        ax.plot(y_r, x_r, label=f"Orbit {i+1}: a={a/1e3:.0f} km, e={e:.3f}")
        # mark periapsis
        peri = np.array([a*(1-e), 0.0])
        peri_rot = R @ peri
        ax.plot(peri_rot[1]/1e3, peri_rot[0]/1e3, 'ro')

    ax.set_aspect('equal', 'box')
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title("Orbit Circularization (periapsis aligned)")
    ax.grid(True)
    plt.show()

# -------------------------------------------------------
# Compute Δv requirements
# -------------------------------------------------------
r_init = R_m + h # pericenter/final orbit radius
V_elliptical = np.sqrt(2 * (mu / r_init - mu / (2 * a0)))
V_circular = np.sqrt(mu / r_init)
v_disturbance = 90 # [m/s] gravity, solar radiation pressure etc.
v_req_total = abs(V_elliptical - V_circular)


# Mass budget calculations (rewritten rocket equation)
m_fo = (m_po*((np.e**((v_req_total+v_disturbance)/v_exhaust))-1)) / (
    1 + c_res + c_tO - (c_res+c_tO) * np.e**((v_req_total+v_disturbance)/v_exhaust)  # [kg] propellant mass orbiter
)
mo = m_fo*(1 + c_tO + c_res) + m_po   # [kg] total mass orbiter
m_f_mtm = ((m_fix_mtm + mo) * (np.e**(delta_v_mtm / v_exhaust_mtm) - 1) /
           (1 + c_res + c_t - np.e**(delta_v_mtm / v_exhaust_mtm)*(c_t + c_res))) # [kg] propellant mass kickstage
m_mtm = m_f_mtm * (1 + c_t + c_res) + m_fix_mtm # [kg] total mass kickstage
m_f_mtm_2 = m_f_mtm * (1 + c_res) # [kg] total kickstage fuel mass (including reserve)
m_fo_2 = m_fo * (1 + c_res) # [kg] total orbiter fuel mass (with reserve)
m_total = m_mtm + mo # [kg] total launch mass

# -------------------------------------------------------
# Burn loop simulation
# -------------------------------------------------------
mass = mo
a_list, e_list = [], [] # initializing list of semi major axis and eccentricity
v_done, t_tot, t_orbit = 0.0, 0.0, 0.0
remaining = v_req_total

while remaining > 1e-3:# accounting for numerical errors
    # simulating delta v change for burns that cover equal true anomaly window
    # calculating new orbit from there
    # makes sure that impulsive shot assumption stays valid 

    # current orbit
    current_speed = V_elliptical - v_done
    a_curr, e_curr = orbit_from_speed_at_r(current_speed, r_init)
    a_list.append(a_curr)
    e_list.append(max(0.0, min(0.9999, e_curr)))

    # orbital period [s]
    period = 2 * np.pi * np.sqrt(a_curr**3 / mu)

    # Symmetric true anomaly window around periapsis
    v_1 = -v_range * np.pi/180
    v_2 =  v_range * np.pi/180

    # Convert to eccentric anomaly
    E_1 = 2*np.arctan(np.tan(v_1/2)/np.sqrt((1+e_curr)/(1-e_curr)))
    E_2 = 2*np.arctan(np.tan(v_2/2)/np.sqrt((1+e_curr)/(1-e_curr)))

    # Mean anomalies
    M_1 = E_1 - e_curr*np.sin(E_1)
    M_2 = E_2 - e_curr*np.sin(E_2)

    # Ensure positive difference
    if M_2 < M_1:
        M_2 += 2*np.pi

    # Burn time in seconds
    burn_time = (M_2 - M_1)/(2*np.pi) * period

   
    # integrate burn
    dv_window, mass, t_used = integrate_burn(mass, T, v_exhaust, remaining, max_time=burn_time)
    v_done += dv_window
    remaining = v_req_total - v_done
    t_tot += t_used
    t_orbit += period

# calculating required pressurization gas tank volume 
V_o = m_fo_2 / rho # [m^3] propellant tank volume
V_s = V_o / (P_0 / P_f - 1) # [m^3] gas tank volume
m_N2 = (P_0*V_s*M_N2)/(R*(273.15 + 27)) # [kg] nitrogen gas mass (assumed to be non compressible)

# monoprop variant 
R_tank = ((3/4)*(V_s/np.pi))**(1/3) # [m] tank radius from volume
t_comp = (P_0*(R_tank + t_Ti))/(sigma_allow*V_f) # [m] required wall thickness using hoop stress relation
V_comp = (4/3) * np.pi * ((R_tank + t_comp + t_Ti)**3 - (R_tank + t_Ti)**3) # [m^3] composite volume
V_Ti = (4/3) * np.pi * ((R_tank + t_Ti)**3 - R_tank**3) # [m^3] titanium volume from tank geometry and fixed liner thickness
M_comp = V_comp * V_f * rho_carbon + V_comp * (1 - V_f) * rho_resin # [kg] mass composite
M_Ti = V_Ti * rho_Ti # [kg] mass titanium liner
M_tot = M_comp + M_Ti # [kg] total mass

R_tank_prop = ((3/4)*(V_o/np.pi))**(1/3) # [m] fuel tank radius
t_Ti_prop = (P_f*R_tank_prop)/(sigma_allow_Ti)
V_Ti_prop = (4/3) * np.pi * ((R_tank_prop + t_Ti_prop)**3 - R_tank_prop**3)
M_Ti_prop = V_Ti_prop * rho_Ti 

# biprop variant (same just based on manual tank volume calculation result)
R_tank_2 = ((3/4)*(0.0262466/np.pi))**(1/3)
t_comp_2 = (P_0*(R_tank_2 + t_Ti))/(sigma_allow*V_f)
V_comp_2 = (4/3) * np.pi * ((R_tank_2 + t_comp_2 + t_Ti)**3 - (R_tank_2 + t_Ti)**3)
V_Ti = (4/3) * np.pi * ((R_tank_2 + t_Ti)**3 - R_tank_2**3)
M_comp_2 = V_comp_2 * V_f * rho_carbon + V_comp_2 * (1 - V_f) * rho_resin
M_Ti_2 = V_Ti * rho_Ti
M_tot_2 = M_comp_2 + M_Ti_2
m_He = (P_0*0.0262466*0.004)/(R*(273.15 + 27))

V_des_2 = [0.132, 0.136] # manually calculated volumes
i = 0
R_2 = []
t_2 = []
M_2 = []
for i in range(len(V_des_2)): # looping through oxidiser and fuel tank
   R_tank_prop_2 = ((3/4)*(V_des_2[i]/np.pi))**(1/3) 
   t_Ti_prop_2 = (P_f*R_tank_prop_2)/(sigma_allow_Ti)
   V_Ti_prop_2 = (4/3) * np.pi * ((R_tank_prop_2 + t_Ti_prop_2)**3 - R_tank_prop_2**3)
   M_Ti_prop_2= V_Ti_prop_2 * rho_Ti 
   R_2.append(R_tank_prop_2)
   t_2.append(t_Ti_prop_2)
   M_2.append(M_Ti_prop_2)
   i += 1

# -------------------------------------------------------
# Outputs
# -------------------------------------------------------
print(f"\n---Simulation results---")
print(f"Elliptical Δv: {V_elliptical:.2f} m/s")
print(f"Final circular Δv: {V_circular:.2f} m/s")
print(f"Final spacecraft mass: {mass:.2f} kg")
print(f"Total Δv achieved: {v_done:.2f} m/s (needed {v_req_total:.2f} m/s)")
print(f"Total burn time: {t_tot/3600:.2f} h")
print(f"Total maneuver duration (including orbits): {t_orbit/(86400):.2f} days")
print(f"Number of orbits: {len(a_list)}")

print(f"\n--- Mass breakdown ---")
print(f"Propellant mass kickstage: {m_f_mtm_2:.2f} kg")
print(f"Propellant mass orbiter: {m_fo_2:.2f} kg for Δv: {(v_req_total+v_disturbance):.2f} m/s")
print(f"Orbiter mass: {mo:.2f} kg")
print(f"Kickstage mass: {m_mtm:.2f} kg")
print(f"Total launch mass: {m_total:.2f} kg")

print(f"\n--- Velocities ---")
print(f"Elliptical speed: {V_elliptical:.2f} m/s")
print(f"Circular speed: {V_circular:.2f} m/s")

print(f"\n--- Characteristics tanks ---")
print(f"\n---V1---")
print(f"Tank volume: {V_o:.4f} m^3")
print(f"Radius prop tank: {R_tank_prop:.4f} m")
print(f"Tank wall thickness: {t_Ti_prop*10**3:.3f} mm")
print(f"Propellant tank mass: {M_Ti_prop:.3f} kg")

print(f"Pressurization gas tank volume: {V_s:.4f} m^3")
print(f"Gas tank radius: {R_tank + t_Ti + t_comp:.4f} m")
print(f"Composite wall thickness: {(t_comp*10**3):.3f} mm")
print(f"Total gas tank wall thickness: {((t_comp+t_Ti)*10**3):.3f} mm")
print(f"Total gas tank mass: {M_tot:.3f} kg")
print(f"Gas mass: {m_N2:.3f} kg")
print(f"Total gas tank mass: {(M_tot+m_N2):.3f} kg")

print(f"\n---V2---")
print(f"Fuel tank volume: {V_des_2[0]:.4f} m^3")
print(f"Oxidiser tank volume: {V_des_2[1]:.4f} m^3")
print(f"Radius fuel tank: {R_2[0]:.4f} m")
print(f"Radius oxidiser tank: {R_2[1]:.4f} m")
print(f"Fuel tank wall thickness: {t_2[0]*10**3:.3f} mm")
print(f"Oxidiser tank wall thickness: {t_2[1]*10**3:.3f} mm")
print(f"Fuel tank mass: {M_2[0]:.3f} kg")
print(f"Oxidiser tank mass: {M_2[1]:.3f} kg")

print(f"Radius gas tank: {(R_tank_2 + t_Ti + t_comp_2):.4f} m")
print(f"Composite wall thickness: {(t_comp_2*10**3):.4f} mm")
print(f"Total gas tank wall thickness: {((t_Ti + t_comp_2)*10**3):.3f} mm")
print(f"Total gas tank structure mass: {M_tot_2:.3f} kg")
print(f"Pressurization gas mass: {m_He:.3f} kg")
print(f"Total gas tank mass: {(M_tot_2 + m_He):.3f} kg")
# -------------------------------------------------------
# Plotting
# -------------------------------------------------------
plot_orbits_periapsis_down(a_list, e_list) # generating figure with the orbits from the maneuver




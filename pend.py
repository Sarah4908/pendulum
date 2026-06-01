import numpy as np
import matplotlib.pyplot as plt


config = {
    'g': 9.81,
    'l': 1.0,
    'b': 0.1,    
    'dt': 0.01   
}

def get_next_state(theta, omega, config):
    """Calculates the state of the pendulum at the next time step."""
    g, l, b, dt = config['g'], config['l'], config['b'], config['dt']
    
    
    alpha = -(g / l) * np.sin(theta) - b * omega + np.random.uniform(-0.01, 0.01)
    
    
    omega_new = omega + alpha * dt
    theta_new = theta + omega_new * dt
    
    return theta_new, omega_new


theta = 170 * np.pi / 180
omega = 0.0
theta_list = []
omega_list = []
t_list = np.arange(0, 100, config['dt'])

for t in t_list:
    theta_list.append(np.degrees(theta))
    omega_list.append(np.degrees(omega))
    theta, omega = get_next_state(theta, omega, config)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))


ax1.plot(t_list, theta_list, color='darkblue', linewidth=1.5)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Angle (degrees)')
ax1.set_title('Motion over Time')
ax1.grid(True, linestyle='--', alpha=0.7)


ax2.plot(theta_list, omega_list, color='red', linewidth=1)
ax2.set_xlabel('Angle (degrees)')
ax2.set_ylabel('Angular Velocity (rad/s)')
ax2.set_title('Phase Portrait')
ax2.grid(True, linestyle='--', alpha=0.7)


plt.tight_layout() 
plt.show()
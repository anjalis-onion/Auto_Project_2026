import matplotlib.pyplot as plt
from complete_control import make_car
from complete_control import update
from complete_control import calculate_desired_acceleration
from complete_control import acceleration_to_throttle_percentage

K_P = 0.6
K_I = 0.1
K_D = 0.6
 
STEPS = 550
 
car = make_car(desired_v=20.0, dt=0.1)

velocities = []
errors = []
times = []

for i in range(STEPS):
    desired_acceleration, error = calculate_desired_acceleration(car, K_P, K_I, K_D)

    throttle = acceleration_to_throttle_percentage(desired_acceleration)
    update(car, throttle)

    velocities.append(car["v"])
    errors.append(error)
    times.append(car["t"])

#velocity graph
plt.figure()

plt.plot(times, velocities)
plt.xlabel("time in sec")
plt.ylabel("velocity in m/s")
plt.title(" Velocity over time")

plt.show()

#error graph

plt.figure()
plt.plot(times, errors)
plt.xlabel("time in sec")
plt.ylabel("error in m/s")
plt.title("Error of velocities over time")

plt.show()
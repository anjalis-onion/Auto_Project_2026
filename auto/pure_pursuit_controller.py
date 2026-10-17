import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# distances in metrs
lookahead = 3.0
wheelbase = 2.5
max_steer = math.radians(30)

track = [(i * 1.0, 4.0 * math.sin(i * 0.15)) for i in range(60)]


def target_point(x, y):
    nearest = min(
        range(len(track)),
        key=lambda i: math.hypot(track[i][0] - x, track[i][1] - y)
    )
   
    for i in range(nearest + 1, len(track)):
        if math.hypot(track[i][0] - x, track[i][1] - y) >= lookahead:
            return track[i]
    return track[-1]


def steering(x, y, yaw):
    tx, ty = target_point(x, y)

    angle = math.atan2(ty - y, tx - x) - yaw
    angle = math.atan2(math.sin(angle), math.cos(angle))  

    distance = math.hypot(tx - x, ty - y)
    turn = math.atan2(2 * wheelbase * math.sin(angle), distance)

    return max(-max_steer, min(max_steer, turn))


x, y, yaw = 0.0, -2.0, 0.0
speed = 5.0
dt = 0.1

xs, ys = [x], [y]

for step in range(200):
    steer = steering(x, y, yaw)

    x += speed * math.cos(yaw) * dt
    y += speed * math.sin(yaw) * dt
    yaw += speed / wheelbase * math.tan(steer) * dt

    xs.append(x)
    ys.append(y)

    if step % 10 == 0:
        print(f"t={step * dt:5.1f}s  x={x:6.2f}  y={y:6.2f}  "
              f"yaw={math.degrees(yaw):7.2f}°  steer={math.degrees(steer):6.2f}°")

    if math.hypot(track[-1][0] - x, track[-1][1] - y) < 1.0:
        print("at the end of track yippee")
        break

fig, ax = plt.subplots()

ax.plot(*zip(*track), "k--", label="Track")
path, = ax.plot([], [], "b-", label="Car path")
car, = ax.plot([], [], "ro", label="Car")

ax.set_xlim(-2, 62)
ax.set_ylim(-7, 7)
ax.set_aspect("equal")
ax.legend()
ax.grid()
ax.set_title("car and track relationship")


def update(frame):
    path.set_data(xs[:frame + 1], ys[:frame + 1])
    car.set_data([xs[frame]], [ys[frame]])
    return path, car


anim = FuncAnimation(fig, update, frames=len(xs), interval=50, blit=True, repeat=False)
plt.show()
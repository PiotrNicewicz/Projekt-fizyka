import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

from PIL import Image
import io

make_gif = True

frames = []
# ==========================
# PARAMETRY MODELU
# ==========================

N_AGENTS = 50

MASS = 80.0

R = 50
A = 3000
B = 20

A_WALL = 2000
B_WALL = 50

TAU = 0.3

DT = 0.01
STEPS = 500

WIDTH = 1100
HEIGHT = 600

AGENT_RADIUS = 15
# ==========================
# AGENT
# ==========================

class Agent:
    def __init__(self):

        self.pos = np.array([
            np.random.uniform(50, WIDTH/2 - 120),
            np.random.uniform(50, HEIGHT - 50)
        ])

        self.vel = np.zeros(2)

        self.goal = np.array([
            WIDTH-100,
            HEIGHT/2
        ])

        self.desired_speed = max(
            30,
            np.random.normal(120, 40)
        )

        self.trajectory = [self.pos.copy()]


# ==========================
# ŚCIANY
# ==========================

DOOR_WIDTH = 45
middle_x = WIDTH / 2
middle_y = HEIGHT / 2

door_bottom = HEIGHT/2 - DOOR_WIDTH/2
door_top = HEIGHT/2 + DOOR_WIDTH/2

walls = [

    # zewnętrzne
    [0, 0, 20, HEIGHT],
    [WIDTH - 20, 0, WIDTH, HEIGHT],
    [0, 0, WIDTH, 20],
    [0, HEIGHT - 20, WIDTH, HEIGHT],

    # ściana środkowa - część górna
    [
        middle_x - 20,
        20,
        middle_x + 20,
        door_bottom
    ],

    # ściana środkowa - część dolna
    [
        middle_x - 20,
        door_top,
        middle_x + 20,
        HEIGHT - 20
    ]
]
def closest_point_rect(rect, point):

    xmin, ymin, xmax, ymax = rect

    x = np.clip(point[0], xmin, xmax)
    y = np.clip(point[1], ymin, ymax)

    return np.array([x, y])

def inside_rect(rect, point):
    xmin, ymin, xmax, ymax = rect

    return (
        xmin <= point[0] <= xmax
        and
        ymin <= point[1] <= ymax
    )

# ==========================
# INICJALIZACJA
# ==========================

agents = [Agent() for _ in range(N_AGENTS)]
goals = {
    "door_top": [middle_x - 30, door_top],
    "door_bottom": [middle_x - 30, door_bottom],
    "door_middle": [middle_x + 80, HEIGHT/2],
    "right_side": [WIDTH-100, HEIGHT/2]
}

# ==========================
# SYMULACJA
# ==========================
plt.ion()

fig, ax = plt.subplots(figsize=(12,7))


for step in range(STEPS):


    forces = []

    # ----------------------
    # Liczenie sił
    # ----------------------

    for agent in agents:
        if agent.pos[0] < middle_x - 50:
            if agent.pos[1] > door_top:
                agent.goal = goals["door_top"]
            elif agent.pos[1] < door_bottom:
                agent.goal = goals["door_bottom"]
            else:
                agent.goal = goals["door_middle"]
        else:
            agent.goal = goals["right_side"]

        goal_vec = agent.goal - agent.pos
        dist_goal = np.linalg.norm(goal_vec)

        if dist_goal > 1e-8:
            goal_dir = goal_vec / dist_goal
        else:
            goal_dir = np.zeros(2)

        f_goal = MASS * (
            agent.desired_speed * goal_dir
            - agent.vel
        ) / TAU

        # ===== f_rep =====

        f_rep = np.zeros(2)

        for other in agents:

            if other is agent:
                continue

            dvec = agent.pos - other.pos
            d = np.linalg.norm(dvec)

            if d < 1e-8:
                continue

            n = dvec / d
            effective_d = d - AGENT_RADIUS
            f_rep += A * np.exp((R - effective_d) / B) * n

        # ===== f_wall =====

        f_wall = np.zeros(2)

        for wall in walls:

            closest = closest_point_rect(
                wall,
                agent.pos
            )

            dvec = agent.pos - closest
            d = np.linalg.norm(dvec)

            if d < 1e-8:
                continue

            n = dvec / d

            f_wall += (
                A_WALL
                * np.exp((R - d) / B_WALL)
                * n
            )

        total_force = f_goal + f_rep + f_wall

        forces.append(total_force)

    # ----------------------
    # Aktualizacja ruchu
    # ----------------------

    for agent, force in zip(agents, forces):

        acc = force / MASS

        agent.vel += acc * DT
        agent.pos += agent.vel * DT
        for wall in walls:

            xmin, ymin, xmax, ymax = wall

            expanded_wall = [
                xmin - AGENT_RADIUS,
                ymin - AGENT_RADIUS,
                xmax + AGENT_RADIUS,
                ymax + AGENT_RADIUS
            ]

            if inside_rect(expanded_wall, agent.pos):

                distances = [
                    abs(agent.pos[0] - (xmin - AGENT_RADIUS)),
                    abs(agent.pos[0] - (xmax + AGENT_RADIUS)),
                    abs(agent.pos[1] - (ymin - AGENT_RADIUS)),
                    abs(agent.pos[1] - (ymax + AGENT_RADIUS))
                ]

                side = np.argmin(distances)

                if side == 0:
                    agent.pos[0] = xmin - AGENT_RADIUS
                    agent.vel[0] = 0

                elif side == 1:
                    agent.pos[0] = xmax + AGENT_RADIUS
                    agent.vel[0] = 0

                elif side == 2:
                    agent.pos[1] = ymin - AGENT_RADIUS
                    agent.vel[1] = 0

                else:
                    agent.pos[1] = ymax + AGENT_RADIUS
                    agent.vel[1] = 0
        # nowy cel
        # if np.linalg.norm(agent.pos - agent.goal) < 50:

        #     agent.goal = np.array([
        #         np.random.uniform(50, WIDTH - 50),
        #         np.random.uniform(50, HEIGHT - 50)
        #     ])

        agent.trajectory.append(
            agent.pos.copy()
        )

    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):

            a = agents[i]
            b = agents[j]

            dvec = a.pos - b.pos
            d = np.linalg.norm(dvec)

            min_dist = 2 * AGENT_RADIUS

            if d < min_dist and d > 1e-8:

                n = dvec / d

                overlap = min_dist - d

                a.pos += n * overlap / 2
                b.pos -= n * overlap / 2

    ax.clear()

    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    for wall in walls:
        xmin, ymin, xmax, ymax = wall

        ax.add_patch(
            Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin
            )
        )

    for agent in agents:
        circle = Circle(
            (agent.pos[0], agent.pos[1]),
            AGENT_RADIUS,
            fill=True,
            alpha=0.9, color="r"
        )
        ax.add_patch(circle)

    ax.set_title(f"Step {step}")
    if make_gif:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        frames.append(Image.open(buf).copy())
    plt.pause(0.001)

plt.ioff()
plt.show()
if make_gif:
    frames[0].save("social.gif",save_all=True,append_images=frames[1:],duration=20,loop=0)
# ==========================
# WYKRES
# ==========================

plt.figure(figsize=(12, 7))

for agent in agents:

    traj = np.array(agent.trajectory)

    plt.plot(
        traj[:, 0],
        traj[:, 1],
        linewidth=1
    )

    plt.scatter(
        traj[0, 0],
        traj[0, 1],
        s=20
    )

plt.xlim(0, WIDTH)
plt.ylim(0, HEIGHT)

plt.xlabel("x")
plt.ylabel("y")

plt.title(
    f"Social Force Model ({N_AGENTS} agentów)"
)

plt.grid(True)

plt.show()
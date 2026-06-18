import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
# ==========================
# PARAMETRY MODELU
# ==========================

N_AGENTS = 100

MASS = 80.0

R = 30
A = 3000
B = 20

A_WALL = 3000
B_WALL = 20

TAU = 0.3

DT = 0.02
MAX_STEPS = 4000
REPEATS = 10

WIDTH = 1100
HEIGHT = 600

AGENT_RADIUS = 15

agent_counts = [10,20,30,40,50,100]
evacuation_times = []

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

DOOR_WIDTH = 60

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
def evacuated(agent):
    return agent.pos[0] > WIDTH - 120
# ==========================
# INICJALIZACJA
# ==========================


goals = {
    "door_top": [middle_x - 30, door_top],
    "door_bottom": [middle_x - 30, door_bottom],
    "door_middle": [middle_x + 80, HEIGHT/2],
    "right_side": [WIDTH-100, HEIGHT/2]
}

# ==========================
# SYMULACJA
# ==========================


for N_AGENTS in agent_counts:
    times = []
    for _ in range(REPEATS):
        agents = [Agent() for _ in range(N_AGENTS)]
        evac_time = None
        for step in range(MAX_STEPS):
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

                    f_rep += A * np.exp((R - d) / B) * n

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
                speed = np.linalg.norm(agent.vel)
                MAX_SPEED = 200
                if speed > MAX_SPEED:
                    agent.vel *= MAX_SPEED / speed

                agent.pos += agent.vel * DT
                for wall in walls:

                    if inside_rect(wall, agent.pos):

                        xmin, ymin, xmax, ymax = wall

                        distances = [
                            abs(agent.pos[0] - xmin),
                            abs(agent.pos[0] - xmax),
                            abs(agent.pos[1] - ymin),
                            abs(agent.pos[1] - ymax)
                        ]

                        side = np.argmin(distances)

                        if side == 0:
                            agent.pos[0] = xmin - 1
                            agent.vel[0] = 0

                        elif side == 1:
                            agent.pos[0] = xmax + 1
                            agent.vel[0] = 0

                        elif side == 2:
                            agent.pos[1] = ymin - 1
                            agent.vel[1] = 0

                        else:
                            agent.pos[1] = ymax + 1
                            agent.vel[1] = 0

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
            agents = [
                agent for agent in agents
                if not evacuated(agent)
            ]
            if len(agents) == 0:
                evac_time = step * DT
                break
        if evac_time is None:
            evac_time = MAX_STEPS * DT
        times.append(evac_time)
    times_success =  [t for t in times if t < MAX_STEPS*DT]
    print(
        f"N={N_AGENTS}, "
        f"mean={np.mean(times_success):.2f}, "
        f"min={np.min(times_success):.2f}, "
        f"max={np.max(times_success):.2f}"
    )
    evacuation_times.append(np.mean(times_success))

plt.figure(figsize=(10,6))

plt.plot(
    agent_counts,
    evacuation_times,
    marker='o'
)

plt.xlabel("Liczba agentów")
plt.ylabel("Czas ewakuacji [s]")
plt.title("Czas ewakuacji w funkcji liczby agentów")
plt.grid(True)

plt.show()
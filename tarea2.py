
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

try:
    env = gym.make("Taxi-v3")
    print("Entorno 'Taxi-v3' cargado")
    print(f"Espacio de Observacion: {env.observation_space.n} estados")
    print(f"Espacio de Acciones: {env.action_space.n} acciones")
except Exception as e:
    print(f"Error al cargar el entorno: {e}")


#entrenamiento agente aleaatorio
random_episodes = 2000
random_rewards = []

for episode in tqdm(range(random_episodes), desc="Entrenando Agente Aleatorio"):
    state = env.reset()[0]
    done = False
    total_reward = 0

    while not done:

        action = env.action_space.sample()


        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated


        total_reward += reward


    random_rewards.append(total_reward)

print("Entrenamiento del agente aleatorio completado")



def evaluate_agent(env, policy, episodes=100):

    total_successes = 0
    total_rewards = 0
    total_steps = 0

    for _ in tqdm(range(episodes), desc="Evaluando Agente"):
        state = env.reset()[0]
        done = False
        episode_steps = 0

        while not done:
            action = policy(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if done and reward == 20:
                total_successes += 1

            episode_steps += 1

        total_rewards += env.rewards if hasattr(env, 'rewards') else -episode_steps
        total_steps += episode_steps

    success_rate = (total_successes / episodes) * 100
    avg_reward = total_rewards / episodes
    avg_episode_length = total_steps / episodes

    return success_rate, avg_reward, avg_episode_length
random_policy = lambda state: env.action_space.sample()

random_success, random_avg_reward, random_avg_length = evaluate_agent(env, random_policy)

print("\nResultados Agente Aleatorio)")
print(f"Porcentaje de exito: {random_success:.2f}%")
print(f"Recompensa media: {random_avg_reward:.2f}")
print(f"Longitud promedio del episodio: {random_avg_length:.2f} pasos")


#entrenamiento agente qlearning

q_table = np.zeros([env.observation_space.n, env.action_space.n])

alpha = 1.0
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01


q_learning_episodes = 2000
q_learning_rewards = []


for episode in tqdm(range(q_learning_episodes), desc="Entrenando Agente Q-Learning"):
    state = env.reset()[0]
    done = False
    total_reward = 0

    while not done:

        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = old_value + alpha * (reward + gamma * next_max - old_value)
        q_table[state, action] = new_value

        state = next_state
        total_reward += reward

    q_learning_rewards.append(total_reward)
    if epsilon > min_epsilon:
        epsilon *= epsilon_decay

print("Entrenamiento con Q-Learning completado")


q_learning_policy = lambda state: np.argmax(q_table[state])

q_success, q_avg_reward, q_avg_length = evaluate_agent(env, q_learning_policy)

print("\nResultados Agente Q-Learning")
print(f"Porcentaje de exito: {q_success:.2f}%")
print(f"Recompensa media: {q_avg_reward:.2f}")
print(f"Longitud promedio del episodio: {q_avg_length:.2f} pasos")


print("\nGenerando Graficas")

def moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

plt.figure(figsize=(14, 7))

plt.plot(
    np.arange(len(moving_average(random_rewards))),
    moving_average(random_rewards),
    label='Agente Aleatorio',
    color='red',
    alpha=0.7
)

plt.plot(
    np.arange(len(moving_average(q_learning_rewards))),
    moving_average(q_learning_rewards),
    label='Agente Q-Learning',
    color='green'
)

plt.title('Recompensa Promedio por Episodio Durante el Entrenamiento', fontsize=16)
plt.xlabel('Episodios', fontsize=12)
plt.ylabel('Recompensa Acumulada', fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)
plt.show()


print("\n--- Tabla Comparativa de Rendimiento ---")
print("----------------------------------------------------------------------")
print(f"| Metrica                     | Agente Aleatorio | Agente Q-Learning |")
print("----------------------------------------------------------------------")
print(f"| Tasa de exito (%)           | {random_success:<16.2f} | {q_success:<17.2f} |")
print(f"| Recompensa Media            | {random_avg_reward:<16.2f} | {q_avg_reward:<17.2f} |")
print(f"| Longitud Promedio (pasos)   | {random_avg_length:<16.2f} | {q_avg_length:<17.2f} |")
print("----------------------------------------------------------------------")

env.close()
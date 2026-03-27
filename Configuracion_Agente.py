import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import random
from IPython.display import clear_output
import optuna
import matplotlib.cm as cm

class motorEnv(gym.Env):
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, archivo):
        super(motorEnv, self).__init__()
        data = np.loadtxt(archivo, delimiter=',', skiprows=1, dtype=np.float32)
        self.data = data[:, 1:]
        self.num_states = self.data.shape[0]
        
        self.col = np.zeros(self.num_states, dtype=np.uint8)
        self.col_t = np.zeros(self.num_states, dtype=np.uint8)
        self.new_state = None
        self.old_state = None
        
        self.action_space = spaces.Discrete(4) 
        
        obs_low  = np.min(self.data, axis=0) 
        obs_high = np.max(self.data, axis=0)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.q_table = np.zeros([self.num_states, 4], dtype=np.float32)
        
        self.min_w = np.min(self.data[:, 2])
 
        self.step_var1 = np.min(np.diff(np.unique(self.data[:, 0])))
        self.step_var2 = np.min(np.diff(np.unique(self.data[:, 1])))
        
        self.state_dict = {(np.round(row[0], 4), np.round(row[1], 4)): idx for idx, row in enumerate(self.data)}

    def step(self, a):
        reward = -0.1
        var1, var2, w = self.data[self.old_state]
        
        # Usamos el step dinámico calculado en el __init__
        if a == 0:
            var1_prima = var1 + self.step_var1
            var2_prima = var2 
        elif a == 1:
            var1_prima = var1 - self.step_var1
            var2_prima = var2
        elif a == 2:
            var1_prima = var1
            var2_prima = var2 + self.step_var2
        else:
            var1_prima = var1
            var2_prima = var2 - self.step_var2
            
        key = (np.round(var1_prima, 4), np.round(var2_prima, 4))
        
        if key in self.state_dict:
            self.new_state = self.state_dict[key]
            reward += (self.data[self.old_state, 2] - self.data[self.new_state, 2]) * 100
        else:
            self.new_state = self.old_state
            reward = -1.0 
            
        if self.data[self.new_state, 2] == self.min_w:
            reward += 10000.0 
            
        self.col[self.new_state] = 1 
        self.col_t[self.new_state] = 1
        self.old_state = self.new_state
        
        return self.new_state, reward

    def reset(self):
        self.old_state = random.randint(0, self.num_states - 1) 
        self.col.fill(0)
        self.col[self.old_state] = 1
        self.col_t[self.old_state] = 1
        return self.old_state

    def epsilon_decay_fn(self, min_epsilon, epsilon, episodes):
        return (min_epsilon / epsilon) ** (1 / episodes)

    def update_parameter(self, value, decay_rate, min_value):
        return max(min_value, value * decay_rate)

    def train(self, alpha, alpha_decay, min_alpha, gamma, epsilon, min_epsilon, episodes):
        self.q_table = np.zeros([self.num_states, 4], dtype=np.float32)
        reward_eps = []
        steps_eps = []
        historial_potencia_online = [] 
        
        mejor_estado_global = self.data[0, :2]
        potencia_min_global = float('inf')
        
        # Tolerancia uniforme de 2000 pasos: le da tiempo suficiente para escapar de mínimos locales
        tolerancia_pasos = 2000
            
        for i in range(1, episodes + 1):
            state = self.reset()
            contador_suma = 0
            reward_ep = 0
            
            steps_sin_mejorar = 0
            potencia_episodio = float('inf')
            
            while contador_suma < 50000:
                if random.uniform(0, 1) < epsilon:
                    action = self.action_space.sample() 
                else:
                    action = np.argmax(self.q_table[state])
                
                next_state, reward = self.step(action)
                
                old_value = self.q_table[state, action]
                next_max = np.max(self.q_table[next_state])
                self.q_table[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                reward_ep += reward
                
                actual_w = self.data[next_state, 2]
                
                if actual_w >= potencia_episodio:
                    steps_sin_mejorar += 1
                else:
                    steps_sin_mejorar = 0
                    potencia_episodio = actual_w
                    
                    if potencia_episodio < potencia_min_global:
                        potencia_min_global = potencia_episodio
                        mejor_estado_global = self.data[next_state, :2]
                
                if steps_sin_mejorar > tolerancia_pasos:
                    break
                
                if actual_w == self.min_w:
                    break
                
                state = next_state
                contador_suma += 1
                
            epsilon *= self.epsilon_decay_fn(min_epsilon, epsilon, episodes)
            alpha = self.update_parameter(alpha, alpha_decay, min_alpha)
            
            historial_potencia_online.append(potencia_min_global)
            
            if i % 10000 == 0:
                clear_output(wait=True)
                print(f"Episode: {i}/{episodes}")
                print(f"Mejor potencia online: {potencia_min_global:.4f} en {mejor_estado_global}")
                
            steps_eps.append(contador_suma)
            reward_eps.append(reward_ep)
            
        arr1 = np.array(steps_eps)
        arr2 = np.array(reward_eps)
        resultado = np.divide(arr2, arr1, out=np.zeros_like(arr2, dtype=float), where=arr1!=0)
        
        print("Training finished.")
        return resultado, steps_eps, reward_eps, mejor_estado_global, potencia_min_global, historial_potencia_online

    def test(self, random_start=False):
        if random_start:
            state = self.reset()
        else:
            self.old_state = 0
            state = 0
            
        contador_suma = 0
        steps_sin_mejorar = 0
        potencia_test = float('inf')
        mejor_estado_test = self.data[state, :2]
        
        camino_x = [self.data[state, 0]]
        camino_y = [self.data[state, 1]]
        camino_z = [self.data[state, 2]]
        
        while contador_suma < 100000:
            action = np.argmax(self.q_table[state]) 
            next_state, _ = self.step(action)
            state = next_state
            contador_suma += 1
            
            actual_w = self.data[next_state, 2]
            camino_x.append(self.data[next_state, 0])
            camino_y.append(self.data[next_state, 1])
            camino_z.append(actual_w)
            
            if actual_w >= potencia_test:
                steps_sin_mejorar += 1
            else:
                steps_sin_mejorar = 0
                potencia_test = actual_w
                mejor_estado_test = self.data[next_state, :2]
                
            # Tolerancia alta en test para que pueda esquivar barreras si sabe cómo
            if steps_sin_mejorar > 5000 or actual_w == self.min_w:
                break
                
        return contador_suma, self.q_table, mejor_estado_test, potencia_test, (camino_x, camino_y, camino_z)

    def optuna_opt(self, epsilon, min_epsilon, episodes, trials):
        def objective(trial):
            alpha = trial.suggest_float('alpha', 0.1, 1.0, log=True)
            min_alpha = trial.suggest_float('min_alpha', 0.0001, alpha, log=True)
            alpha_decay = trial.suggest_float('alpha_decay', 0.9997, 1.0)
            
            # Gamma alto (entre 0.95 y 0.999) para que le dé importancia a la meta final (+10000)
            gamma = trial.suggest_float('gamma', 0.95, 0.999, log=True)
            
            self.train(alpha, alpha_decay, min_alpha, gamma, epsilon, min_epsilon, episodes)
            
            potencia_media = 0
            for _ in range(3):
                _, _, _, potencia, _ = self.test(random_start=True)
                potencia_media += potencia
                
            return potencia_media / 3.0 

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=trials)
        return study.best_trial.params

    def get_policy_map(self):
        U = np.zeros(self.num_states)
        V = np.zeros(self.num_states)
        for state in range(self.num_states):
            action = np.argmax(self.q_table[state])
            if action == 0: U[state], V[state] = 1, 0
            elif action == 1: U[state], V[state] = -1, 0
            elif action == 2: U[state], V[state] = 0, 1
            elif action == 3: U[state], V[state] = 0, -1
        return U, V
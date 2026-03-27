import numpy as np
import json
import os
import gymnasium
from Configuracion_Agente import motorEnv

def simular_malla(csv_name, qtable_name, output_json):
    print(f">> Generando trayectoria única para: {output_json}")

    ruta_base = os.path.dirname(os.path.abspath(__file__))
    path_csv = os.path.join(ruta_base, csv_name)
    path_qtable = os.path.join(ruta_base, qtable_name)

    env = motorEnv(path_csv)
    q_table = np.load(path_qtable)

    estado = np.random.randint(len(env.data))
    env.old_state = estado

    v1_ini, v2_ini, w_ini = env.data[estado]

    tray_obj = {
        "inicio": {"v1": float(v1_ini), "v2": float(v2_ini), "w": float(w_ini)},
        "fin": {},
        "pasos": 0,
        "exito": False,
        "min_w_tray": float(w_ini), # Inicializamos con el primer valor de w
        "trayectoria": []
    }

    pasos = 0
    epsilon = 0.15
    max_pasos = 500
    done = False

    while pasos < max_pasos:
        v1, v2, w = env.data[estado]
        
        # ACTUALIZACIÓN DEL MÍNIMO: 
        # Si el w actual es menor que el registrado, lo guardamos
        if w < tray_obj["min_w_tray"]:
            tray_obj["min_w_tray"] = float(w)

        tray_obj["trayectoria"].append({
            "paso": pasos,
            "v1": float(v1),
            "v2": float(v2),
            "w": float(w)
        })

        if np.random.rand() < epsilon:
            accion = np.random.randint(q_table.shape[1])
        else:
            accion = int(np.argmax(q_table[estado]))

        nuevo_estado, _ = env.step(accion)

        if nuevo_estado == estado and np.random.rand() < 0.3:
            nuevo_estado = np.random.randint(len(env.data))
            env.old_state = nuevo_estado

        if env.data[nuevo_estado, 2] == env.min_w:
            tray_obj["exito"] = True
            if pasos > 50:
                done = True

        estado = nuevo_estado
        pasos += 1
        if done: break

    v1_fin, v2_fin, w_fin = env.data[estado]
    tray_obj["fin"] = {"paso": pasos, "v1": float(v1_fin), "v2": float(v2_fin), "w": float(w_fin)}
    tray_obj["pasos"] = pasos

    path_out = os.path.join(ruta_base, output_json)
    with open(path_out, 'w') as f:
        json.dump([tray_obj], f, indent=4)

    print(f"Listo: {output_json} (Min W: {tray_obj['min_w_tray']})")

if __name__ == "__main__":
    simular_malla("./Datos/Originales/02_Reinforcement_learning/Datos_v1.csv", "./Datos/Transformados/RL/q_table_m1_150k.npy", "./Datos/Transformados/Big_Data/sim_m1_nodered.json")
    simular_malla("./Datos/Originales/02_Reinforcement_learning/Datos_v2.csv", "./Datos/Transformados/RL/q_table_m2_150k.npy", "./Datos/Transformados/Big_Data/sim_m2_nodered.json")
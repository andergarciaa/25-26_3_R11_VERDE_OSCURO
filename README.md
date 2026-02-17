# Reto 11 - NextGen E-Drives: Optimización y Mantenimiento Predictivo

### Industrial AI, Smart Manufacturing & Fault Diagnosis

  **Equipo:** Verde Oscuro
**Project Managers:** Beatriz Chicote, Ander Juarez, Carlos Cernuda, Aitor Duo

---

## 📝 Descripción del Proyecto
Este proyecto aborda la transformación digital de procesos industriales bajo el paradigma de **Industria 4.0**. El objetivo es actuar como una consultora analítica para optimizar el ciclo de vida completo de un accionamiento eléctrico de imanes permanentes enterrados, abarcando desde el diseño geométrico asistido por IA hasta la detección proactiva de fallos en rodamientos.

El desarrollo se divide en cuatro pilares estratégicos:
1.  **Optimización Multiobjetivo (MOO):** Diseño de motores eficientes y de bajo coste mediante algoritmos evolutivos.
2.  **Simulación de Fabricación:** Modelado de la cadena de montaje con Sympy para optimizar la asignación de recursos.
3.  **Control Inteligente (RL):** Minimización del consumo energético mediante Aprendizaje por Refuerzo en banco de pruebas.
4.  **Diagnóstico de Fallos (DSP):** Identificación de anomalías mecánicas mediante el procesamiento de señales de alta frecuencia (51.2 kHz).

---

## ⚙️ Lógica de Ingeniería y Objetivos

El núcleo técnico consiste en optimizar un motor eléctrico definido por un vector de diseño de 6 parámetros geométricos: $x=(h_{m},\alpha_{m},e_{r},d_{si},b_{st},b_{ss})$.



### Objetivos de Minimización (MOO):
* **Eficiencia ($o1$):** Minimizar la eficiencia negada para maximizar el rendimiento operativo del motor.
* **Par de Cogging ($o2$):** Reducir las vibraciones y el ruido acústico en condiciones de funcionamiento sin carga.
* **Coste Material ($o3$):** Minimizar el coste global de materiales (cobre, aluminio, imanes).
* **Ripple de Par ($o4$):** Reducir las fluctuaciones de par bajo carga para garantizar la estabilidad del sistema.

### Control por Refuerzo (RL):
Se implementa un agente para descubrir las condiciones operativas que minimizan la potencia consumida:
* **Reward:** +1000 al alcanzar el objetivo, -100 por estados inexistentes y -1 por cada iteración sin éxito.
* **Algoritmo:** Implementación de Q-Learning para encontrar la política óptima de consumo.

---

## 📂 Estructura del Repositorio

El flujo de ejecución está diseñado secuencialmente, garantizando la trazabilidad desde la optimización inicial hasta la validación de fallos.

| Orden | Script / Notebook | Descripción Técnica |
| :--- | :--- | :--- |
| 1 | `01_Optimizacion_MOO.ipynb` | Implementación de **MOEAs** para proponer diseños óptimos con el mínimo de observaciones. |
| 2 | `02_Simulacion_Sympy.ipynb` | Simulación de la línea de montaje y fabricación de componentes (Rotor/Estator). |
| 3 | `03_Control_RL.ipynb` | Entrenamiento del agente de **Aprendizaje por Refuerzo** para estabilizar la potencia activa. |
| 4 | `04_Deteccion_Anomalias.ipynb` | Procesamiento de señal (FFT) para identificar fallos en pistas y bolas del rodamiento. |
| 5 | `05_Ingesta_NodeRED.json` | Flujo de ingesta y transformación de datos para el ecosistema digital. |
| 6 | `app_visualizacion.py` | Dashboard interactivo que integra el frente de Pareto y el análisis de vibraciones. |

---

## 🛠️ Requisitos e Instalación

El sistema analiza señales de 4 acelerómetros (IMI 601A01 y 604B31), un micrófono SM81 y un tacómetro MT-190.



```bash
# Clonar el repositorio
git clone [https://github.com/tu-usuario/R11_Entrega_EQUIPO.git](https://github.com/tu-usuario/R11_Entrega_EQUIPO.git)

# Instalación de dependencias
pip install -r requirements.txt

# Challenge 11: Design Optimization and Development of Electrical Drives

## Project Description
This repository contains the code developed for Challenge 11 of the 3rd year of the Business Data Analytics Degree at Mondragon Unibertsitatea. The main objective of this project is to optimize, test, and validate motor designs based on 6 geometric parameters. 

To achieve this, we have approached the problem through different disciplines and phases:
* **Design Optimization (MOO):** Proposing the best motor designs using multi-objective evolutionary algorithms.
* **Manufacturing Simulation:** Modeling the assembly line process using Sympy.
* **Testing (RL):** Stabilizing the active power of the motor at its minimum using reinforcement learning.
* **Validation (DSP):** Identifying potential anomalies in bearings using signal analysis techniques.

## Repository Structure
The code is divided into modular scripts that represent the different functional units of the project:

### 1. Design Optimization (MOEA)
* `1.1_Optimizacion_EDA.ipynb`: Initial Exploratory Data Analysis on the design parameters.
* `1.2_Optimizacion_surrogates.ipynb`: Approximation of new simulations and performance metrics.
* `1.3_Optimizacion_MOEA.ipynb`: Implementation, evaluation, and evolution of the evolutionary strategies.
* `1.4_Optimizacion_Top5.ipynb`: Selection of the top 5 motor designs.

### 2. Testing (Reinforcement Learning)
* `2.1_Mallas_RL.ipynb`: Implementation of the RL environment to minimize the drive's energy consumption.
* `Configuracion_Agente.py`: Script containing the agent's configuration, hyperparameters, and policies (Q-Learning).

### 3. Manufacturing Simulation
* `3_Data_science_Simpy.ipynb`: Simulation of the assembly process for the electrical drives production line.

### 4. Validation (Signal Processing)
* `4.1_1Preprocesamiento_de_Datos_Industria.ipynb`: Feature extraction and signal processing from the installed sensors (accelerometers, microphone, tachometer).
* `4.2_Modelaje_industria.ipynb`: Implementation of predictive models for identifying bearing anomalies (unbalance, misalignment, ball/race defects).

### 5. Big Data and Data Generation
* `5.1_Big_data_generar_simulacion.py`: Script for data generation and ingestion, aimed at feeding the Node-RED system and the final Dashboard.

## Installation and Requirements
To install the necessary dependencies to run the notebooks and scripts in this project, run the following command:
```bash
pip install -r requirements.txt

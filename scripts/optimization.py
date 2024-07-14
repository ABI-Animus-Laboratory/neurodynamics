import numpy as np
import nest
import itertools
import os
import datetime
import math
import copy
import time
import pickle
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

import quantities as pq
from neo.core import SpikeTrain
from elephant.spike_train_dissimilarity import victor_purpura_distance
from scripts import model
from simanneal import Annealer

nest.set_verbosity('M_FATAL')

class SimulatedAnnealing1(Annealer):
    def __init__(self, g_e, g_i, state, place_obs, spikes_obs, int_obs, int_spikes_obs, lamb, categorized_neurons, runtime = 17998, move_params = None, cost_type = '', resolution=0.1):
        self.categorized_neurons = categorized_neurons
        self.place_obs = place_obs

        self.spikes_obs = spikes_obs
        self.int_obs = int_obs
        self.int_spikes_obs = int_spikes_obs
        self.lamb = lamb
        self.state = state
        self.cost_type = cost_type
        self.objs = []
        self.best_objs = []
        self.cost_type = cost_type
        self.runtime = runtime
        self.place_preds = [] 
        self.spikes_preds = []
        self.int_preds = [] 
        self.int_spikes_preds = []
        self.resolution = resolution

        self.g_e = g_e
        self.g_i = g_i

        if move_params is None:
            #Default move params
            self.move_params = {
                'num_weights': 3000,
                'weights_range': (-5, 5),
                'weights_change_range': (-0.2, 0.2),
            }
        else:
            self.move_params = move_params

        self.voltage_traces_place = None
        self.voltage_traces_int = None
        self.spikes_place = None
        self.spikes_int = None

    def energy(self):
        network = model.Model1(self.categorized_neurons, self.state, G_e = self.g_e, G_i = self.g_i, runtime = self.runtime, resolution=self.resolution)
        network.simulate()
        place_pred = network.get_voltage_traces('Place')
        spikes_pred = network.get_spike_trains('Place')
        self.external_connectivity_indices = network.external_connectivity_indices


        int_pred = network.get_voltage_traces('Inter')
        int_spikes_pred = network.get_spike_trains('Inter')

        place_pred = self.align_simulation_and_biological(self.place_obs, place_pred)
        spikes_pred = self.align_simulation_and_biological(self.spikes_obs, spikes_pred)

        int_pred = self.align_simulation_and_biological(self.int_obs, int_pred)
        int_spikes_pred = self.align_simulation_and_biological(self.int_spikes_obs, int_spikes_pred)

        match self.cost_type:
            case 'standard':
                cost_function = self.ssd_with_l1(place_pred, int_pred)
            case 'quadratic':
                cost_funcion = self.ssd_with_l1_with_quad(place_pred, int_pred)
            case 'boundary':
                cost_function = self.ssd_with_l1_emphasise_boundaries(place_pred, int_pred)
            case _:
                print('Invalid cost function!')

        cost_function = self.ssd_with_l1(place_pred, int_pred)
        self.objs.append(cost_function)
        self.place_preds.append(place_pred)
        self.int_preds.append(int_pred)
        self.spikes_preds.append(spikes_pred)
        self.int_spikes_preds.append(int_spikes_pred)

        if self.best_objs == []:
            self.best_objs.append(cost_function)
        else:
            if cost_function < self.best_objs[-1]:
                self.best_objs.append(cost_function)
            else:
                self.best_objs.append(self.best_objs[-1])

        self.voltage_traces_place = place_pred
        self.voltage_traces_int = int_pred
        self.spikes_place = spikes_pred
        self.spikes_int = int_spikes_pred
        
        return cost_function
    
    def align_simulation_and_biological(self, obs, pred):
        return pred[:, ::33][:, :np.size(obs, axis=1)]

    def anneal(self):
        """Minimizes the energy of a system by simulated annealing.

        Parameters
        state : an initial arrangement of the system

        Returns
        (state, energy): the best state and energy found.
        """
        step = 0
        self.start = time.time()

        #modifications
        objs = []
        best_objs = []

        # Precompute factor for exponential cooling from Tmax to Tmin
        if self.Tmin <= 0.0:
            raise Exception('Exponential cooling requires a minimum "\
                "temperature greater than zero.')
        Tfactor = -math.log(self.Tmax / self.Tmin)

        # Note initial state
        T = self.Tmax
        E = self.energy()

        objs.append(E)
        best_objs.append(E)

        prevState = self.copy_state(self.state)
        prevEnergy = E

        self.best_state = self.copy_state(self.state)
        self.best_energy = E

        self.best_state = self.copy_state(self.state)
        self.best_energy = E
        trials, accepts, improves = 0, 0, 0
        if self.updates > 0:
            updateWavelength = self.steps / self.updates
            self.update(step, T, E, None, None)

        # Attempt moves to new states
        while step < self.steps and not self.user_exit and (step < 1000 or best_objs[-1] != best_objs[-1000]):
            #modification
            # objs.append(E)
            # best_objs.append(E)

            step += 1
            T = self.Tmax * math.exp(Tfactor * step / self.steps)
            dE = self.move()
            if dE is None:
                E = self.energy()
                dE = E - prevEnergy
            else:
                E += dE
            trials += 1
            if dE > 0.0 and math.exp(-dE / T) < random.random():
                # Restore previous state
                self.state = self.copy_state(prevState)
                E = prevEnergy

                #Modification (change most recently added place pred to previous one)
                self.place_preds[-1] = self.place_preds[-2]
                self.int_preds[-1] = self.int_preds[-2]
            else:
                # Accept new state and compare to best state
                accepts += 1
                if dE < 0.0:
                    improves += 1
                prevState = self.copy_state(self.state)
                prevEnergy = E
                if E < self.best_energy:
                    self.best_state = self.copy_state(self.state)
                    self.best_energy = E
            #modification
            best_objs.append(self.best_energy)
            objs.append(E)

            if self.updates > 1:
                if (step // updateWavelength) > ((step - 1) // updateWavelength):
                    self.update(
                        step, T, E, accepts / trials, improves / trials)
                    trials, accepts, improves = 0, 0, 0

        self.state = self.copy_state(self.best_state)

        if self.save_state_on_exit:
            self.save_state()

        #modification
        self.objs = objs
        self.best_objs = best_objs
        self.best_trace = self.place_preds[self.objs.index(self.best_objs[-1])]
        self.best_int_trace = self.int_preds[self.objs.index(self.best_objs[-1])]

        self.trace_error_place = np.abs(self.best_trace - self.place_obs)
        self.trace_error_int = np.abs(self.best_int_trace - self.int_obs)

        # Return best state and energy
        return self.best_state, self.best_energy

    def move(self):
            for i in range(np.size(self.state, axis=0)):
                for j in range(np.size(self.state, axis=1)):
                    if self.state[i][j] != 0 and i not in self.external_connectivity_indices:
                        self.state[i][j] = min(max(self.state[i][j] + np.random.uniform(self.move_params['weights_change_range'][0],
                                                                                self.move_params['weights_change_range'][1]), 
                                                                            self.move_params['weights_range'][0]), self.move_params['weights_range'][1])    

    def ssd_with_l1(self, place_pred, int_pred):
        '''
        Sum of squared differences with lasso regularization cost function between two same-shape 2d numpy arrays, with interneurons
        '''

        sum_squared_difference = np.sum(0.5 * ((np.concatenate((place_pred, int_pred)) - np.concatenate((self.place_obs, self.int_obs))) ** 2))
        l1_penalty = np.sum(self.lamb * np.abs(self.state))
        return sum_squared_difference + l1_penalty

    def ssd_with_l1_with_quad(self, place_pred, int_pred):
        '''
        Sum of squared differences with lasso regularization cost function between two same-shape 2d numpy arrays and quadratic to place heavier emphasis on higher observed values
        '''

        combined_obs = np.concatenate((self.place_obs, self.int_obs))
        weights = (0.1 + (combined_obs - combined_obs.min()) * (0.9) / (combined_obs.max() - combined_obs.min())) ** 2
        sum_squared_difference = np.sum(0.5 * weights * (np.concatenate((place_pred, int_pred)) - combined_obs) ** 2) 
        l1_penalty = np.sum(self.lamb * np.abs(self.state[0]))
        return sum_squared_difference + l1_penalty
    
    def ssd_with_l1_emphasise_boundaries(self, place_pred, int_pred):
        '''
        Sum of squared differences with lasso regularization cost function between two same-shape 2d numpy arrays that place heavier emphasis on boundary observed values
        '''
        combined_obs = np.concatenate((self.place_obs, self.int_obs))
        weights = (0.01 * np.abs(combined_obs + 60) ** 2) * 0.9 + 0.1
        sum_squared_difference = np.sum(0.5 * weights * ((np.concatenate((place_pred, int_pred)) - combined_obs) ** 2))
        l1_penalty = np.sum(self.lamb * np.abs(self.state[0]))
        return sum_squared_difference + l1_penalty


    def generate_param_permutations(self):
        permutations = itertools.product(*(self.move_param_ranges[param] for param in self.move_param_ranges))
        param_perms = [dict(zip(self.move_param_ranges.keys(), permutation)) for permutation in permutations]
        return param_perms

    def get_best_spikes_pred(self):
        return self.spikes_preds[self.objs.index(self.best_objs[-1])]
    
    def get_best_spikes_pred_int(self):
        return self.int_spikes_preds[self.objs.index(self.best_objs[-1])]

    def plot_objs(self):
        plt.figure(figsize=(15, 5))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(self.objs, label='Objective Function')
        plt.plot(self.best_objs, label='Best Objective Function')
        plt.legend()

    def plot_confusion_matrix(self):
        spikes_pred = np.concatenate((self.get_best_spikes_pred(), self.get_best_spikes_pred_int()))
        spikes_obs = np.concatenate((self.spikes_obs, self.int_spikes_obs))
        cm = confusion_matrix(spikes_obs.flatten(), spikes_pred.flatten())

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.plot()

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.plot()

    def plot_obs_vs_best_pred(self, neuron_index=0):
        plt.figure(figsize=(15, 5))
        plt.plot(self.place_obs[neuron_index], label = 'Observed')
        plt.plot(self.place_preds[self.objs.index(self.best_objs[-1])][neuron_index], label = 'Predicted')
        plt.xlabel('Frame')
        plt.ylabel('Voltage (mV)')

        plt.legend()

    def plot_obs_vs_best_pred_int(self, neuron_index=0):
        plt.figure(figsize=(15, 5))
        plt.plot(self.int_obs[neuron_index], label = 'Observed')
        plt.plot(self.int_preds[self.objs.index(self.best_objs[-1])][neuron_index], label = 'Predicted')
        plt.xlabel('Frame')
        plt.ylabel('Voltage (mV)')

        plt.legend()
        
    def plot_obs_vs_best_pred_error(self, neuron_index=0):
        plt.figure(figsize=(15, 5))
        plt.plot(self.place_preds[self.objs.index(self.best_objs[-1])][neuron_index] - self.place_obs[neuron_index])
        plt.xlabel('Frame')
        plt.ylabel('Voltage (mV)')

    def plot_obs_vs_best_pred_error_int(self, neuron_index=0):
        plt.figure(figsize=(15, 5))
        plt.plot(self.int_preds[self.objs.index(self.best_objs[-1])][neuron_index] - self.int_obs[neuron_index])
        plt.xlabel('Frame')
        plt.ylabel('Voltage (mV)')

    def plot_obs_vs_best_pred_spikes(self, neuron_index=0):
        plt.figure(figsize=(15, 5))
        plt.eventplot(np.where(self.spikes_obs[neuron_index])[0], lineoffsets=0,  color = 'b', label = 'Observed')
        plt.eventplot(np.where(self.spikes_preds[self.objs.index(self.best_objs[-1])][neuron_index])[0], lineoffsets=1, color = 'r', label = 'Predicted')
        plt.xlabel('Frame')
        plt.ylabel('Spike')
        plt.legend()

    def plot_obs_vs_best_pred_spikes_int(self, neuron_index=0):
        plt.figure(figsize=(15, 5))
        plt.eventplot(np.where(self.int_spikes_obs[neuron_index])[0], lineoffsets=0,  color = 'b', label = 'Observed')
        plt.eventplot(np.where(self.int_spikes_preds[self.objs.index(self.best_objs[-1])][neuron_index])[0], lineoffsets=1, color = 'r', label = 'Predicted')
        plt.xlabel('Frame')
        plt.ylabel('Spike')
        plt.legend()

    def plot_obs_vs_first_pred(self):
        plt.figure(figsize=(15, 5))
        plt.plot(self.place_obs[1], label = 'Observed')
        plt.plot(self.place_preds[0][1], label = 'Predicted')
        plt.xlabel('Frame')
        plt.ylabel('Voltage (mV)')
        plt.legend()

    def plot_obs_vs_first_pred_int(self):
        plt.figure(figsize=(15, 5))
        plt.plot(self.int_obs[1], label = 'Observed')
        plt.plot(self.int_preds[0][1], label = 'Predicted')
        plt.xlabel('Frame')
        plt.ylabel('Voltage (mV)')
        plt.legend()

    def plot_obs_vs_first_pred_error(self):
        plt.figure(figsize=(15, 5))
        plt.plot(self.place_preds[0][1] - self.place_obs[1])
        plt.xlabel('Frame')
        plt.ylabel('Voltage (mV)')

    def plot_obs_vs_first_pred_error_int(self):
        plt.figure(figsize=(15, 5))
        plt.plot(self.int_preds[0][1] - self.int_obs[1])
        plt.xlabel('Frame')
        plt.ylabel('Voltage (mV)')

    def plot_obs_vs_first_pred_spikes(self):
        plt.figure(figsize=(15, 5))
        plt.eventplot(np.where(self.spikes_obs[1])[0], lineoffsets=0,  color = 'b', label = 'Observed')
        plt.eventplot(np.where(self.spikes_preds[0][1])[0], lineoffsets=1, color = 'r', label = 'Predicted')
        plt.xlabel('Frame')
        plt.ylabel('Spike')
        plt.legend()

    def plot_obs_vs_first_pred_spikes_int(self):
        plt.figure(figsize=(15, 5))
        plt.eventplot(np.where(self.int_spikes_obs[1])[0], lineoffsets=0,  color = 'b', label = 'Observed')
        plt.eventplot(np.where(self.int_spikes_preds[0][1])[0], lineoffsets=1, color = 'r', label = 'Predicted')
        plt.xlabel('Frame')
        plt.ylabel('Spike')
        plt.legend()

class SimulatedAnnealing2(Annealer):

    def __init__(self, weights, place_obs, spikes_obs, lamb, categorized_neurons, input_weights, move_params = None, alternate = False, cost_type='standard', runtime=3000, resolution = 0.1, biological=False):
        self.objs = []
        self.best_objs = []
        self.categorized_neurons = categorized_neurons
        self.place_obs = place_obs
<<<<<<< HEAD
        self.spikes_obs = spikes_obs
        self.place_preds = [] 
        self.spikes_preds = []
        self.lamb = lamb
        self.cost_type = cost_type
        self.alternate = alternate
        self.runtime = runtime
        self.resolution = resolution
        self.biological = biological
        if self.alternate:
            self.count = 0
=======
        self.place_preds = [] 
        self.lamb = lamb
>>>>>>> a6ec544 (Experiment + some bug fixes and small modification)

        if move_params is None:
            #Default move params
            self.move_params = {
                'weights_range': (-5, 5),
                'weights_change_range': (-0.2, 0.2),
                'input_weights_range': (-5, 5),
                'input_weights_change_range': (-2, 2),
                'input_weights_prob' : 0.2,
                'optimise_input_weights': False
            }
        else:
            self.move_params = move_params

        self.voltage_traces = None

        self.state = weights, input_weights
        self.first_weights = np.copy(weights)

    def energy(self):
        network = model.Model2(self.categorized_neurons, self.state[0], self.state[1], runtime=self.runtime, resolution=self.resolution)
        network.simulate()
        place_pred = network.get_voltage_traces()
        spikes_pred = network.get_spike_trains()

        if self.biological:
            place_pred = self.align_simulation_and_biological(self.place_obs, place_pred)
            spikes_pred = self.align_simulation_and_biological(self.spikes_obs, spikes_pred)

        self.place_preds.append(place_pred)
        self.spikes_preds.append(spikes_pred)

        if self.cost_type == 'input_weights':
            cost_function = self.ssd_with_l1_input_weights(place_pred)
        elif self.cost_type == 'vp':
            cost_function = self.vp(place_pred, spikes_pred)
        elif self.cost_type == 'quadratic':
            cost_function = self.ssd_with_l1_with_quad(place_pred)
        elif self.cost_type == 'boundaries':
            cost_function = self.ssd_with_l1_emphasise_boundaries(place_pred)
        else:    
            cost_function = self.ssd_with_l1(place_pred)

        self.voltage_traces = place_pred
        return cost_function
    
    def align_simulation_and_biological(self, obs, pred):
        return pred[:, ::33][:, :np.size(obs, axis=1)]
        

    def anneal(self):
        """Minimizes the energy of a system by simulated annealing.

        Parameters
        state : an initial arrangement of the system

        Returns
        (state, energy): the best state and energy found.
        """
        step = 0
        self.start = time.time()

        #modifications
        objs = []
        best_objs = []

        # Precompute factor for exponential cooling from Tmax to Tmin
        if self.Tmin <= 0.0:
            raise Exception('Exponential cooling requires a minimum "\
                "temperature greater than zero.')
        Tfactor = -math.log(self.Tmax / self.Tmin)

        # Note initial state
        T = self.Tmax
        E = self.energy()

        #modification
        objs.append(E)
        best_objs.append(E)

        prevState = self.copy_state(self.state)
        prevEnergy = E
        self.best_state = self.copy_state(self.state)
        self.best_energy = E
        trials, accepts, improves = 0, 0, 0
        if self.updates > 0:
            updateWavelength = self.steps / self.updates
            self.update(step, T, E, None, None)

        # Attempt moves to new states
        while step < self.steps and not self.user_exit and (step < 1000 or best_objs[-1] != best_objs[-1000]):

            step += 1
            T = self.Tmax * math.exp(Tfactor * step / self.steps)
            dE = self.move()
            if dE is None:
                E = self.energy()
                dE = E - prevEnergy
            else:
                E += dE
            trials += 1

            if dE > 0.0 and math.exp(-dE / T) < random.random():
                # Restore previous state
                self.state = self.copy_state(prevState)
                E = prevEnergy

                #Modification (change most recently added place pred to previous one)
                self.place_preds[-1] = self.place_preds[-2]

            else:
                # Accept new state and compare to best state
                accepts += 1
                if dE < 0.0:
                    improves += 1
                prevState = self.copy_state(self.state)
                prevEnergy = E
                if E < self.best_energy:
                    self.best_state = self.copy_state(self.state)
                    self.best_energy = E
            #modification
            best_objs.append(self.best_energy)
            objs.append(E)

            if self.updates > 1:
                if (step // updateWavelength) > ((step - 1) // updateWavelength):
                    self.update(
                        step, T, E, accepts / trials, improves / trials)
                    trials, accepts, improves = 0, 0, 0

        self.state = self.copy_state(self.best_state)
            
        if self.save_state_on_exit:
            self.save_state()

        #modification
        self.objs = objs
        self.best_objs = best_objs
        self.best_trace = self.place_preds[self.objs.index(self.best_objs[-1])]
        self.trace_error = np.abs(self.best_trace - self.place_obs)
        # Return best state and energy
        return self.best_state, self.best_energy

    def move(self):
        if self.alternate:
            self.move_alternate()
        else:
            self.move_no_alternate()

    def move_alternate(self):
        weights, input_weights = self.state

        if (self.count // (self.steps // 20)) % 3 == 0:
            for i in range(5):
                for j in range(5):
                    if weights[i][j] != 0:
                        weights[i][j] = min(max(weights[i][j] + np.random.uniform(self.move_params['weights_change_range'][0],
                                                                                self.move_params['weights_change_range'][1]), 
                                                                            self.move_params['weights_range'][0]), self.move_params['weights_range'][1])    
                    
        else:
            if self.move_params['optimise_input_weights']:
                for i in range(np.size(input_weights, axis=0)):
                    for j in range(np.size(input_weights, axis=1)):
                        if np.random.rand(1) < self.move_params['input_weights_prob']:
                            input_weights[i][j] = min(max(input_weights[i][j] + np.random.uniform(self.move_params['input_weights_change_range'][0],
                                                                                    self.move_params['input_weights_change_range'][1]), 
                                                                                    self.move_params['input_weights_range'][0]), self.move_params['input_weights_range'][1]) 
        self.count += 1
        self.state = weights, input_weights 

    def move_no_alternate(self):
        weights, input_weights = self.state
        
        for i in range(5):
            for j in range(5):
                if weights[i][j] != 0:
                    weights[i][j] = min(max(weights[i][j] + np.random.uniform(self.move_params['weights_change_range'][0],
                                                                            self.move_params['weights_change_range'][1]), 
                                                                         self.move_params['weights_range'][0]), self.move_params['weights_range'][1])    
                    
        #Need to modify this later so its more efficient
        if self.move_params['optimise_input_weights']:
            for i in range(np.size(input_weights, axis=0)):
                for j in range(np.size(input_weights, axis=1)):
                    if np.random.rand(1) < self.move_params['input_weights_prob']:
                        input_weights[i][j] = min(max(input_weights[i][j] + np.random.uniform(self.move_params['input_weights_change_range'][0],
                                                                                self.move_params['input_weights_change_range'][1]), 
                                                                                self.move_params['input_weights_range'][0]), self.move_params['input_weights_range'][1]) 
                        
        self.state = weights, input_weights 

    def get_best_spikes_pred(self):
        return self.spikes_preds[self.objs.index(self.best_objs[-1])]
        
    def plot_first_confusion_matrix(self):
        spikes_obs = self.spikes_obs
        spikes_pred = self.spikes_preds[0]
        cm = confusion_matrix(spikes_obs.flatten(), spikes_pred.flatten())

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.plot()

    def plot_confusion_matrix(self):
        spikes_pred = self.get_best_spikes_pred()
        spikes_obs = self.spikes_obs
        cm = confusion_matrix(spikes_obs.flatten(), spikes_pred.flatten())

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.plot()

    def plot_objs(self):
        plt.figure(figsize=(15, 5))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(self.objs, label='Objective Function')
        plt.plot(self.best_objs, label='Best Objective Function')
        plt.legend()
        
<<<<<<< HEAD
    def plot_obs_vs_best_pred(self, neuron_index=0):
        plt.figure(figsize=(15, 5))
        plt.plot(self.place_obs[neuron_index], label = 'Observed')
        plt.plot(self.place_preds[self.objs.index(self.best_objs[-1])][neuron_index], label = 'Predicted')
        if not self.biological:
            plt.xlabel('Time (ms)')
        else:
            plt.xlabel('Frame')
        plt.ylabel('Voltage (mV)')
=======
    def plot_obs_vs_best_pred(self):
        plt.title('Original Trace vs Best Predicted Trace')
        plt.plot(self.place_obs[0], label = 'Observed')
        plt.plot(self.place_preds[self.objs.index(self.best_objs[-1])][0], label = 'Predicted')
        plt.xlabel('Timestep')
        plt.ylabel('Voltage (V)')
>>>>>>> a6ec544 (Experiment + some bug fixes and small modification)

        plt.legend()

    def plot_obs_vs_best_pred_error(self, neuron_index=0):
        plt.figure(figsize=(15, 5))
        plt.plot(self.place_preds[self.objs.index(self.best_objs[-1])][neuron_index] - self.place_obs[neuron_index])
        if not self.biological:
            plt.xlabel('Time (ms)')
        else:
            plt.xlabel('Frame')
        plt.ylabel('Voltage (mV)')

    def plot_obs_vs_best_pred_spikes(self, neuron_index=0):
        plt.figure(figsize=(15, 5))
        plt.eventplot(np.where(self.spikes_obs[neuron_index])[0], lineoffsets=0,  color = 'b', label = 'Observed')
        plt.eventplot(np.where(self.spikes_preds[self.objs.index(self.best_objs[-1])][neuron_index])[0], lineoffsets=1, color = 'r', label = 'Predicted')
        if not self.biological:
            plt.xlabel('Time (ms)')
        else:
            plt.xlabel('Frame')
        plt.ylabel('Spike')

        plt.legend()

    def plot_obs_vs_first_pred(self):
        plt.figure(figsize=(15, 5))
        plt.plot(self.place_obs[0], label = 'Observed')
        plt.plot(self.place_preds[0][0], label = 'Predicted')
        if not self.biological:
            plt.xlabel('Time (ms)')
        else:
            plt.xlabel('Frame')
        plt.ylabel('Voltage (mV)')
        plt.legend()

    def plot_obs_vs_first_pred_error(self):
        plt.figure(figsize=(15, 5))
        plt.plot(self.place_preds[0][0] - self.place_obs[0])
        if not self.biological:
            plt.xlabel('Time (ms)')
        else:
            plt.xlabel('Frame')
        plt.ylabel('Voltage (mV)')

    def plot_obs_vs_first_pred_spikes(self):
        plt.figure(figsize=(15, 5))
        plt.eventplot(np.where(self.spikes_obs[0])[0], lineoffsets=0,  color = 'b', label = 'Observed')
        plt.eventplot(np.where(self.spikes_preds[0][0])[0], lineoffsets=1, color = 'r', label = 'Predicted')
        if not self.biological:
            plt.xlabel('Time (ms)')
        else:
            plt.xlabel('Frame')
        plt.ylabel('Spike')
        plt.legend()

    def ssd_with_l1(self, place_pred):
        '''
        Sum of squared differences with lasso regularization cost function between two same-shape 2d numpy arrays
        '''
        sum_squared_difference = np.sum(0.5 * (place_pred - self.place_obs) ** 2)
        l1_penalty = np.sum(self.lamb * np.abs(self.state[0]))
        return sum_squared_difference + l1_penalty
    
    def ssd_with_l1_with_quad(self, place_pred):

        '''
        Sum of squared differences with lasso regularization cost function between two same-shape 2d numpy arrays and quadratic to place heavier emphasis on higher observed values
        '''
        weights = (0.1 + (self.place_obs - self.place_obs.min()) * (0.9) / (self.place_obs.max() - self.place_obs.min())) ** 2
        sum_squared_difference = np.sum(0.5 * weights * (place_pred - self.place_obs) ** 2)
        l1_penalty = np.sum(self.lamb * np.abs(self.state[0]))
        return sum_squared_difference + l1_penalty
    
    def ssd_with_l1_emphasise_boundaries(self, place_pred):
        '''
        Sum of squared differences with lasso regularization cost function between two same-shape 2d numpy arrays that place heavier emphasis on boundary observed values
        '''
        weights = (0.01 * np.abs(self.place_obs + 60) ** 2) * 0.9 + 0.1 #Adjusted 0,009 to 0,01
        sum_squared_difference = np.sum(0.5 * weights * (place_pred - self.place_obs) ** 2)
        l1_penalty = np.sum(self.lamb * np.abs(self.state[0]))
        return sum_squared_difference + l1_penalty

    def vp(self, place_pred, spikes_pred):
        sum_squared_difference = np.sum(0.5 * (place_pred - self.place_obs) ** 2)
        total = 0
        for i in range(np.size(spikes_pred, axis=0)):
            total += vp_distance(self.spikes_obs[i], spikes_pred[i])[0][-1]
        return total * 2000 + sum_squared_difference
        
    def ssd_with_l1_input_weights(self, place_pred):
        '''
        Sum of squared differences with lasso regularization on input weights cost function between two same-shape 2d numpy arrays
        '''
        sum_squared_difference = np.sum(0.5 * (place_pred - self.place_obs) ** 2)
        l1_penalty = np.sum(self.lamb * np.abs(self.state[1]))
        return sum_squared_difference + l1_penalty


    def generate_param_permutations(self):
        permutations = itertools.product(*(self.move_param_ranges[param] for param in self.move_param_ranges))
        param_perms = [dict(zip(self.move_param_ranges.keys(), permutation)) for permutation in permutations]
        return param_perms
<<<<<<< HEAD
class SensitivityAnalysis1:
    def __init__(self, move_param_ranges, optimiser, param_keys, weights1, results_dir, num_iters = 3, save_results = False, save_calcium = False):
=======


class SensitivityAnalysis:
    def __init__(self, move_param_ranges, optimiser, param_keys, weights1, num_iters = 3, save_results = False, save_calcium = False):
>>>>>>> a6ec544 (Experiment + some bug fixes and small modification)
        self.move_param_ranges = move_param_ranges
        self.optimiser = optimiser
        self.save_results = save_results
        self.param_keys = param_keys
        self.num_iters = num_iters
        self.weights1 = weights1
<<<<<<< HEAD
        self.results_dir = results_dir
        self.save_calcium = save_calcium
        self.optimise_input_weights = move_param_ranges['optimise_input_weights'][0]
=======
        self.save_calcium = save_calcium
>>>>>>> a6ec544 (Experiment + some bug fixes and small modification)
    
    def generate_param_permutations(self):
        permutations = itertools.product(*(self.move_param_ranges[param] for param in self.move_param_ranges))
        param_perms = [dict(zip(self.move_param_ranges.keys(), permutation)) for permutation in permutations]
        return param_perms
    
    def create_experiment_dir(self):

        current_datetime = str(datetime.datetime.now())[:-7]
        results_dir = self.results_dir
        experiment_dir = results_dir + current_datetime
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
        return experiment_dir + '/'

    def run_analysis(self):

        param_perms = self.generate_param_permutations()
        param_metrics = []

        if self.save_results:
            experiment_dir = self.create_experiment_dir()

        start = time.time()

        for i in range(len(param_perms)):

            param_perm_subset = {key: param_perms[i][key] for key in self.param_keys if key in param_perms[i]}

            if self.save_results:
                perm_dir = experiment_dir + str(param_perm_subset) + '/'
                if not os.path.exists(perm_dir):
                    os.makedirs(perm_dir)
            
            if isinstance(self.optimiser, Annealer):
                self.optimiser.move_params = param_perms[i]

                if self.optimise_input_weights :
                    self.optimiser.l1_input_weights = True

                objs = []
                best_objs = []
                optimized_weights = []
                vp_distances = []
                total_trace_errors = []
                
                for run in range(self.num_iters):
                    temp_optimiser = copy.deepcopy(self.optimiser)
                    x, func = temp_optimiser.anneal()
                    best_objs.append(temp_optimiser.best_energy)
                    objs.append(temp_optimiser.objs)
                    optimized_weights.append(x)
                    
                    if self.save_results:
                        spikes_obs = temp_optimiser.spikes_obs
                        int_spikes_obs = temp_optimiser.int_spikes_obs
                        combined_trace_error = np.sum(np.abs(np.concatenate((temp_optimiser.trace_error_place, temp_optimiser.trace_error_int))))
                        total_trace_errors.append(combined_trace_error / (len(spikes_obs) + len(int_spikes_obs)))
                        best_spikes_pred = temp_optimiser.get_best_spikes_pred()
                        best_int_spikes_pred = temp_optimiser.get_best_spikes_pred_int()
                    

                        vpd = 0
                        
                        for j in range(len(spikes_obs)):
                            victor_purpura_distance = vp_distance(best_spikes_pred[j], spikes_obs[j])
                            vpd += victor_purpura_distance[0][-1]
                        for j in range(len(int_spikes_obs)):
                            victor_purpura_distance = vp_distance(best_int_spikes_pred[j], int_spikes_obs[j])
                            vpd += victor_purpura_distance[0][-1]
                        vp_distances.append(vpd / (len(spikes_obs) + len(int_spikes_obs)))

                        temp_optimiser.plot_objs()
                        plt.savefig(perm_dir + f'Run {run+1} Objective Functions.png')
                        plt.close('all')
                        
                        traces_dir = os.path.join(perm_dir, f'Run {run+1}')
                        os.mkdir(traces_dir)

<<<<<<< HEAD
                        for k in range(5):
                            temp_optimiser.plot_obs_vs_best_pred(k)
                            plt.savefig(traces_dir + f'/Place {k+1} Observed vs Predicted Calcium.png')
                            plt.close('all')
                            temp_optimiser.plot_obs_vs_best_pred_error(k)
                            plt.savefig(traces_dir + f'/Place {k+1} Observed vs Predicted Calcium Error.png')
                            plt.close('all')
                            temp_optimiser.plot_obs_vs_best_pred_spikes(k)
                            plt.savefig(traces_dir + f'/Place {k+1} Observed vs Predicted Spikes.png')
                            plt.close('all')
                            temp_optimiser.plot_obs_vs_best_pred_int(k)
                            plt.savefig(traces_dir + f'/Int {k+1} Observed vs Predicted Calcium.png')
                            plt.close('all')
                            temp_optimiser.plot_obs_vs_best_pred_error_int(k)
                            plt.savefig(traces_dir + f'/Int {k+1} Observed vs Predicted Calcium Error.png')
                            plt.close('all')
                            temp_optimiser.plot_obs_vs_best_pred_spikes_int(k)
                            plt.savefig(traces_dir + f'/Int {k+1} Observed vs Predicted Spikes.png')
                            plt.close('all')
                            temp_optimiser.plot_objs()
                            plt.savefig(traces_dir + '/Objective Function.png')
                            plt.close('all')
                            temp_optimiser.plot_confusion_matrix()
                            plt.savefig(traces_dir + '/Confusion Matrix.png')
                            plt.close('all')


                        if run == 1:
                            np.save(perm_dir + 'Observed Place Trace.npy', temp_optimiser.place_obs)
                            np.save(perm_dir + 'Observed Place Spikes.npy', temp_optimiser.spikes_obs)
                            np.save(perm_dir + 'Observed Int Trace.npy', temp_optimiser.int_obs)
                            np.save(perm_dir + 'Observed Int Spikes.npy', temp_optimiser.int_spikes_obs)
                        np.save(perm_dir + f'Run {run+1} Best Predicted Place Spikes.npy', temp_optimiser.spikes_preds[temp_optimiser.objs.index(best_objs[run])])
                        np.save(perm_dir + f'Run {run+1} Best Place Predicted.npy', temp_optimiser.place_preds[temp_optimiser.objs.index(best_objs[run])])
                        np.save(perm_dir + f'Run {run+1} Best Predicted Int Spikes.npy', temp_optimiser.int_spikes_preds[temp_optimiser.objs.index(best_objs[run])])
                        np.save(perm_dir + f'Run {run+1} Best Int Predicted.npy', temp_optimiser.int_preds[temp_optimiser.objs.index(best_objs[run])])

                        if self.save_calcium:
                            np.save(perm_dir + f'Run {run+1} Place Calcium Traces', temp_optimiser.place_preds)
                            np.save(perm_dir + f'Run {run+1} Int Calcium Traces', temp_optimiser.int_preds)
=======
                        temp_optimiser.plot_obs_vs_best_pred()
                        plt.savefig(perm_dir + f'Run {run+1} Observed vs Predicted.png')
                        plt.close('all')                 

                        if self.save_calcium:
                            np.save(perm_dir + f'Run {run+1} Calcium Traces', temp_optimiser.place_preds)

>>>>>>> a6ec544 (Experiment + some bug fixes and small modification)
                mse_between_matrices = 0
                for matrix in optimized_weights:
                    mse_between_matrices += (1 / 20) * np.sum((matrix[0] - self.weights1) ** 2) 
                mean_mse_between_matrices  = mse_between_matrices / self.num_iters
            
                param_metrics.append((*param_perm_subset.values(), round(best_objs[0], 3), round(best_objs[1], 3), 
                                      round(best_objs[2], 3), round(sum(best_objs)/len(best_objs),3 ), round(mean_mse_between_matrices), round(sum(vp_distances) / 3),
                                      round(sum(total_trace_errors) / 3)))

                if self.save_results:
                    np.save(perm_dir + 'Run 1 Optimized Weights', optimized_weights[0][0])
                    np.save(perm_dir + 'Run 2 Optimized Weights', optimized_weights[1][0])
                    np.save(perm_dir + 'Run 3 Optimized Weights', optimized_weights[2][0])

                    with open(perm_dir + 'params.pkl', 'wb') as file:
                        pickle.dump(param_perms[i], file)

        if self.save_results:
            original_mse_between_weights = 0
            original_vpd = 0

            for j in range(len(spikes_obs)):
                victor_purpura_distance = vp_distance(temp_optimiser.spikes_preds[0][j], temp_optimiser.spikes_obs[j])
                original_vpd += victor_purpura_distance[0][-1]
            for j in range(len(int_spikes_obs)):
                victor_purpura_distance = vp_distance(temp_optimiser.int_spikes_preds[0][j], temp_optimiser.int_spikes_obs[j])
                original_vpd += victor_purpura_distance[0][-1]
            original_vpd /= (len(spikes_obs) + len(int_spikes_obs))
            original_trace_error = np.sum(np.abs(np.concatenate((temp_optimiser.place_preds[0], temp_optimiser.int_preds[0])) - 
                                                    np.concatenate((temp_optimiser.place_obs, temp_optimiser.int_obs)))) / (len(spikes_obs) + len(int_spikes_obs))
            param_metrics.append(('Original', '', '', '', '' , '', round(original_mse_between_weights), round(original_vpd), round(original_trace_error)))
            end = time.time()
            duration = end - start
            df_cols = [*self.param_keys, 'Best Obj 1', 'Best Obj 2', 'Best Obj 3', 'Mean Best Obj', 'MSE between Matrices', 'Mean VP Spike Distance', 'Mean Trace Difference']
            df = pd.DataFrame(param_metrics, columns = df_cols)
            df.to_csv(experiment_dir + 'summary_stats.csv')
            np.save(experiment_dir + 'weights1', self.weights1)
            np.save(experiment_dir + 'weights2', self.optimiser.state[0])
<<<<<<< HEAD
=======
            np.save(experiment_dir + 'spike weights', self.optimiser.state[1])

>>>>>>> a6ec544 (Experiment + some bug fixes and small modification)

            weights1_statistics = get_2d_statistics(self.weights1)
            weights2_statistics = get_2d_statistics(self.optimiser.state[0])

            temp_optimiser.plot_obs_vs_first_pred()
            plt.savefig(experiment_dir + 'Original Place Observed vs Predicted.png')
            plt.close('all')  
        
            temp_optimiser.plot_obs_vs_first_pred_spikes()
            plt.savefig(experiment_dir + 'Original Place Observed vs Predicted Spikes.png')
            plt.close('all')  

            temp_optimiser.plot_obs_vs_first_pred_error()
            plt.savefig(experiment_dir + 'Original Place Observed vs Predicted Error.png')
            plt.close('all')  

            temp_optimiser.plot_obs_vs_first_pred_int()
            plt.savefig(experiment_dir + 'Original Int  Observed vs Predicted.png')
            plt.close('all')  
        
            temp_optimiser.plot_obs_vs_first_pred_spikes_int()
            plt.savefig(experiment_dir + 'Original Int Observed vs Predicted Spikes.png')
            plt.close('all')  

            temp_optimiser.plot_obs_vs_first_pred_error_int()
            plt.savefig(experiment_dir + 'Original Int Observed vs Predicted Error.png')
            plt.close('all')  

            with open(experiment_dir + 'details.txt', 'w') as file:
                file.write(f'Duration: {duration} seconds\n')
                file.write(f'Number of Iterations Per Experiment: {self.num_iters}\n')
                file.write(f'Number of Optimisation Steps: {self.optimiser.steps}\n')
                file.write(f'Lambda: {self.optimiser.lamb}\n')
                file.write(f'Cost Function: {self.optimiser.cost_type}')
                file.write(f'Cost Function at Start: {round(temp_optimiser.objs[0], 3)}\n\n')
                file.write(f'PARAMETER RANGES\n')
                for param_range in self.move_param_ranges.keys():
                    file.write(f'{param_range}: {self.move_param_ranges[param_range]}\n') 


                file.write(f'\nWEIGHTS 1 STATISTICS\n')
                for statistic in weights1_statistics.keys():
                    file.write(f'{statistic}: {round(weights1_statistics[statistic], 3)}\n') 

                file.write(f'\nWEIGHTS 2 STATISTICS\n')
                for statistic in weights2_statistics.keys():
                    file.write(f'{statistic}: {round(weights2_statistics[statistic], 3)}\n') 
        
        return param_metrics
class SensitivityAnalysis2:
    def __init__(self, move_param_ranges, optimiser, param_keys, weights1, input_weights1, results_dir, num_iters = 3, save_results = False, save_calcium = False):
        self.move_param_ranges = move_param_ranges
        self.optimiser = optimiser
        self.save_results = save_results
        self.param_keys = param_keys
        self.num_iters = num_iters
        self.weights1 = weights1
        self.input_weights1 = input_weights1
        self.results_dir = results_dir
        self.save_calcium = save_calcium
        self.optimise_input_weights = move_param_ranges['optimise_input_weights'][0]
    
    def generate_param_permutations(self):
        permutations = itertools.product(*(self.move_param_ranges[param] for param in self.move_param_ranges))
        param_perms = [dict(zip(self.move_param_ranges.keys(), permutation)) for permutation in permutations]
        return param_perms
    
    def create_experiment_dir(self):

        current_datetime = str(datetime.datetime.now())[:-7]
        results_dir = self.results_dir
        experiment_dir = results_dir + current_datetime
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
        return experiment_dir + '/'

    def run_analysis(self):

        param_perms = self.generate_param_permutations()
        param_metrics = []

        if self.save_results:
            experiment_dir = self.create_experiment_dir()

        start = time.time()

        for i in range(len(param_perms)):

            param_perm_subset = {key: param_perms[i][key] for key in self.param_keys if key in param_perms[i]}

            if self.save_results:
                perm_dir = experiment_dir + str(param_perm_subset) + '/'
                if not os.path.exists(perm_dir):
                    os.makedirs(perm_dir)
            
            if isinstance(self.optimiser, Annealer):
                self.optimiser.move_params = param_perms[i]

                if self.optimise_input_weights :
                    self.optimiser.l1_input_weights = True

                objs = []
                best_objs = []
                optimized_weights = []
                vp_distances = []
                total_trace_errors = []
                
                for run in range(self.num_iters):
                    temp_optimiser = copy.deepcopy(self.optimiser)
                    x, func = temp_optimiser.anneal()
                    best_objs.append(temp_optimiser.best_energy)
                    objs.append(temp_optimiser.objs)
                    optimized_weights.append(x)
                    
                    if self.save_results:
                        total_trace_error = np.sum(temp_optimiser.trace_error) / 5
                        total_trace_errors.append(total_trace_error)

                        best_spikes_pred = temp_optimiser.get_best_spikes_pred()
                        spikes_obs = temp_optimiser.spikes_obs

                        vpd = 0
                        for j in range(5):
                            victor_purpura_distance = vp_distance(best_spikes_pred[j], spikes_obs[j])
                            vpd += victor_purpura_distance[0][-1]
                        vp_distances.append(vpd / 5)

                        temp_optimiser.plot_objs()
                        plt.savefig(perm_dir + f'Run {run+1} Objective Functions.png')
                        plt.close('all')
                        
                        traces_dir = os.path.join(perm_dir, f'Run {run+1}')
                        os.mkdir(traces_dir)

                        for k in range(5):
                            temp_optimiser.plot_obs_vs_best_pred(k)
                            plt.savefig(traces_dir + f'/Neuron {k+1} Observed vs Predicted Calcium.png')
                            plt.close('all')
                            temp_optimiser.plot_obs_vs_best_pred_error(k)
                            plt.savefig(traces_dir + f'/Neuron {k+1} Observed vs Predicted Calcium Error.png')
                            plt.close('all')
                            temp_optimiser.plot_obs_vs_best_pred_spikes(k)
                            plt.savefig(traces_dir + f'/Neuron {k+1} Observed vs Predicted Spikes.png')
                            plt.close('all')
                            temp_optimiser.plot_objs()
                            plt.savefig(traces_dir + '/Objective Function.png')
                            plt.close('all')
                            temp_optimiser.plot_confusion_matrix()
                            plt.savefig(traces_dir + '/Confusion Matrix.png')
                            plt.close('all')

                        if run == 1:
                            np.save(perm_dir + 'Observed Trace.npy', temp_optimiser.place_obs)
                            np.save(perm_dir + 'Observed Spikes.npy', temp_optimiser.spikes_obs)
                        np.save(perm_dir + f'Run {run+1} Best Predicted Spikes.npy', temp_optimiser.spikes_preds[temp_optimiser.objs.index(best_objs[run])])
                        np.save(perm_dir + f'Run {run+1} Best Predicted.npy', temp_optimiser.place_preds[temp_optimiser.objs.index(best_objs[run])])

                        if self.save_calcium:
                            np.save(perm_dir + f'Run {run+1} Calcium Traces', temp_optimiser.place_preds)

                mse_between_matrices = 0
                for matrix in optimized_weights:
                    mse_between_matrices += (1 / 20) * np.sum((matrix[0] - self.weights1) ** 2) 
                mean_mse_between_matrices  = mse_between_matrices / self.num_iters
            
                param_metrics.append((*param_perm_subset.values(), round(best_objs[0], 3), round(best_objs[1], 3), 
                                      round(best_objs[2], 3), round(sum(best_objs)/len(best_objs),3 ), round(mean_mse_between_matrices), round(sum(vp_distances) / 3),
                                      round(sum(total_trace_errors) / 3)))

                if self.save_results:

                    np.save(perm_dir + 'Run 1 Optimized Weights', optimized_weights[0][0])
                    np.save(perm_dir + 'Run 2 Optimized Weights', optimized_weights[1][0])
                    np.save(perm_dir + 'Run 3 Optimized Weights', optimized_weights[2][0])

                    if self.optimise_input_weights :
                        np.save(perm_dir + 'Run 1 Optimized Input Weights', optimized_weights[0][1])
                        np.save(perm_dir + 'Run 2 Optimized Input Weights', optimized_weights[1][1])
                        np.save(perm_dir + 'Run 3 Optimized Input Weights', optimized_weights[2][1])

                    with open(perm_dir + 'params.pkl', 'wb') as file:
                        pickle.dump(param_perms[i], file)

        if self.save_results:

            original_mse_between_weights = (1 / 20) * np.sum((temp_optimiser.first_weights - self.weights1) ** 2) 
            original_vpd = 0
            for j in range(5):
                victor_purpura_distance = vp_distance(temp_optimiser.spikes_preds[0][j], temp_optimiser.spikes_obs[j])
                original_vpd += victor_purpura_distance[0][-1]
            original_trace_error = np.sum(np.abs(temp_optimiser.place_preds[0] - temp_optimiser.place_obs)) / 5
            original_vpd /= 5
            param_metrics.append(('Original', '', '', '', '' , '', round(original_mse_between_weights), round(original_vpd), round(original_trace_error)))

            end = time.time()
            duration = end - start
            df_cols = [*self.param_keys, 'Best Obj 1', 'Best Obj 2', 'Best Obj 3', 'Mean Best Obj', 'MSE between Matrices', 'Mean VP Spike Distance', 'Mean Trace Difference']
            df = pd.DataFrame(param_metrics, columns = df_cols)
            df.to_csv(experiment_dir + 'summary_stats.csv')
            np.save(experiment_dir + 'weights1', self.weights1)
            np.save(experiment_dir + 'weights2', self.optimiser.state[0])

            if self.optimise_input_weights :
                np.save(experiment_dir + 'spike weights1', self.input_weights1)
                np.save(experiment_dir + 'spike weights2', self.optimiser.state[1])

            weights1_statistics = get_2d_statistics(self.weights1)
            weights2_statistics = get_2d_statistics(self.optimiser.state[0])
            input_weights_statistics1 = get_2d_statistics(self.input_weights1)
            input_weights_statistics2 = get_2d_statistics(self.optimiser.state[1])

            temp_optimiser.plot_first_confusion_matrix()
            plt.savefig(experiment_dir + 'Original Confusion Matrix.png')
            plt.close('all')  

            temp_optimiser.plot_obs_vs_first_pred()
            plt.savefig(experiment_dir + 'Original Observed vs Predicted.png')
            plt.close('all')  
        
            temp_optimiser.plot_obs_vs_first_pred_spikes()
            plt.savefig(experiment_dir + 'Original Observed vs Predicted Spikes.png')
            plt.close('all')  

            temp_optimiser.plot_obs_vs_first_pred_error()
            plt.savefig(experiment_dir + 'Original Observed vs Predicted Error.png')
            plt.close('all')  

            with open(experiment_dir + 'details.txt', 'w') as file:
                file.write(f'Duration: {duration} seconds\n')
                file.write(f'Number of Iterations Per Experiment: {self.num_iters}\n')
                file.write(f'Number of Optimisation Steps: {self.optimiser.steps}\n')
                file.write(f'Lambda: {self.optimiser.lamb}\n')
                file.write(f'Alternate: {self.optimiser.alternate}\n')
                file.write(f'Cost Function: {self.optimiser.cost_type}\n')
                file.write(f'Cost Function at Start: {round(temp_optimiser.objs[0], 3)}\n\n')
                file.write(f'PARAMETER RANGES\n')
                for param_range in self.move_param_ranges.keys():
                    file.write(f'{param_range}: {self.move_param_ranges[param_range]}\n') 


                file.write(f'\nWEIGHTS 1 STATISTICS\n')
                for statistic in weights1_statistics.keys():
                    file.write(f'{statistic}: {round(weights1_statistics[statistic], 3)}\n') 

                file.write(f'\nWEIGHTS 2 STATISTICS\n')
                for statistic in weights2_statistics.keys():
                    file.write(f'{statistic}: {round(weights2_statistics[statistic], 3)}\n') 

                file.write(f'\nSPIKE WEIGHTS 1 STATISTICS\n')
                for statistic in input_weights_statistics1.keys():
                    file.write(f'{statistic}: {round(input_weights_statistics1[statistic], 3)}\n') 

                file.write(f'\nSPIKE WEIGHTS 2 STATISTICS\n')
                for statistic in input_weights_statistics1.keys():
                    file.write(f'{statistic}: {round(input_weights_statistics2[statistic], 3)}\n') 
        
        return param_metrics

def get_2d_statistics(twod_array):
    
    statistics_dict = {}

    statistics_dict['mean'] = np.mean(twod_array)
    statistics_dict['median'] = np.median(twod_array)
    statistics_dict['std'] = np.std(twod_array)
    statistics_dict['min'] = np.min(twod_array)
    statistics_dict['max'] = np.max(twod_array)

    return statistics_dict

def vp_distance(st_1, st_2):
    '''
    Returns the victor purpura distance between two spike trains
    '''
    st_1 = SpikeTrain(np.where(st_1 == 1)[0].tolist() * pq.ms, t_stop = len(st_1))
    st_2 = SpikeTrain(np.where(st_2 == 1)[0].tolist() * pq.ms, t_stop = len(st_2))
    

    return victor_purpura_distance([st_1, st_2])


    
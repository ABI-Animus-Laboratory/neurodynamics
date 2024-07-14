import numpy as np
import nest
import matplotlib.pyplot as plt

class CorrelationAnalysis:
    def __init__(self, spike_trains, place_cell_ids, interneuron_ids):
        self.spike_trains = spike_trains
        self.place_cell_ids = place_cell_ids
        self.interneuron_ids = interneuron_ids
        self.correlation_matrix = None

    def compute_correlation_matrix(self):
        total_neurons = len(self.place_cell_ids) + len(self.interneuron_ids)
        correlation_matrix = np.zeros((total_neurons, total_neurons))
        place_cell_traces = self.spike_trains[[int(x) - 1 for x in self.place_cell_ids]]
        interneuron_traces = self.spike_trains[[int(x) - 1 for x in self.interneuron_ids]]
        combined_traces = np.concatenate((place_cell_traces, interneuron_traces))
        correlation_matrix = np.corrcoef(combined_traces)
        np.fill_diagonal(correlation_matrix, np.nan)
        self.correlation_matrix = correlation_matrix

    def plot_correlation_matrix(self):
        fig, ax = plt.subplots()
        img = ax.imshow(self.correlation_matrix, cmap='plasma', aspect='auto')
        
        # Add a color bar
        cbar = plt.colorbar(img, ax=ax)
        ax.axhline(y=38.5, color='black', linewidth=1)
        ax.axvline(x=38.5, color='black', linewidth=1)
    
    def plot_box_plot(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_ylim([0, 0.13])
        flierprops = dict(marker='o', color='r', markersize=4)
        flattened_matrix = self.correlation_matrix.flatten()
        cleaned_flattened_matrix = flattened_matrix[~np.isnan(flattened_matrix)]
        plt.boxplot(cleaned_flattened_matrix, flierprops=flierprops)
        plt.show()

    def print_stats(self):
        
        place_place = self.correlation_matrix[:38, :38].flatten()
        place_place = place_place[~np.isnan(place_place)]
        
        place_interneuron =  self.correlation_matrix[:38, 38:].flatten()
        place_interneuron = place_interneuron[~np.isnan(place_interneuron)]

        interneuron_place =  self.correlation_matrix[38:, :38].flatten()
        interneuron_place = interneuron_place[~np.isnan(interneuron_place)]

        interneuron_interneuron =  self.correlation_matrix[38:, 38:].flatten()
        interneuron_interneuron = interneuron_interneuron[~np.isnan(interneuron_interneuron)]
        

        print('Place to place mean: ', np.mean(place_place))
        print('Place to place std: ', np.std(place_place))       
        print('Place to interneuron mean: ', np.mean(place_interneuron))
        print('Place to interneuron std: ', np.std(place_interneuron))        
        
        print('Interneuron to place mean: ', np.mean(interneuron_place))
        print('Interneuron to place std: ', np.std(interneuron_place))        
        
        print('Interneuron to interneuron mean: ', np.mean(interneuron_interneuron))
        print('Interneuron to neuron std: ', np.std(interneuron_interneuron))     

        flattened_matrix = self.correlation_matrix.flatten()
        cleaned_flattened_matrix = flattened_matrix[~np.isnan(flattened_matrix)]

        print('Full mean: ', np.mean(cleaned_flattened_matrix))
        print('Full std: ', np.std(cleaned_flattened_matrix))          



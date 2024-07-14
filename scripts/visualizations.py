import matplotlib.pyplot as plt
import tifffile
import numpy as np

#Plots membrane potentials over time for specified neuron ids from a multimeter
def plot_vms_from_device(device, id_list):
    
    plt.figure(figsize=(18, 5))
    plt.title(f'Membrane potential(s) for Neuron(s) {id_list}')
    for id in id_list:
        ts = device.get('events')['times'][::id]
        vms = device.get('events')['V_m'][::id]
        plt.plot(ts, vms)
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (mV)')

def plot_spikes_from_device(device, title='Raster Plot of Network Spiking Activity'):
    spike_events = device.get('events')
    spikes = spike_events['senders']
    spike_times = spike_events['times']
    plt.figure(figsize=(12,5))
    plt.title(title)
    plt.plot(spike_times, spikes, '.')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron id')
    plt.xlim(xmin=-8)

def plot_spikes_from_device_side_by_side(pyr_device, int_device, title='Raster Plot of Network Spiking Activity'):

    pyr_spike_events = pyr_device.get('events')
    pyr_spikes = pyr_spike_events['senders']
    pyr_spike_times = pyr_spike_events['times']
    int_spike_events = int_device.get('events')
    int_spikes = int_spike_events['senders']
    int_spike_times = int_spike_events['times']

    fig, axs = plt.subplots(1, 2, figsize=(24, 8))  
    plt.subplots_adjust(hspace=0.5)

    # Plot the data
    axs[0].plot(pyr_spike_times, pyr_spikes, '.', markersize=7)
    axs[1].plot(int_spike_times, int_spikes, '.', markersize=7, color='red')

    # Set the titles for individual plots
    axs[0].set_title('Place Cells', fontsize=25)
    axs[1].set_title('Interneurons', fontsize=25)

    # Set the labels and titles
    fig.suptitle(title)
    axs[0].set_ylabel('Neuron ID', fontsize=20)
    axs[0].set_xlabel('Time (ms)', fontsize=20)
    axs[1].set_xlabel('Time (ms)', fontsize=20)


    # Set whole number ticks on both axes
    axs[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axs[0].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axs[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axs[1].yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    image = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
    return image

def plot_matrix(matrix, cmap='cividis', colorbar=True):
    plt.figure(figsize=(7, 5))
    img = plt.imshow(matrix, cmap=cmap, aspect='auto')
    row_lines = [38, 58, 78, 96]
    for line in row_lines:
        plt.axhline(y=line, color='black', linestyle='-',linewidth=1)

    if colorbar:
        plt.colorbar(img)
    plt.xlabel('Neuron Id')
    plt.ylabel('Neuron Id')
    plt.show()
<<<<<<< HEAD
def save_calcium_predictions_over_optimisation(place_preds, place_obs, filename, title='Observed vs Predicted Voltage Trace', xlabel='Timestep', ylabel='Voltage (V)'):
=======

def save_calcium_predictions_over_optimisation(place_preds, place_obs, filename, xlabel='Timestep', ylabel='Voltage (V)'):
>>>>>>> a6ec544 (Experiment + some bug fixes and small modification)
    plots = []
    count = 1
    for pred in place_preds[::5]:
        print('Image: ' + str(count))
        fig = plt.figure()
        plt.plot(place_obs, label='Observed')
        plt.plot(pred, label='Predicted')
<<<<<<< HEAD
        plt.title(title)
=======
        plt.title('Observed vs Predicted Voltage Trace')
>>>>>>> a6ec544 (Experiment + some bug fixes and small modification)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        fig.canvas.draw()
        plot_array = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8)
        plot_array = plot_array.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
        plt.legend()
        plots.append(plot_array)
        plt.close()
        count += 1
    
    image_stack = np.stack(plots)
<<<<<<< HEAD
    tifffile.imwrite(filename, image_stack, photometric='rgb')

=======
    tifffile.imwrite(filename, image_stack, photometric='rgb')
>>>>>>> a6ec544 (Experiment + some bug fixes and small modification)

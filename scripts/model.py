import numpy as np
import matplotlib.pyplot as plt
import nest
import time 
from params import pyr_hcamp_deco2012, int_hcamp_deco2012

nest.rng_seed = 1
np.random.seed(1)
class Model:
    '''
    Class that represents the model
    '''
    def __init__(self, categorized_neurons, G_e = 1, G_i = -1, runtime = 17988, gamma_rate = 40, theta_rate = 7, resolution = 0.1):

        self.G_e = G_e
        self.G_i = G_i
        self.runtime = runtime
        self.gamma_rate = gamma_rate
        self.theta_rate = theta_rate

        self.resolution = resolution
        self.V_e = 5
        self.V_i = 5
        self.simulated = False             

        self.categorized_neurons = categorized_neurons
        self.num_pyr = len(categorized_neurons['Place'])
        
        self.spike_trains_pyr = None
        self.voltage_traces_pyr = None
        self.spike_recorder_pyr = None
        
        self.external_connectivity_indices = self.get_external_connectivity_indices()
    def check_simulated(self):
        '''
        Checks if a simulation has been run
        Takes in no parameters and returns a boolean
        '''

        return self.simulated
        
class Model1(Model):

    def __init__(self, categorized_neurons, weights = None, G_e = 1, G_i = -1, runtime = 17988, gamma_rate = 40, theta_rate = 7, resolution = 0.1):

        super().__init__(categorized_neurons, G_e, G_i, runtime, gamma_rate, theta_rate, resolution)

        self.spike_trains_int = None
        self.voltage_traces_int = None
        self.spike_recorder_int = None

        self.num_int = len(categorized_neurons['Interneuron'])
        if weights is None:
            self.weights = self.initialize_connectivity_matrix_normal_distribution()
        else:
            self.weights = weights

    def simulate(self):
        '''
        Runs a simulation and assigns class variables 
        Takes in no inputs and has no output
        '''

        nest.ResetKernel()

        nest.resolution = self.resolution
        
        #Initialization of pyramimdal and interneurons
        pyr = initialize_neuron_group('iaf_psc_alpha', self.num_pyr, pyr_hcamp_deco2012.params)
        inter = initialize_neuron_group('iaf_psc_alpha', self.num_int, int_hcamp_deco2012.params)

        #Initialization of EC neurons and connections
        ec_input = nest.Create('poisson_generator')
        ec_input.set(rate=self.gamma_rate)
        ec_parrot = nest.Create('parrot_neuron', n=20)
        nest.Connect(ec_input, ec_parrot)

        #Initialization of CA3 neurons and connections
        ca3_input = nest.Create('poisson_generator')
        ca3_input.set(rate=self.gamma_rate)
        ca3_parrot = nest.Create('parrot_neuron', n=20)
        nest.Connect(ca3_input, ca3_parrot)

        #Initialization of MS neurons and connections
        ms_input = nest.Create('poisson_generator')
        ms_input.set(rate=self.theta_rate)
        ms_parrot = nest.Create('parrot_neuron', n=10)
        nest.Connect(ms_input, ms_parrot)

        #Initialization of spike recorders and connections for pyramidal and interneurons
        spike_recorder_pyr = nest.Create('spike_recorder')
        nest.Connect(pyr, spike_recorder_pyr)
        spike_recorder_inter = nest.Create('spike_recorder')
        nest.Connect(inter, spike_recorder_inter)

        #Initialization of multimeter and connections for pyramidal and interneurons
        multimeter_pyr = nest.Create('multimeter')
        multimeter_pyr.set(record_from=["V_m"])
        nest.Connect(multimeter_pyr, pyr)
        multimeter_inter = nest.Create('multimeter')
        multimeter_inter.set(record_from=["V_m"])
        nest.Connect(multimeter_inter, inter)


        #Initialization of connectivity weights
        self.set_connection_weights(pyr, ec_parrot, ca3_parrot, inter, ms_parrot, self.weights, self.G_e, self.G_i, self.V_e, self.V_i,
                               self.num_pyr, self.num_int)


        nest.Simulate(self.runtime)

        self.simulated = True

        #Accessing, processing and storing recorded place cell variables from simulation
        spikes_pyr = nest.GetStatus(spike_recorder_pyr, "events")[0]
        senders = spikes_pyr["senders"]
        times = spikes_pyr["times"]
        dmm_pyr = multimeter_pyr.get()
        Vms_pyr = dmm_pyr["events"]["V_m"] 

        spike_trains_pyr = [times[senders == neuron_id] for neuron_id in pyr]
        spike_trains_pyr = simulation_results_to_spike_trains(spike_trains_pyr, self.runtime)
        self.spike_trains_pyr = spike_trains_pyr
        self.voltage_traces_pyr = tidy_Vms(Vms_pyr, self.num_pyr)
        self.spike_recorder_pyr = spike_recorder_pyr

        #Accessing, processing and storing recorded interneuron variables from simulation
        spikes_int = nest.GetStatus(spike_recorder_inter, "events")[0]
        senders = spikes_int["senders"]
        times = spikes_int["times"]
        dmm_int = multimeter_inter.get()
        Vms_int = dmm_int["events"]["V_m"] 
        spike_timings_int = [times[senders == neuron_id] for neuron_id in inter]
        spike_timings_int = simulation_results_to_spike_trains(spike_timings_int, self.runtime)
        self.spike_trains_int = spike_timings_int
        self.voltage_traces_int = tidy_Vms(Vms_int, self.num_int)
        self.spike_recorder_int = spike_recorder_inter

    def get_spike_trains(self, category):
        '''
        Gets the spike trains for each neuron in a specific category produced by the simulation
        Takes in a category paramater and returns a 2d numpy array
        '''
        if self.simulated:
            if category == 'Place':
                return self.spike_trains_pyr
            elif category == 'Inter':
                return self.spike_trains_int
            else:
                print('Not a valid category!')
                return None
    
    def get_voltage_traces(self, category):
        '''
        Gets the voltage traces for each neuron in a specific category produced by the simulation
        Takes in a category paramater and returns a 2d numpy array
        '''
        if self.simulated:
            if category == 'Place':
                return self.voltage_traces_pyr
            elif category == 'Inter':
                return self.voltage_traces_int
            else:
                print('Not a valid category!')
                return None

    def initialize_connectivity_matrix_normal_distribution(self):
        '''
        order and quantities
        num_pyr pyramidal neurons
        20 ec
        20 ca3
        num_int interneurons
        10 medial septum

        excitatory connections
        pyr -> pyr
        ca3 -> pyr, inter
        ec -> pyr, inter

        inh connections
        int -> pyr
        ms -> int
        '''

        num_pyr = len(self.categorized_neurons['Place'])
        num_int = len(self.categorized_neurons['Interneuron'])
        num_neurons = num_pyr + num_int + 50 #20 EC + 20 CA3 + 10 MS neurons = 50 total neurons

        matrix = np.zeros((num_neurons, num_neurons))
        #Pyramidal neurons
        for i in range(num_pyr):
            #Pyramidal to Pyramidal
            matrix[i][0:num_pyr] = np.abs(np.random.normal(1, scale=0.4, size=num_pyr))
        #EC
        for i in range(num_pyr, num_pyr+20):
            #EC to Pyramidal
            matrix[i][0:num_pyr] = np.abs(np.random.normal(1, scale=0.4, size=num_pyr))
            #EC to Interneurons
            matrix[i][num_pyr+40:num_pyr+40+num_int] = np.abs(np.random.normal(1, scale=0.4, size=num_int))
        #CA3
        for i in range(num_pyr+20, num_pyr+40):
            #CA3 to Pyramidal
            matrix[i][0:num_pyr] = np.abs(np.random.normal(1, scale=0.4, size=num_pyr))
            #CA3 to Interneurons
            matrix[i][num_pyr+40:num_pyr+40+num_int] = np.abs(np.random.normal(1, scale=0.4, size=num_int))
        #Interneurons
        for i in range(num_pyr+40, num_pyr+40+num_int):
            #Interneurons to Pyramidal
            matrix[i][0:num_pyr] = np.abs(np.random.normal(1, scale=0.4, size=num_pyr))
        #Medial Septum
        for i in range(num_pyr+40+num_int, num_pyr+50+num_int):
            #Medial Septum to Interneurons
            matrix[i][num_pyr+40:num_pyr+40+num_int] = np.abs(np.random.normal(1, scale=0.4, size=num_int))

        return matrix

    def get_external_connectivity_indices(self):
        num_pyr = len(self.categorized_neurons['Place'])
        num_int = len(self.categorized_neurons['Interneuron']) 
        num_ec = 20
        num_ca3 = 20
        num_ms = 10

        external_connectivity_indices = []
        for i in range(num_pyr, num_pyr+num_ec):
            external_connectivity_indices.append(i)
        for i in range(num_pyr+num_ec, num_pyr+num_ec+num_ca3):
            external_connectivity_indices.append(i)
        for i in range(num_pyr+num_ec+num_ca3+num_int, num_pyr+num_ec+num_ca3+num_int+num_ms):
            external_connectivity_indices.append(i)
        return external_connectivity_indices

    def set_connection_weights(self, pyr, ec, ca3, inter, ms, weights, G_e, G_i, V_e, V_i, num_pyr, num_int):

        '''
        Sets all connection weightings
        '''
        
        pyr_pyr_conns = weights[0:num_pyr, 0:num_pyr]
        connect_weights(pyr, pyr, pyr_pyr_conns, G_e, V_e)

        ec_pyr_conns = weights[num_pyr:num_pyr+20, 0:num_pyr]
        connect_weights(ec, pyr, ec_pyr_conns, G_e, V_e)

        ec_inter_conns = weights[num_pyr:num_pyr+20, num_pyr+40:num_pyr+40+num_int]
        connect_weights(ec, inter, ec_inter_conns, G_e, V_e)

        ca3_pyr_conns = weights[num_pyr+20:num_pyr+40, 0:num_pyr]
        connect_weights(ca3, pyr, ca3_pyr_conns, G_e, V_e)

        ca3_inter_conns = weights[num_pyr+20:num_pyr+40, num_pyr+40:num_pyr+40+num_int]
        connect_weights(ca3, inter, ca3_inter_conns, G_e, V_e)

        inter_pyr_conns = weights[num_pyr+40:num_pyr+40+num_int, 0:num_pyr]
        connect_weights(inter, pyr, inter_pyr_conns, G_i, V_i)

        ms_inter_conns = weights[num_pyr+40+num_int: num_pyr+50+num_int, num_pyr+40:num_pyr+40+num_int]
        connect_weights(ms, inter, ms_inter_conns, G_i, V_i)

class Model2(Model):

    def __init__(self, categorized_neurons, weights = None, input_weights = None, G_e = 1, G_i = -1, runtime = 3000, gamma_rate = 40, theta_rate = 7, resolution=0.1):
        super().__init__(categorized_neurons, G_e, G_i, runtime, gamma_rate, theta_rate, resolution=resolution)

        self.input_weights = input_weights
        if input_weights is None:
            self.input_weights = np.ones((5, self.runtime))
        else:
            self.input_weights = input_weights

        if weights is None:
            self.weights = self.initialize_connectivity_matrix_normal_distribution()
        else:
            self.weights = weights
        

    def simulate(self):

        nest.ResetKernel()

        nest.resolution = self.resolution
        pyr = initialize_neuron_group('iaf_psc_alpha', 5, pyr_hcamp_deco2012.params)

        spike_times = [t for t in range(1, self.runtime+1)]

        input1 = nest.Create("spike_generator", params={"spike_times": spike_times, "spike_weights": self.input_weights[0]}, n=1)
        input2 = nest.Create("spike_generator", params={"spike_times": spike_times, "spike_weights": self.input_weights[1]}, n=1)
        input3 = nest.Create("spike_generator", params={"spike_times": spike_times, "spike_weights": self.input_weights[2]}, n=1)
        input4 = nest.Create("spike_generator", params={"spike_times": spike_times, "spike_weights": self.input_weights[3]}, n=1)
        input5 = nest.Create("spike_generator", params={"spike_times": spike_times, "spike_weights": self.input_weights[4]}, n=1)


        spike_recorder_pyr = nest.Create('spike_recorder')
        nest.Connect(pyr, spike_recorder_pyr)
        multimeter_pyr = nest.Create('multimeter')
        multimeter_pyr.set(record_from=["V_m"])
        
        nest.Connect(multimeter_pyr, pyr)

        self.set_connection_weights(pyr, input1, input2, input3, input4, input5)

        nest.Simulate(self.runtime)

        self.simulated = True

        spikes_pyr = nest.GetStatus(spike_recorder_pyr, "events")[0]
        senders = spikes_pyr["senders"]
        times = spikes_pyr["times"]
        dmm_pyr = multimeter_pyr.get()
        Vms_pyr = dmm_pyr["events"]["V_m"] 


        spike_trains_pyr = [times[senders == neuron_id] for neuron_id in pyr]
        spike_trains_pyr = simulation_results_to_spike_trains(spike_trains_pyr, self.runtime)
        self.spike_timings_pyr = spike_trains_pyr
        self.voltage_traces_pyr = tidy_Vms(Vms_pyr, self.num_pyr)
        self.spike_recorder_pyr = spike_recorder_pyr

    def set_connection_weights(self, pyr, input1, input2, input3, input4, input5):

        pyr_pyr_conns = self.weights[0:5, 0:5]
        connect_weights(pyr, pyr, pyr_pyr_conns, self.G_e, self.V_e)

        input1_pyr1_cons = self.weights[5][0]
        connect_weights(input1, pyr[0], input1_pyr1_cons, self.G_e, self.V_e)

        input2_pyr2_cons = self.weights[6][1]
        connect_weights(input2, pyr[1], input2_pyr2_cons, self.G_e, self.V_e)

        input3_pyr3_cons = self.weights[7][2]
        connect_weights(input3, pyr[2], input3_pyr3_cons, self.G_e, self.V_e)
    
        input4_pyr4_cons = self.weights[8][3]
        connect_weights(input4, pyr[3], input4_pyr4_cons, self.G_e, self.V_e)

        input5_pyr5_cons = self.weights[9][4]
        connect_weights(input5, pyr[4], input5_pyr5_cons, self.G_e, self.V_e)

    def get_spike_trains(self):
        '''
        Gets the spike trains for each neuron 
        Takes in a category paramater and returns a 2d numpy array
        '''
        if self.simulated:
            return self.spike_timings_pyr
    
    def get_voltage_traces(self):
        '''
        Gets the voltage traces for each neuron 
        Takes in a category paramater and returns a 2d numpy array
        '''
        if self.simulated:
            return self.voltage_traces_pyr

    def show_raster(self):
        '''
        Displays two raster plots for place cells based on the spike trains obtained from the simulation
        Takes in no paramters and returns nothing
        '''
        if self.simulated:
            nest.raster_plot.from_device(self.spike_recorder_pyr)
        else:
            print("No simulation has been run!")

    def initialize_connectivity_matrix_normal_distribution(self):

        matrix = np.zeros((10, 10))

        for i in range(5):
            #Pyramidal to Pyramidal
            matrix[i][0:5] = np.abs(np.random.normal(1, scale=0.4, size=5))
            matrix[i + 5][i] = 1

        for i in range(10):
            matrix[i][i] = 0

        return matrix

#Helper functions for the model classes begin here

        
def tidy_Vms(Vms, num_neurons):
    '''
    Converts the Vms recorded from the simulation into a nested array of voltage traces for each neuron
    Takes in a 1d aray of voltages ([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]), the number of neurons (integer), and outputs a 2d numpy array
    '''
    voltage_traces = []
    for i in range(0, num_neurons):
        voltage_trace = []
        for j in range(i, len(Vms), num_neurons):
            voltage_trace.append(Vms[j])

        voltage_traces.append(voltage_trace)

    
    
    return np.array(voltage_traces)

def simulation_results_to_spike_trains(spike_timings, runtime):
    '''
    Input is a list of np arrays, each representing the times at which a neuron spiked
    Output is an array of arrays, each of which is a spike train
    '''

    num_neurons = len(spike_timings)
    spike_trains = np.zeros((num_neurons, runtime))

    for i in range(num_neurons):
        spike_train = np.zeros(runtime)
        for time in spike_timings[i]:
            spike_train[int(time)-1] = 1.0
        spike_trains[i] = spike_train
    return spike_trains

def connect_weights(A, B, W, G, V):

    '''
    Connects all neurons in groups A and B according to weight matrix W with global scaling of G and voltage of V
    '''
    nest.Connect(A, B, 'all_to_all', syn_spec={'weight': np.transpose(W) * G * V})

def initialize_neuron_group(type, n=1, params={}):
    neurons = nest.Create(type, n=n, params=params)
    if n>1:
        Vth = neurons.get('V_th')[0]
        Vreset = neurons.get('V_reset')[0]
    else:
        Vth = neurons.get('V_th')
        Vreset = neurons.get('V_reset')
    neurons.set({"V_m": Vreset})
    return neurons
import numpy as np
import nest
from scripts import initializations, optimization
from params import pyr_hcamp_deco2012, int_hcamp_deco2012

def run_s1(weights, G_e = 3.8, G_i = -1, runtime = 17988, gamma_rate = 40, theta_rate = 7):
    nest.ResetKernel()
    nest.resolution = 1
    V_e, V_i = 5, 5
    pyr = initializations.initialize_neuron_group('iaf_psc_alpha', 206, pyr_hcamp_deco2012.params)
    inter = initializations.initialize_neuron_group('iaf_psc_alpha', 20, int_hcamp_deco2012.params)
    ec_input = nest.Create('poisson_generator')
    ec_input.set(rate=gamma_rate)
    ec_parrot = nest.Create('parrot_neuron', n=20)
    nest.Connect(ec_input, ec_parrot)

    ca3_input = nest.Create('poisson_generator')
    ca3_input.set(rate=gamma_rate)
    ca3_parrot = nest.Create('parrot_neuron', n=20)
    nest.Connect(ca3_input, ca3_parrot)

    ms_input = nest.Create('poisson_generator')
    ms_input.set(rate=theta_rate)
    ms_parrot = nest.Create('parrot_neuron', n=10)
    nest.Connect(ms_input, ms_parrot)

    spike_recorder = nest.Create('spike_recorder')
    nest.Connect(pyr, spike_recorder)

    optimization.set_connection_weights_s1(pyr, ec_parrot, ca3_parrot, inter, ms_parrot, weights, G_e, G_i, V_e, V_i)

    nest.Simulate(runtime)

    spikes = nest.GetStatus(spike_recorder, "events")[0]
    
    senders = spikes["senders"]
    times = spikes["times"]

    results = [times[senders == neuron_id] for neuron_id in pyr]
    results = optimization.simulation_results_to_spike_trains(results, runtime)
    return results, spike_recorder
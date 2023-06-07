import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np
import neurodynex3.tools.input_factory as input_factory

# default neuron parameters (squid giant axon)
# Hodgkin, A. L., & Huxley, A. F. (1952). A quantitative description of membrane current and its application to conduction and excitation in nerve. The Journal of physiology, 117(4), 500.

El = 10.6 * b2.mV
EK = -12 * b2.mV
ENa = 115 * b2.mV
gl = 0.3 * b2.msiemens
gK = 36 * b2.msiemens
gNa = 120 * b2.msiemens
C = 1 * b2.ufarad

V_REST = 0 * b2.mV
spike_threshold = 50 * b2.mV
spike_refractory = 40 * b2.mV

eqs = '''
I_e = input_current(t,i) : amp
membrane_Im = I_e + gNa*m**3*h*(ENa-vm) + gl*(El-vm) + gK*n**4*(EK-vm) : amp
alphah = .07*exp(-.05*vm/mV)/ms    : Hz
alpham = .1*(25*mV-vm)/(exp(2.5-.1*vm/mV)-1)/mV/ms : Hz
alphan = .01*(10*mV-vm)/(exp(1-.1*vm/mV)-1)/mV/ms : Hz
betah = 1./(1+exp(3.-.1*vm/mV))/ms : Hz
betam = 4*exp(-.0556*vm/mV)/ms : Hz
betan = .125*exp(-.0125*vm/mV)/ms : Hz
dh/dt = alphah*(1-h)-betah*h : 1
dm/dt = alpham*(1-m)-betam*m : 1
dn/dt = alphan*(1-n)-betan*n : 1
dvm/dt = membrane_Im/C : volt
'''

# hippocampal and neocortical fast-spiking interneurons
# Wang, X. J., & Buzsáki, G. (1996). Gamma oscillation by synaptic inhibition in a hippocampal interneuronal network model. Journal of neuroscience, 16(20), 6402-6413.
# """
El = -65 * b2.mV
EK = -90 * b2.mV
ENa = 55 * b2.mV
gl = 0.1 * b2.msiemens
gK = 9 * b2.msiemens
gNa = 35 * b2.msiemens * 2
C = 1 * b2.ufarad

V_REST = -70 * b2.mV
spike_threshold = 1 * b2.mV
spike_refractory = 0 * b2.mV

eqs = '''
I_e = input_current(t,i) : amp
dvm/dt = (-gNa*m**3*h*(vm-ENa)-gK*n**4*(vm-EK)-gl*(vm-El)+I_e)/C : volt
alpham = 0.1/mV*10*mV/exprel(-(vm+35*mV)/(10*mV))/ms : Hz
betam = 4*exp(-(vm+60*mV)/(18*mV))/ms : Hz
dh/dt = 5*(alphah*(1-h)-betah*h) : 1
alphah = 0.07*exp(-(vm+58*mV)/(20*mV))/ms : Hz
betah = 1./(exp(-0.1/mV*(vm+28*mV))+1)/ms : Hz
dn/dt = 5*(alphan*(1-n)-betan*n) : 1
dm/dt = 5*(alpham*(1-m)-betam*m) : 1
alphan = 0.01/mV*10*mV/exprel(-(vm+34*mV)/(10*mV))/ms : Hz
betan = 0.125*exp(-(vm+44*mV)/(80*mV))/ms : Hz
'''
# """

def plot_data(state_monitor, title=None):
    """Plots the state_monitor variables ["vm", "I_e", "m", "n", "h"] vs. time.

    Args:
        state_monitor (StateMonitor): the data to plot
        title (string, optional): plot title to display
    """

    plt.subplot(311)
    plt.plot(state_monitor.t / b2.ms, state_monitor.vm[0] / b2.mV, lw=2)
    plt.xlabel("t [ms]")
    plt.ylabel("v [mV]")
    plt.grid()

    plt.subplot(312)
    plt.plot(state_monitor.t / b2.ms, state_monitor.m[0] / b2.volt, "black", lw=2)
    plt.plot(state_monitor.t / b2.ms, state_monitor.n[0] / b2.volt, "blue", lw=2)
    plt.plot(state_monitor.t / b2.ms, state_monitor.h[0] / b2.volt, "red", lw=2)
    plt.xlabel("t (ms)")
    plt.ylabel("act./inact.")
    plt.legend(("m", "n", "h"))
    plt.ylim((0, 1))
    plt.grid()

    plt.subplot(313)
    plt.plot(state_monitor.t / b2.ms, state_monitor.I_e[0] / b2.uamp, lw=2)
    plt.axis((
        0,
        np.max(state_monitor.t / b2.ms),
        min(state_monitor.I_e[0] / b2.uamp) * 1.1,
        max(state_monitor.I_e[0] / b2.uamp) * 1.1
    ))
    plt.xlabel("t [ms]")
    plt.ylabel("I [micro A]")
    plt.grid()

    if title is not None:
        plt.suptitle(title)

    plt.show()


def simulate_HH_neuron(input_current, simulation_time):
    """A Hodgkin-Huxley neuron implemented in Brian2.

    Args:
        input_current (TimedArray): Input current injected into the HH neuron
        simulation_time (float): Simulation time [seconds]

    Returns:
        StateMonitor: Brian2 StateMonitor with recorded fields
        ["vm", "I_e", "m", "n", "h"]
    """
    neuron = b2.NeuronGroup(1, eqs, method="exponential_euler")

    # parameter initialization
    neuron.vm = V_REST
    neuron.m = 0.05
    neuron.h = 0.60
    neuron.n = 0.32

    # tracking parameters
    st_mon = b2.StateMonitor(neuron, ["vm", "I_e", "m", "n", "h"], record=True)

    # running the simulation
    hh_net = b2.Network(neuron)
    hh_net.add(st_mon)
    hh_net.run(simulation_time)

    return st_mon


def current_input(start, end, amplitude):
    """
    An example to quickly get started with the Hodgkin-Huxley module.
    """
    current = input_factory.get_step_current(int(start * 1000), int(end * 1000), b2.us, amplitude * b2.uA)
    state_monitor = simulate_HH_neuron(current, (end + 30) * b2.ms)
    plot_data(state_monitor, title="HH Neuron, step current")


def gain_function(max_amplitude, time_period):

    feq_list = []
    amp_list = range(0, max_amplitude * 1000, max_amplitude * 20)
    counter = 0

    for amplitude in amp_list:

        amplitude = amplitude / 1000.0
        input_current = input_factory.get_step_current(0, time_period, b2.ms, amplitude * b2.uA)

        neuron = b2.NeuronGroup(1, eqs, refractory='vm>spike_refractory', threshold='vm>spike_threshold',
                                method="exponential_euler")
        M = b2.SpikeMonitor(neuron)

        # parameter initialization
        neuron.vm = V_REST
        neuron.m = 0.05
        neuron.h = 0.60
        neuron.n = 0.32

        # running the simulation
        hh_net = b2.Network(neuron)
        hh_net.add(M)
        hh_net.run(time_period * b2.ms)
        feq_list.append(len(list(M.values('t')[0])))

        # trails counting
        if counter == 0:
            print("Trails:", end=' ')
        counter = counter + 1
        print(counter, end=' ')
        if counter % 10 == 0:
            print('')

    plt.scatter(np.divide(amp_list, 1000), feq_list)
    plt.grid()
    plt.show()
    return feq_list


def least_time(least_amplitude, max_amplitude):

    time_list = []
    amp_list = range(least_amplitude * 1000, max_amplitude * 1000, (max_amplitude - least_amplitude) * 50)
    counter = 0

    for amplitude in amp_list:

        amplitude = amplitude / 1000.0

        for time_test in range(0, 10001, 200):
            input_current = input_factory.get_step_current(0, time_test, b2.us, amplitude * b2.uA)
            neuron = b2.NeuronGroup(1, eqs, refractory='vm>spike_refractory', threshold='vm>spike_threshold', method="exponential_euler")
            M = b2.SpikeMonitor(neuron)

            # parameter initialization
            neuron.vm = V_REST
            neuron.m = 0.05
            neuron.h = 0.60
            neuron.n = 0.32

            # running the simulation
            hh_net = b2.Network(neuron)
            hh_net.add(M)
            hh_net.run(time_test * b2.us)
            if len(list(M.values('t')[0])) > 0 or time_test == 10000:
                time_list.append(time_test)
                break

        # trails counting
        if counter == 0:
            print("Trails:", end=' ')
        counter = counter + 1
        print(counter, end=' ')
        if counter % 10 == 0:
            print('')

    plt.scatter(np.divide(amp_list, 1000), time_list)
    plt.grid()
    plt.show()
    return time_list

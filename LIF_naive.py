import numpy as np
from brian2 import *
import random

def getting_started():

    # 设置模拟参数
    duration = 100 * ms
    n_neuron = 100

    # 定义神经元模型（LIF模型）
    eqs = '''
    dv/dt = (-(v - 0) + I_ext)/ms : 1 (unless refractory)
    I_ext : 1
    '''

    # 创建神经元群
    num_neurons = 100
    neuronA = NeuronGroup(num_neurons, eqs, threshold='v > 1', reset='v = 0', refractory=5 * ms, method='euler')
    neuronB = NeuronGroup(num_neurons, eqs, threshold='v > 1', reset='v = 0', refractory=5 * ms, method='euler')
    neuron1 = NeuronGroup(num_neurons, eqs, threshold='v > 1', reset='v = 0', refractory=5 * ms, method='euler')
    neuron2 = NeuronGroup(num_neurons, eqs, threshold='v > 1', reset='v = 0', refractory=5 * ms, method='euler')
    neuronR = NeuronGroup(num_neurons, eqs, threshold='v > 1', reset='v = 0', refractory=5 * ms, method='euler')

    # 设置神经元参数
    neuronA.v = 'rand()'
    neuronB.v = 'rand()'
    neuron1.v = 'rand()'
    neuron2.v = 'rand()'
    neuronR.v = 'rand()'

    # 添加恒定电流作为输入
    neuronA.I_ext = 1.5
    neuronB.I_ext = 1.5
    neuron1.I_ext = 0
    neuron2.I_ext = 0
    neuronR.I_ext = 0

    mat = []
    label = list(range(100)) * 10
    for i in range(24):
        random.shuffle(label)
        mat.append(label[:])
    for line in mat:
        print(line)

    # 定义连接模式
    connectionAB = Synapses(neuronA, neuronB, 'w : 1', on_pre='v_post += w ')
    connectionAB.connect(i=mat[0], j=mat[1])
    connectionAB.w = '-0.5'

    connectionBA = Synapses(neuronB, neuronA, 'w : 1', on_pre='v_post += w ')
    connectionBA.connect(i=mat[2], j=mat[3])
    connectionBA.w = '-0.5'

    connection1A = Synapses(neuron1, neuronA, 'w : 1', on_pre='v_post += w ')
    connection1A.connect(i=mat[4], j=mat[5])
    connection1A.w = '0.1'

    connectionA1 = Synapses(neuronA, neuron1, 'w : 1', on_pre='v_post += w ')
    connectionA1.connect(i=mat[6], j=mat[7])
    connectionA1.w = '-0.1'

    connection2A = Synapses(neuron2, neuronA, 'w : 1', on_pre='v_post += w ')
    connection2A.connect(i=mat[8], j=mat[9])
    connection2A.w = '0.1'

    connection1B = Synapses(neuron1, neuronB, 'w : 1', on_pre='v_post += w ')
    connection1B.connect(i=mat[10], j=mat[11])
    connection1B.w = '0.1'

    connection2B = Synapses(neuron2, neuronB, 'w : 1', on_pre='v_post += w ')
    connection2B.connect(i=mat[12], j=mat[13])
    connection2B.w = '-0.1'

    connectionB2 = Synapses(neuronA, neuron1, 'w : 1', on_pre='v_post += w ')
    connectionB2.connect(i=mat[14], j=mat[15])
    connectionB2.w = '0.1'

    connection12 = Synapses(neuron1, neuron2, 'w : 1', on_pre='v_post += w ')
    connection12.connect(i=mat[16], j=mat[17])
    connection12.w = '-0.3'

    connection21 = Synapses(neuron2, neuron1, 'w : 1', on_pre='v_post += w ')
    connection21.connect(i=mat[18], j=mat[19])
    connection21.w = '-0.3'

    connectionR1 = Synapses(neuronR, neuron1, 'w : 1', on_pre='v_post += w ')
    connectionR1.connect(i=mat[20], j=mat[21])
    connectionR1.w = '1.5'

    connectionR2 = Synapses(neuronR, neuron2, 'w : 1', on_pre='v_post += w ')
    connectionR2.connect(i=mat[22], j=mat[23])
    connectionR2.w = '4'

    # 创建监视器来记录神经元活动
    mon1 = StateMonitor(neuronA, 'v', record=True)
    mon2 = StateMonitor(neuronB, 'v', record=True)

    spk1 = SpikeMonitor(neuronA)
    spk2 = SpikeMonitor(neuronB)

    # 运行模拟
    run(duration)

    spk1_all = []
    tag1 = []
    spk2_all = []
    tag2 = []
    for i in range(100):
        spk1_all.extend(list(spk1.values('t')[i]))
        spk2_all.extend(list(spk2.values('t')[i]))
        tag1.extend([i] * len(list(spk1.values('t')[i])))
        tag2.extend([i+0.5] * len(list(spk2.values('t')[i])))
    plt.scatter(spk1_all, tag1)
    plt.scatter(spk2_all, tag2)
    plt.show()

    '''
    # 绘制神经元活动
    plot(mon1.t/ms, mon1.v[7], 'b', label='Neuron 1')
    plot(mon2.t/ms, mon2.v[9], 'r', label='Neuron 2')
    xlabel('Time (ms)')
    ylabel('Membrane potential (V)')
    legend()
    show()
    '''
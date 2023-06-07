from brian2 import *

def getting_started():

    # 设置模拟参数
    duration = 50
    num_neurons = 100
    syn = num_neurons / 3  # 平均突触个数
    rand_init = 10  # N 随机初值 mV
    rand_syn = 0  # N 随机突触权重
    rand_delay = 0  # U 随机突触延迟 ms
    const_delay = 0 * ms  # 固定突触延迟 # 不能延迟
    sup_th = 1.5 * mV  # 阈上刺激保留
    sub_ur = -15 * mV  # 超极化
    refractory_time = 5 * ms  # 多留一点 IMPORTANT
    stimuli = 2.2 * nA  # Ist
    vs = 5  # ref > 3.4

    # 定义神经元模型（LIF模型）
    tau_m = 6 * ms
    tau_i = 8.75 * ms
    tau_e = 5 * ms
    u_th = 15 * mV
    u_rest = -55 * mV
    u_reset = - 55 * mV
    c_m = 0.1875 * nF
    R_m = 32 * Mohm   # tau_m / c_m

    eqs = '''
    dv/dt = (u_rest - v + R_m * (I_ext + I_st)) / tau_m : volt (unless refractory)
    I_ext : amp
    I_st : amp 
    '''
    # dI_ext/dt = -I_ext / tau_m : amp

    eqs_e = 'w : 1'
    eqs_i = 'w : 1'

    on_e = 'v_post = clip(v_post + R_m * w * (1 + rand_syn * randn()) * nA, u_rest + sub_ur, u_th  + sup_th)'
    on_i = 'v_post = clip(v_post + R_m * w * (1 + rand_syn * randn()) * nA, u_rest + sub_ur , u_th + sup_th)'

    # 创建神经元群
    neuronA = NeuronGroup(num_neurons, eqs, threshold='v > u_th', reset='v = u_reset', refractory=refractory_time, method='euler')
    neuronB = NeuronGroup(num_neurons, eqs, threshold='v > u_th', reset='v = u_reset', refractory=refractory_time, method='euler')
    neuron1 = NeuronGroup(num_neurons, eqs, threshold='v > u_th', reset='v = u_reset', refractory=refractory_time, method='euler')
    neuron2 = NeuronGroup(num_neurons, eqs, threshold='v > u_th', reset='v = u_reset', refractory=refractory_time, method='euler')
    neuronR = PoissonGroup(num_neurons // 2, rates = '(280 * vs - 950) / 3 * hertz')

    # 设置神经元参数
    neuronA.v = u_reset + rand_init * randn() * mV
    neuronB.v = u_reset + rand_init * randn() * mV
    neuron1.v = u_reset + rand_init * randn() * mV
    neuron2.v = u_reset + rand_init * randn() * mV

    # 添加恒定电流作为输入
    neuronA.I_st = stimuli
    neuronB.I_st = stimuli

    # 定义连接模式
    SynAB = Synapses(neuronA, neuronB, model=eqs_i, on_pre=on_i)
    SynAB.connect(p = 0.75 * syn / num_neurons)
    SynAB.delay = const_delay + rand_delay * rand() * ms
    SynAB.w = -0.5

    SynBA = Synapses(neuronB, neuronA, model=eqs_i, on_pre=on_i)
    SynBA.connect(p = 0.75 * syn / num_neurons)
    SynBA.delay = const_delay + rand_delay * rand() * ms
    SynBA.w = -0.5

    SynAi = Synapses(neuronA, neuronA, model=eqs_i, on_pre=on_i)
    SynAi.connect(condition = 'i != j', p = 0.75 * syn / num_neurons)
    SynAi.delay = const_delay + rand_delay * rand() * ms
    SynAi.w = -1.5

    SynBi = Synapses(neuronB, neuronB, model=eqs_i, on_pre=on_i)
    SynBi.connect(condition = 'i != j', p = 0.75 * syn / num_neurons)
    SynBi.delay = const_delay + rand_delay * rand() * ms
    SynBi.w = -1.5

    SynAe = Synapses(neuronA, neuronA, model=eqs_e, on_pre=on_e )
    SynAe.connect(condition = 'i != j', p = 0.25 * syn / num_neurons)
    SynAe.delay = const_delay + rand_delay * rand() * ms
    SynAe.w = 4

    SynBe = Synapses(neuronB, neuronB, model=eqs_e, on_pre=on_e)
    SynBe.connect(condition = 'i != j', p = 0.25 * syn / num_neurons)
    SynBe.delay = const_delay + rand_delay * rand() * ms
    SynBe.w = 4

    SynA1 = Synapses(neuronA, neuron1, model=eqs_e, on_pre=on_e)  # A1.w -> 0.75
    SynA1.connect(p = 0.25 * syn / num_neurons)
    SynA1.delay = const_delay + rand_delay * rand() * ms
    SynA1.w = 0.1

    Syn1A = Synapses(neuron1, neuronA, model=eqs_i, on_pre=on_i)
    Syn1A.connect(p = 0.75 * syn / num_neurons)
    Syn1A.delay = const_delay + rand_delay * rand() * ms
    Syn1A.w = -0.1

    SynB2 = Synapses(neuronB, neuron2, model=eqs_i, on_pre=on_i)
    SynB2.connect(p = 0.75 * syn / num_neurons)
    SynB2.delay = const_delay + rand_delay * rand() * ms
    SynB2.w = -0.1

    Syn2B = Synapses(neuron2, neuronB, model=eqs_e, on_pre=on_e)
    Syn2B.connect(p =0.25 * syn / num_neurons)
    Syn2B.delay = const_delay + rand_delay * rand() * ms
    Syn2B.w = 0.1

    Syn1B = Synapses(neuron1, neuronB, model=eqs_i, on_pre=on_i)
    Syn1B.connect(p = 0.75 * syn / num_neurons)
    Syn1B.delay = const_delay + rand_delay * rand() * ms
    Syn1B.w = -0.1

    Syn2A = Synapses(neuron2, neuronA, model=eqs_e, on_pre=on_e)
    Syn2A.connect(p = 0.25 * syn / num_neurons)
    Syn2A.delay = const_delay + rand_delay * rand() * ms
    Syn2A.w = 0.1

    Syn12 = Synapses(neuron1, neuron2, model=eqs_i, on_pre=on_i)
    Syn12.connect(p = 0.75 * syn / num_neurons)
    Syn12.delay = const_delay + rand_delay * rand() * ms
    Syn12.w = -3

    Syn21 = Synapses(neuron2, neuron1, model=eqs_i, on_pre=on_i)
    Syn21.connect(p = 0.75 * syn / num_neurons)
    Syn21.delay = const_delay + rand_delay * rand() * ms
    Syn21.w = -3

    Syn1i = Synapses(neuron1, neuron1, model=eqs_i, on_pre=on_i)
    Syn1i.connect(condition = 'i != j', p = 0.75 * syn / num_neurons)
    Syn1i.delay = const_delay + rand_delay * rand() * ms
    Syn1i.w = -1.5

    Syn1e = Synapses(neuron1, neuron1, model=eqs_e, on_pre=on_e)
    Syn1e.connect(condition = 'i != j', p = 0.25 * syn / num_neurons)
    Syn1e.delay = const_delay + rand_delay * rand() * ms
    Syn1e.w = 4

    Syn2i = Synapses(neuron2, neuron2, model=eqs_i, on_pre=on_i)
    Syn2i.connect(condition = 'i != j', p = 0.75 * syn / num_neurons)
    Syn2i.delay = const_delay + rand_delay * rand() * ms
    Syn2i.w = -1.5

    Syn2e = Synapses(neuron2, neuron2, model=eqs_e, on_pre=on_e)
    Syn2e.connect(condition = 'i != j', p = 0.25 * syn / num_neurons)
    Syn2e.delay = const_delay + rand_delay * rand() * ms
    Syn2e.w = 4

    SynR1 = Synapses(neuronR, neuron1, model=eqs_i, on_pre=on_i)
    SynR1.connect(p = 0.75 * syn / num_neurons)
    SynR1.delay = const_delay + rand_delay * rand() * ms
    SynR1.w = -1.5

    SynR2 = Synapses(neuronR, neuron2, model=eqs_e, on_pre=on_e)
    SynR2.connect(p = 0.25 * syn / num_neurons)
    SynR2.delay = const_delay + rand_delay * rand() * ms
    SynR2.w = 4


    # 创建监视器来记录神经元活动
    mon1 = StateMonitor(neuron1, 'v', record=True)
    mon2 = StateMonitor(neuron2, 'v', record=True)
    mon3 = StateMonitor(neuronA, 'v', record=True)
    mon4 = StateMonitor(neuronB, 'v', record=True)

    spk1 = SpikeMonitor(neuron1)
    spk2 = SpikeMonitor(neuron2)
    spk3 = SpikeMonitor(neuronA)
    spk4 = SpikeMonitor(neuronB)

    # 运行模拟
    run(duration * ms)

    # 绘制神经元活动
    subplot(2, 2, 1)
    plot(mon1.t/ms, mon1.v[0], 'b', label='Neuron 1')
    plot(mon2.t/ms, mon2.v[0], 'r', label='Neuron 2')
    xlabel('Time (ms)')
    ylabel('Membrane potential (V)')
    xlim(0, duration)
    legend()

    subplot(2, 2, 2)
    plot(mon3.t / ms, mon3.v[0], 'b', label='Neuron A')
    plot(mon4.t / ms, mon4.v[0], 'r', label='Neuron B')
    xlabel('Time (ms)')
    ylabel('Membrane potential (V)')
    xlim(0, duration)
    legend()

    spk1_all, spk2_all, spk3_all, spk4_all = [], [], [], []
    tag1, tag2, tag3, tag4 = [], [], [], []
    for i in range(num_neurons):
        spk1_all.extend(list(spk1.values('t')[i]))
        spk2_all.extend(list(spk2.values('t')[i]))
        spk3_all.extend(list(spk3.values('t')[i]))
        spk4_all.extend(list(spk4.values('t')[i]))
        tag1.extend([i] * len(list(spk1.values('t')[i])))
        tag2.extend([i+0.5] * len(list(spk2.values('t')[i])))
        tag3.extend([i] * len(list(spk3.values('t')[i])))
        tag4.extend([i+0.5] * len(list(spk4.values('t')[i])))

    subplot(2, 2, 3)
    scatter([x * 1000 for x in spk1_all], tag1, label = 'Neuron 1')
    scatter([x * 1000 for x in spk2_all], tag2, label = 'Neuron 2')
    xlim(0, duration)
    xlabel('Time (ms)')
    ylabel('Neuron Index')
    legend()

    subplot(2, 2, 4)
    scatter([x * 1000 for x in spk3_all], tag3, label = 'Neuron A')
    scatter([x * 1000 for x in spk4_all], tag4, label = 'Neuron B')
    xlim(0, duration)
    xlabel('Time (ms)')
    ylabel('Neuron Index')
    legend()
    show()

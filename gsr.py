import csv
import numpy as np
import pyxdf
from scipy import signal
from matplotlib import pyplot as plt
import neurokit2 as nk
import pandas as pd
import scipy
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


if __name__ == '__main__':
    file = "data/recording_3.xdf"
    num = int(file.rstrip(".xdf").lstrip("data/recording_"))
    print('Recording:', num)
    raw = pyxdf.load_xdf(file)

# Some pre-calculated  parameters
    if num == 1:
        e, g, k = 3, 2, 1
        time = (3459485, 3459665, 3459666, 3459942, 3460018, 3460445)
    elif num == 2:
        e, g, k = 0, 3, 2
        time = (3463045, 3463225, 3463226, 3463480, 3463548, 3463986)
    elif num == 3:
        e, g, k = 3, 0, 2
        time = (3541070, 3541255, 3541305, 3541608, 3541723, 3542159)
    elif num == 4:
        e, g, k = 3, 0, 1
        time = (3544184, 3544380, 3544381, 3544611, 3544841, 3545268)

    GSR_values = raw[0][g]['time_series']
    GSR_time = raw[0][g]['time_stamps']

    max_gsr = GSR_time[-1]
    min_gsr = GSR_time[0]

    duration = (max_gsr - min_gsr)
    t = round(duration)
    print(t)
    fs = 250
    total_samples = fs * t
    nyq = 0.5 * fs


    plt.rcParams['figure.figsize'] = [15, 10]
    # plt.plot(GSR_values)
    # plt.xlabel('Time stamps')
    # plt.ylabel('Raw data')
    # plt.title("GSR signal")
    # plt.show()

    index = []
    for i in range(0, len(GSR_time)):
        if int(GSR_time[i]) in time:
            index.append(i)
    cut = index[::40]
    print(cut)

    neutral = GSR_values[cut[0]:cut[1]]
    stress = GSR_values[cut[2]:cut[3]]
    calming = GSR_values[cut[4]:cut[5]]

    state = [neutral, stress, calming]


    def processData(signal):
        minutes = []
        mean = []
        for i in range(0, len(signal), 2400):
            minutes.append(signal[i:i + 2400])

        if len(minutes) >= 3:
            minutes = minutes[-5:-1:]

        for j in range(0, len(minutes)):
            sr = np.mean(minutes[j])
            mean.append(sr)
        return mean

    def gsr_stat(neutral, other):
        U = []
        P = []
        u, p = scipy.stats.mannwhitneyu(neutral, other)
        U.append(u)
        P.append(p)
        return U, P


    mean_n = processData(neutral)
    mean_s = processData(stress)
    mean_c = processData(calming)

    print('neutral', mean_n, len(mean_n))
    print('stress', mean_s, len(mean_s))
    print('calm', mean_c, len(mean_c))

    neutral_vs_stress = gsr_stat(mean_n, mean_s)
    neutral_vs_calm = gsr_stat(mean_n, mean_c)
    stress_vs_calm = gsr_stat(mean_s, mean_c)

    print(neutral_vs_stress)
    print(neutral_vs_calm)
    print(stress_vs_calm)

    output = [[], [], [], [], [], []]
    output[0] = 'neutral_vs_stress: U',neutral_vs_stress[0]
    output[1] = 'neutral_vs_stress: p',neutral_vs_stress[1]
    output[2] = 'neutral_vs_calm: U',neutral_vs_calm[0]
    output[3] = 'neutral_vs_calm: p',neutral_vs_calm[1]
    output[4] = 'stress_vs_calm: U',stress_vs_calm[0]
    output[5]= 'stress_vs_calm: p',stress_vs_calm[1]
    file_name = 'results/statistics_GSR{}.csv'.format(num)
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(output)

#plotting
    data = [mean_s, mean_c]
    fig = plt.figure(figsize=(20, 10))
    fig, ax = plt.subplots()
    box = ax.boxplot(data, labels=['Stress', 'Calm'], patch_artist=True)
    colors = ['lightgreen', 'pink']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.title('Subject ID = ' + str(num) + ': GSR values in stressful and calm state ')
    plt.savefig('results/Subject_{}/s_c_GSR.jpg'.format(str(num)))
    plt.show()

    data = [mean_n, mean_s, mean_c]
    fig = plt.figure(figsize=(20, 10))
    fig, ax = plt.subplots()
    box = ax.boxplot(data, labels=['Neutral','Stress', 'Calm'], patch_artist=True)
    colors = ['lightgreen', 'pink']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.title('Subject ID = ' + str(num) + ': GSR values in  neutral, stressful and calm state')
    plt.savefig('results/Subject_{}/n_s_c_GSR.jpg'.format(str(num)))
    plt.show()







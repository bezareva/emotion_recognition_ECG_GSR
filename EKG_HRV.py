import csv
import numpy
import numpy as np
import pyxdf
from scipy import signal
from matplotlib import pyplot as plt
import scipy
import pyhrv
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def process_data(signal, name):
    minute_s = []
    for i in range(0, len(signal), 15000):
        minute_s.append(signal[i:i + 15000])

    if len(minute_s) >= 4:
        minute_s = minute_s[-5:-1:]

    print(name, len(minute_s))

    filter_s = []
    b, a = scipy.signal.butter(N, [2 / 125, 40 / 125], 'band')
    for i in range(0, len(minute_s)):
        f = scipy.signal.filtfilt(b, a, minute_s[i])
        filter_s.append(f)

    invS = []
    for i in range(0, len(filter_s)):
        d = np.diff(filter_s[i])
        ind = (-1) * d
        invS.append(ind)

    peak_array = []
    for i in range(0, len(invS)):
        peaks, _ = scipy.signal.find_peaks(invS[i], height=55, distance=50, prominence=50)
        pik = peaks[0]
        _peak_array = [pik]

        for x in peaks[1:]:
            if x - pik >= P:
                _peak_array.append(x)
                pik = x
        peak_array.append(_peak_array)

    peaks_x = peak_array
    peaks_y = []
    for i in range(0, len(filter_s)):
        peak = filter_s[i][peak_array[i]]
        peaks_y.append(peak)

    for i in range(0, len(peaks_x)):
        for znj in range(100):
            for x in peaks_x[i][0:]:
                if peaks_y[i][peaks_x[i].index(x)] < filter_s[i][x - 1]:
                    peaks_y[i][peaks_x[i].index(x)] = filter_s[i][x - 1]
                    peaks_x[i][peaks_x[i].index(x)] = x - 1

    rr_intervals = []
    for i in range(0, len(peaks_x)):
        RR = []
        for x, y in zip(peaks_x[i][0::], peaks_x[i][1::]):
            RR.append((y - x) / fs)
        rr_intervals.append(RR)

    HRV = []
    for i in range(0, len(rr_intervals)):
        res = pyhrv.hrv(rr_intervals[i], fs, show=False)
        HRV.append(res)

    return filter_s, peaks_x, peaks_y, HRV


def process_HRV(HRV, index):
    hrv_param = []
    for i in range(0, len(index)):
        a = index[i]
        param = []
        for i in range(0, len(HRV)):
            param.append(HRV[i][a])
        hrv_param.append(param)
    return hrv_param


def hrv_stat(neutral, other):
    U = []
    P = []
    for i in range(0, len(neutral)):
        u, p = scipy.stats.mannwhitneyu(neutral[i], other[i])
        U.append(u)
        P.append(p)
    return U, P


if __name__ == '__main__':
    file = "data/recording_4.xdf"
    num = int(file.rstrip(".xdf").lstrip("data/recording_"))
    print('Recording:', num)
    raw = pyxdf.load_xdf(file)

    # Some pre-calculated  parameters
    if num == 1:
        e, g, k = 3, 2, 1
        time = (3459485, 3459665, 3459666, 3459942, 3460018, 3460445)
        N, P = 4, 150
        graph_index = [0, 1, 3, 4, 5, 6, 7, 8, 11, 18, 19, 21]

    elif num == 2:
        e, g, k = 0, 3, 2
        time = (3463045, 3463225, 3463226, 3463480, 3463548, 3463986)
        N, P = 4, 130
        graph_index = [0, 1, 3, 5, 6, 9, 10, 18, 19]

    elif num == 3:
        e, g, k = 3, 0, 2
        time = (3541070, 3541255, 3541305, 3541608, 3541723, 3542159)
        N, P = 5, 150
        graph_index = [0, 1, 2, 5, 6, 14, 16, 17, 19]
    elif num == 4:
        e, g, k = 3, 0, 1
        time = (3544184, 3544380, 3544381, 3544611, 3544841, 3545268)
        N, P = 4, 80
        graph_index = [0, 1, 2]

    hrv_index = [1, 4, 7, 8, 11, 14, 15, 16, 17, 18, 19, 23, 24, 31, 32, 44, 45, 55, 56, 63, 64, 65]
    hrv_name = ['nni_mean', 'hr', 'hr_std', 'nni_diff', 'sdnn', 'rmssd', 'sdsd', 'nn50', 'pnn50', 'nn20', 'pnn20',
                'tinn', 'tri_index', 'fft_ratio', 'fft_total', 'lomb_ratio', 'lomb_total', 'ar_ratio', 'ar_total',
                'sd1', 'sd2', 'sd_ratio']

    ECG_values = raw[0][e]['time_series'][:, 19]
    ECG_time = raw[0][e]['time_stamps']
    print(ECG_time)

    max_eeg = ECG_time[-1]
    min_eeg = ECG_time[0]

    duration = (max_eeg - min_eeg)
    t = round(duration)
    fs = 250
    total_samples = fs * t
    nyq = 0.5 * fs

    index = []
    for i in range(0, len(ECG_time)):
        if int(ECG_time[i]) in time:
            index.append(i)

    print(index)
    cut = index[::250]
    print(cut)

    neutral = ECG_values[cut[0]:cut[1]]
    stress = ECG_values[cut[2]:cut[3]]
    calm = ECG_values[cut[4]:cut[5]]

    filterN, peak_xN, peak_yN, hrvN = process_data(neutral, 'neutral')
    filterS, peak_xS, peak_yS, hrvS = process_data(stress, 'stress')
    filterC, peak_xC, peak_yC, hrvC = process_data(calm, 'calm')

    # Checking R peak position
    plt.close("all")
    # plt.scatter(peak_xN[0], peak_yN[0], color='red')
    # plt.plot(filterN[0])
    # plt.xlim([0, 1000])
    # plt.xlabel('Time stamps')
    # plt.ylabel('Amplitude (uV)')
    # plt.title("signal")
    # plt.show()
    #
    # plt.close("all")
    # plt.scatter(peak_xS[0], peak_yS[0], color='red')
    # plt.plot(filterS[0][0:1000])
    # plt.xlim([0, 1000])
    # plt.xlabel('Time stamps')
    # plt.ylabel('Amplitude (uV)')
    # plt.title("signal")
    # plt.show()
    #
    # plt.close("all")
    # plt.scatter(peak_xC[0], peak_yC[0], color='red')
    # plt.plot(filterC[0][1:1000])
    # plt.xlim([0, 1000])
    # plt.xlabel('Time stamps')
    # plt.ylabel('Amplitude (uV)')
    # plt.title("signal")
    # plt.show()

    hrv_paramN = process_HRV(hrvN, hrv_index)
    hrv_paramS = process_HRV(hrvS, hrv_index)
    hrv_paramC = process_HRV(hrvC, hrv_index)
    print('neutral hrv', hrv_paramN, len(hrv_paramN))
    print('neutral hrv', hrv_paramS, len(hrv_paramS))
    print('neutral hrv', hrv_paramC, len(hrv_paramC))

    neutral_vs_stress = hrv_stat(hrv_paramN, hrv_paramS)
    neutral_vs_calm = hrv_stat(hrv_paramN, hrv_paramC)
    stress_vs_calm = hrv_stat(hrv_paramS, hrv_paramC)

    print(neutral_vs_stress)
    print(neutral_vs_calm)
    print(stress_vs_calm)

    output = [[], [], [], [], [], [], []]
    output[0] = hrv_name
    output[1] = neutral_vs_stress[0]
    output[2] = neutral_vs_stress[1]
    output[3] = neutral_vs_calm[0]
    output[4] = neutral_vs_calm[1]
    output[5] = stress_vs_calm[0]
    output[6] = stress_vs_calm[1]
    file_name = 'results/statistics_{}.csv'.format(num)
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(output)

    # plotting
    plt.close("all")
    for i in range(0, len(graph_index)):
        a = graph_index
        data = [hrv_paramS[a[i]], hrv_paramC[a[i]]]
        fig = plt.figure(figsize=(20, 10))
        fig, ax = plt.subplots()
        box = ax.boxplot(data, labels=['Stress', 'Calm'], patch_artist=True)
        colors = ['lightgreen', 'pink']
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        plt.title('Subject ID = ' + str(num) + ': ' + hrv_name[a[i]] + ' in stressful and calm state')
        plt.savefig('results/Subject_{}/s_c_{}.jpg'.format(str(num), str(hrv_name[a[i]])))

    plt.close("all")
    for i in range(0, len(graph_index)):
        a = graph_index
        data = [hrv_paramN[a[i]], hrv_paramS[a[i]], hrv_paramC[a[i]]]
        fig = plt.figure(figsize=(20, 10))
        fig, ax = plt.subplots()
        box = ax.boxplot(data, labels=['Neutral', 'Stress', 'Calm'], patch_artist=True)
        colors = ['cyan', 'lightgreen', 'pink']
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        plt.title('Subject ID = ' + str(num) + ': ' + hrv_name[a[i]] + ' in neutral, stressful and calm state')
        plt.savefig('results/Subject_{}/n_s_c_{}.jpg'.format(str(num), str(hrv_name[a[i]])))

        plt.show()

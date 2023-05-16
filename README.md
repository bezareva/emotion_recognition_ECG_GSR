# emotion_recognition_ECG_GSR
ECG and GSR signal processing and data analysis

<hr>

Aim of this project was to detect and analyzing emotional 
reactions to stressful and calm scenarios in virtual reality games.  

ECG and GSR signals were recorded during gameplay, then  processed 
and analyzed using Python. HRV parameters were extracted from ECG 
signals, using `pyHRV` library.  
Statistical analysis (Mann-Whitney U test) showed unambiguous difference
in reactions.

## Results

**ECG signal - filtered, RR peaks detected:**

![ecg_rr_peakes.png](https://raw.githubusercontent.com/bezareva/static/master/emotional_recognition_ECG_GSR/ecg_rr_peakes.png)


**Boxplot - HRV parameters for three emotional states: no visual stimuli, stresseful and calm scenario (test subject ID = 1):**

![hrv_parameters.png](https://raw.githubusercontent.com/bezareva/static/master/emotional_recognition_ECG_GSR/hrv_parameters.png)

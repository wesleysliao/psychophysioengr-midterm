import numpy as np
from scipy import fftpack, signal, interpolate
import matplotlib.pyplot as plt
import csv


##############################################################################
# Section copied from A. Coady posted on stackoverflow, accessed 2018.10.29:
# https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary#280156
def keywithmaxval(d):
     """ a) create a list of the dict's keys and values; 
         b) return the key with the max value"""  
     v=list(d.values())
     k=list(d.keys())
     return k[v.index(max(v))]
# End of copied section
###############################################################################
 
    
###############################################################################
# Filters
#     
def lowpass(data, cutoff_Hz, order):
    nyquist_Hz = data.samplerate_Hz/2
    Wn = (cutoff_Hz/data.samplerate_Hz)/nyquist_Hz
    b, a = signal.butter(order, Wn, btype='low', analog=False)
    filtereddata = signal.lfilter(b, a, data.values)
    return Timeseries(filtereddata, timestamps=data.times, samplerate_Hz=data.samplerate_Hz, units=data.units)

def notch(data, notch_Hz, bandwidth):
    nyquist_Hz = data.samplerate_Hz/2
    w0 = notch_Hz/nyquist_Hz
    quality = w0/bandwidth
    b, a = signal.iirnotch(w0, quality)
    filtereddata = signal.lfilter(b, a, data.values)
    return Timeseries(filtereddata, timestamps=data.times, samplerate_Hz=data.samplerate_Hz, units=data.units)
#
###############################################################################


###############################################################################
# Core Classes
#     
class Timeseries:
    def __init__(self, datapoints, timestamps=None, samplerate_Hz=1.0, starttime=0.0, units=None):
        self.values = datapoints
        self.units = units
        self.samplerate_Hz = samplerate_Hz
        if timestamps is not None and (len(datapoints) == len(timestamps)):
            self.times = timestamps
        else:
            self.times = np.arange(starttime, samplerate_Hz*len(datapoints), samplerate_Hz)
        self.interp = interpolate.interp1d(self.times, self.values)
            
    def plot(self):
        plt.plot(self.times, self.values)
        plt.xlabel("Seconds")
        plt.ylabel(str(self.units))
                
class SensorData:
    def __init__(self, name, rawdata):
        self.name = name
        self.rawdata = rawdata
        
        print("        - Cleaning data")
        self.cleandata = self.process(self.rawdata)
        
    def process(self, data):
        return data
    
    def plot(self):
        self.rawdata.plot()
        self.cleandata.plot()
        plt.legend(["Raw Data", "Cleaned Data"])
        plt.title(self.name)
     

class PeriodicSensorData(SensorData):
    def __init__(self, name, rawdata):
        super().__init__(name, rawdata)

#        print("        - Processing FFT")
#        self.fft = fftpack.fftshift(fftpack.rfft(self.cleandata.values))
#        bins = len(self.fft)
#        nyquist = self.cleandata.samplerate_Hz/2
#        freqmax = nyquist/2
#        self.freqrange = np.linspace(-freqmax, freqmax, bins)
        
        
        print("        - Finding Peaks")
        self.peaks = self.find_peaks(self.cleandata)
        
    def plot(self):
        super().plot()
        plt.plot(self.peaks.times, self.peaks.values, linestyle="None", marker="^", markersize=10)
        plt.legend(["Raw Data", "Cleaned Data", "Peaks"])
    
#    def plotfft(self):
#        plt.plot(self.freqrange, self.fft)
#        plt.xlabel("Frequency (Hz)")
#        plt.ylabel(str(self.cleandata.units))
    
    def find_peaks(self,data):
        peaks, properties = signal.find_peaks(data.values)
        if len(peaks)>0:
            return Timeseries(data.values[peaks], timestamps=data.times[peaks], samplerate_Hz=None)
        else:
            return None
#
###############################################################################


###############################################################################
# Sensor Data Classes
#     
class BloodPressure(PeriodicSensorData):
    def __init__(self, data, times, peakinterval, lowbound):
        self.peakinterval = peakinterval
        self.lowbound = lowbound
    
        
        rawdata = Timeseries(data, timestamps=times, samplerate_Hz=1000, units="mmHg")
        super().__init__("Blood Pressure", rawdata)
    
    def process(self, data):
        cutoff_Hz = 12500
        order = 1
        return lowpass(data, cutoff_Hz, order)
    
    def find_peaks(self,data):
        peaks, properties = signal.find_peaks(data.values, distance=self.peakinterval, height=self.lowbound)
        if len(peaks)>0:
            return Timeseries(data.values[peaks], timestamps=data.times[peaks], samplerate_Hz=None)
        else:
            return None


class RSP(PeriodicSensorData):
    def __init__(self, data, times):
        rawdata = Timeseries(data, timestamps=times, samplerate_Hz=1000, units="volts")
        super().__init__("RSP", rawdata)
    
    def process(self, data):
        cutoff_Hz = 100
        order = 1
        lp = lowpass(data, cutoff_Hz, order)
        
        notch_Hz = 0.0001
        bandwidth = 0.0001
        return notch(lp, notch_Hz, bandwidth) 
    
    def find_peaks(self,data):
        peaks, properties = signal.find_peaks(data.values, height=0)
        if len(peaks)>0:
            return Timeseries(data.values[peaks], timestamps=data.times[peaks], samplerate_Hz=None)
        else:
            return None
        
class PPG(PeriodicSensorData):
    def __init__(self, data, times, peakinterval, lowbound):
        self.peakinterval = peakinterval
        self.lowbound = lowbound
        
        rawdata = Timeseries(data, timestamps=times, samplerate_Hz=1000, units="volts")
        super().__init__("PPG", rawdata)

    
    def process(self, data):
        cutoff_Hz = 1000
        order = 1
        return lowpass(data, cutoff_Hz, order)
    
    def find_peaks(self,data):
        peaks, properties = signal.find_peaks(data.values, distance=self.peakinterval, height=self.lowbound)
        if len(peaks)>0:
            return Timeseries(data.values[peaks], timestamps=data.times[peaks], samplerate_Hz=None)
        else:
            return None

class AvgBloodPressure(SensorData):
    def __init__(self, data, times):
        rawdata = Timeseries(data, timestamps=times, samplerate_Hz=1000, units="mmHg")
        super().__init__("Mean Arterial Pressure", rawdata)
        
class AvgPulse(SensorData):
    def __init__(self, data, times):
        rawdata = Timeseries(data, timestamps=times, samplerate_Hz=1000, units="BPM")
        super().__init__("Pulse", rawdata)
    
class ElectrodermalActivity(SensorData):
    def __init__(self, data, times):
        rawdata = Timeseries(data, timestamps=times, samplerate_Hz=1000, units="microsiemens")
        super().__init__("Electrodermal Activity", rawdata)
    
    def process(self, data):
        timestep = 0.001
        timerange = np.arange(0,60,timestep)
        decay = 10
        weightingfunc = (np.exp(-timerange/decay)/decay)*timestep
        filtereddata = signal.fftconvolve(data.values, weightingfunc)[:len(data.values)]
        notch_Hz = 0.001
        bandwidth = 0.001
        return notch(data, notch_Hz, bandwidth)
        return Timeseries(filtereddata, timestamps=data.times, samplerate_Hz=data.samplerate_Hz, units=data.units)
#
###############################################################################


###############################################################################
# Parameter Classes
#            
class Parameter():
    def __init__(self, name, sensor):
        self.name = name
        self.rawdata = self.extract(sensor)
        self.cleandata = self.process(self.rawdata)
    def extract(self, sensor):
        return sensor.cleandata
    def process(self, data):
        return data
    
class SystolicPressure(Parameter):
    def extract(self, sensor):
        return Timeseries(sensor.peaks.values,
                          timestamps=sensor.peaks.times,
                          samplerate_Hz=None,
                          units=sensor.peaks.units)
class IBI(Parameter):
    def extract(self, sensor):
        return Timeseries(np.diff(sensor.peaks.times),
                          timestamps=sensor.peaks.times[1:],
                          samplerate_Hz=None,
                          units=sensor.peaks.units)
    def process(self, data):
        dropped = Timeseries(data.values[np.where(data.values<1.2)],
                          timestamps=data.times[np.where(data.values<1.2)],
                          samplerate_Hz=None,
                          units=data.units)
        timerange= np.arange(dropped.times[0], dropped.times[-1], 0.01)
        resampled = Timeseries(dropped.interp(timerange), timerange, samplerate_Hz=100, units="seconds")
        return lowpass(resampled, 5, 1)
        
class RespRMS(Parameter):
    def extract(self, sensor):
        sampleperiod = 0.001
        fs = 1000
        seconds = 20
        overlap = 0.95
        timerange = np.arange(sensor.cleandata.times[0], sensor.cleandata.times[-1], sampleperiod)
        f, t, Sxx = signal.spectrogram(sensor.cleandata.interp(timerange),
                                       fs,
                                       nperseg=seconds*fs,
                                       noverlap=seconds*fs*overlap,
                                       scaling='spectrum')
#        plt.figure()
#        plt.pcolormesh(Sxx)
#        plt.ylim([0, seconds])
        return Timeseries(np.sqrt(np.sum(Sxx, axis=0)),
                          timestamps=t,
                          samplerate_Hz=fs,
                          units="RMS Volts")
#
###############################################################################


###############################################################################
# Tests
#
        
class Test():
    def __init__(self, trials):
        self.scores = dict()
        for subject in trials:
            self.scores[subject] = self.procedure(trials[subject])
            
    def procedure(self, trial):
        return 1

class PrePostCompare(Test):    
    def __init__(self, trials, events, offset, delay, signal):
        self.events = events
        self.offset = offset
        self.delay = delay
        self.signal = signal
        self.sampleperiod = 0.01
        super().__init__(trials)
        
    def procedure(self, trial):
        scores = dict()
        for event in self.events:
            try:
                pre = trial.signals[self.signal].cleandata.interp(self.prerange(trial, event))
                post = trial.signals[self.signal].cleandata.interp(self.postrange(trial, event))
                scores[event] = self.compare(self.scorepre(pre), self.scorepost(post))
            except KeyError:
                scores[event] = None
        return scores
    
    def prerange(self, trial, event):
         trigger = trial.events[event]+self.offset
         return np.arange(trigger-self.delay, trigger, self.sampleperiod)
         
    def postrange(self, trial, event):
         trigger = trial.events[event]+self.offset
         return np.arange(trigger, trigger+self.delay, self.sampleperiod)
     
    def scorepre(self, pre):
        return np.sum(pre)
    
    def scorepost(self, post):
        return self.scorepre(post)
    
    def compare(self, pre, post):
        return (pre-post).astype(float)
    
class BasePostCompare(PrePostCompare):
    def prereange(self, trial, event):
        baseline_start = trial.events["BLS"]
        if  trial.signals[self.signal].cleandata.times[0] > trial.events["BLS"]:
            baseline_start = trial.signals[self.signal].cleandata.times[0]    
        baseline_end = trial.events["BLE"]
        return np.arange(baseline_start, baseline_end, self.sampleperiod)

class PrePostMax(PrePostCompare):
    def scorepre(self, pre):
        return np.mean(pre)
    def scorepost(self, post):
        return np.max(post)
    
class PrePostMin(PrePostCompare):
    def scorepre(self, pre):
        return np.mean(pre)
    def scorepost(self, post):
        return np.min(post)
    
class PrePostRel(PrePostCompare):
    def compare(self, pre, post):
        return (pre/post).astype(float)-1.0
    

class BasePostMax(BasePostCompare):
    def scorepre(self, pre):
        return np.mean(pre)
    def scorepost(self, post):
        return np.max(post)

class BasePostMin(BasePostCompare):
    def scorepre(self, pre):
        return np.mean(pre)
    def scorepost(self, post):
        return np.min(post)

class BasePostRel(BasePostCompare):
    def compare(self, pre, post):
        return (pre/post).astype(float)-1.0
           
#
###############################################################################


###############################################################################
# Trial Class
#     
class Trial:
    def __init__(self, subjectname, rawdata, param):
        print("+ Creating "+subjectname)
        self.name = subjectname
                
        print("    - Reading event data from file")        
        rawevents = np.loadtxt(param["eventfile"], dtype=str, delimiter=',', skiprows=2)
        
        print("    - Processing event data")
        self.events = dict()
        for i in range(len(rawevents)):
            self.events[rawevents[i, 1]]=(rawevents[i,0].astype(float))*60.0 #convert minutes to seconds
        
        times = (rawdata[:,0]*60)
        bp_data = rawdata[:,1]
        rsp_data = rawdata[:,2]
        ppg_data = rawdata[:,3]
#        mbp_data = rawdata[:,4]
#        bpm_data = rawdata[:,5]
        eda_data = rawdata[:,6]
        
        self.signals = dict()        
        
        print("    - Processing Blood Pressure data")
        bp = BloodPressure(bp_data, times, param["bp_peakdist"], param["bp_lowbound"])        
        self.signals[bp.name] = bp
        
        sbp = SystolicPressure("Systolic Pressure", self.signals[bp.name])
        self.signals[sbp.name] = sbp
        
        
        print("    - Processing RSP data")
        rsp = RSP(rsp_data, times)        
        self.signals[rsp.name] = rsp
        
        rrms = RespRMS("RespRMS", self.signals[rsp.name])
        self.signals[rrms.name] = rrms
        
        print("    - Processing PPG data")
        ppg = PPG(ppg_data, times,  param["ppg_peakdist"], param["ppg_lowbound"])        
        self.signals[ppg.name] = ppg
        
        ibi = IBI("IBI", self.signals[ppg.name])
        self.signals[ibi.name] = ibi
        
#        mbp = AvgBloodPressure(mbp_data, times)        
#        self.signals[mbp.name] = mbp
#        
#        bpm = AvgPulse(bpm_data, times)        
#        self.signals[bpm.name] = bpm
        
        print("    - Processing EDA data")
        eda = ElectrodermalActivity(eda_data, times)        
        self.signals[eda.name] = eda
        
    def plot_events(self, y):
        for event in self.events:
            plt.annotate(event, (self.events[event], y))

def load_data(params):
    print("+ Loading trial data")
    rawdata = dict()
    for subject in iter(params):
        print("    - loading "+subject)
        rawdata[subject] = np.genfromtxt(params[subject]["datafile"], delimiter='	', skip_header=17)
    return rawdata

def generate_subjects(params, rawdata):
    subjects = dict()
    for subject in iter(params):
        subjects[subject] = Trial(subject, rawdata[subject], params[subject])
    return subjects
#
###############################################################################


###############################################################################
# Experiment Parameters
#  
params = {    "Subject A": {"datafile":     "data/Subject A.txt",
                            "eventfile":    "events/SubjectA_events.csv",
                            "bp_peakdist":  500,
                            "bp_lowbound":  50,
                            "ppg_peakdist":  500,
                            "ppg_lowbound":  0},

              "Subject B": {"datafile":     "data/Subject B.txt",
                            "eventfile":    "events/SubjectB_events.csv",
                            "bp_peakdist":  300,
                            "bp_lowbound":  50,
                            "ppg_peakdist":  300,
                            "ppg_lowbound":  0},
                            
              "Subject C": {"datafile":     "data/Subject C.txt",
                            "eventfile":    "events/SubjectC_events.csv",
                            "bp_peakdist":  500,
                            "bp_lowbound":  80,
                            "ppg_peakdist":  500,
                            "ppg_lowbound":  0},
                            
              "Subject D": {"datafile":     "data/Subject D.txt",
                            "eventfile":    "events/SubjectD_events.csv",
                            "bp_peakdist":  500,
                            "bp_lowbound":  100,
                            "ppg_peakdist":  500,
                            "ppg_lowbound":  0},
                            
              "Subject E": {"datafile":     "data/Subject E.txt",
                            "eventfile":    "events/SubjectE_events.csv",
                            "bp_peakdist":  500,
                            "bp_lowbound":  20,
                            "ppg_peakdist":  500,
                            "ppg_lowbound":  0},
                            
              "Subject F": {"datafile":     "data/Subject F.txt",
                            "eventfile":    "events/SubjectF_events.csv",
                            "bp_peakdist":  500,
                            "bp_lowbound":  85,
                            "ppg_peakdist":  500,
                            "ppg_lowbound":  0},
                            
              "Subject G": {"datafile":     "data/Subject G.txt",
                            "eventfile":    "events/SubjectG_events.csv",
                            "bp_peakdist":  500,
                            "bp_lowbound":  65,
                            "ppg_peakdist":  500,
                            "ppg_lowbound":  0}
             }

eventsets = {   "Mailbox": ["4B","5B","6B","7B","8B","9B","1C","2C"],
             "Bag": ["8C","9C","1D","2D","3D","4D","5D","6D"],
             "Cash": ["4B","5B","6B","7B","8C","9C","1D","2D"],
             "Check": ["8B","9B","1C","2C","3D","4D","5D","6D"],
             "$41": ["6B", "1D", "1C", "5D"],
             "$470.16": ["7B","2D","2C","6D"]
         }

events_pertinent = set()
for eventset in eventsets:
    events_pertinent.update(eventsets[eventset])

#rawdata = load_data(params)
subjects = generate_subjects(params, rawdata)

quesdelay = 3
tests = { 
          "Pre-Post EDA": PrePostRel(subjects, events_pertinent, quesdelay+1,  8, "Electrodermal Activity"),
          "Pre-Post Systolic Pressure": PrePostRel(subjects, events_pertinent, quesdelay+3, 11, "Systolic Pressure"),
          "Pre-Post Resp RMS": PrePostRel(subjects, events_pertinent, quesdelay+1, 20, "RespRMS"),
          "Pre-Post IBI": PrePostRel(subjects, events_pertinent, quesdelay+4, 10, "IBI"),

#          "Pre-Post Binary EDA": PrePostBin(subjects, events_pertinent, 1, 8, "Electrodermal Activity", 0.05),
#          "Pre-Post Binary Systolic Pressure": PrePostBin(subjects, events_pertinent, 20, "Systolic Pressure", 0.05),
#          "Pre-Post Binary IBI": PrePostBin(subjects, events_pertinent, 4, 10, "IBI", 0.05),

          "Baseline-Post EDA": BasePostRel(subjects, events_pertinent, quesdelay+1, 8, "Electrodermal Activity"),
          "Baseline-Post Systolic Pressure": BasePostRel(subjects, events_pertinent, quesdelay+3, 11, "Systolic Pressure"),
          "Baseline-Post Resp RMS": BasePostRel(subjects, events_pertinent, quesdelay+1, 20, "RespRMS")
          }

test_weights = { "Pre-Post EDA":                         1.0,
                 "Pre-Post Systolic Pressure":           1.0,
                 "Pre-Post Resp RMS":                   -1.0,
                 "Pre-Post IBI":                        1.0,
                 
                 "Pre-Post Binary EDA":                  1.0,
                 "Pre-Post Binary Systolic Pressure":    1.0,
                 "Pre-Post Binary IBI":                 1.0,
                 
                 "Baseline-Post EDA":                    1.0,
                 "Baseline-Post Systolic Pressure":      1.0,
                 "Baseline-Post Resp RMS":              -1.0}
#
###############################################################################


###############################################################################
# Main Program
#  
test_scores = dict()
for test in tests:
    print("")
    print("From "+test+" test results:")
    test_scores[test] = dict()
    for subject in subjects:
        test_scores[test][subject] = dict()
        for eventset in eventsets:
            score = 0.0
            for event in eventsets[eventset]:
                if tests[test].scores[subject][event] is not None:
                    score += (tests[test].scores[subject][event])*test_weights[test]
            test_scores[test][subject][eventset] = score

        place = "Bag"
        if test_scores[test][subject]["Mailbox"] > test_scores[test][subject]["Bag"]:
            place = "Mailbox"
        item = "a Check for $470.16"
        if (test_scores[test][subject]["Cash"]+test_scores[test][subject]["$41"]
             > test_scores[test][subject]["Check"]+test_scores[test][subject]["$470.16"]):
            item = "$41 in Cash"
                
        print("    "+subject+" stole "+item+" from "+place)

print("")
print("Aggregate Test Significance:")
for test in test_weights:
    print(test, test_weights[test])
    
print("")
print("Aggregating test results:")
aggregate = dict()
for subject in subjects:
    aggregate[subject] = dict()
    for eventset in eventsets:
        score = 0
        for test in tests:
            score += test_scores[test][subject][eventset]
        aggregate[subject][eventset] = score
        
    place = "Bag"
    if aggregate[subject]["Mailbox"] > aggregate[subject]["Bag"]:
        place = "Mailbox"
    item = "a Check for $470.16"
    if (aggregate[subject]["Cash"]+aggregate[subject]["$41"]
         > aggregate[subject]["Check"]+aggregate[subject]["$470.16"]):
        item = "$41 Cash"
        
    print("    "+subject+" stole "+item+" from "+place)
#
###############################################################################


###############################################################################
# Plots
#  
#for subject in subjects.values(): 
#    plt.figure(subject.name+" Blood Pressure")
#    plt.clf()
#    subject.signals["Blood Pressure"].plot()
#    subject.signals["Systolic Pressure"].cleandata.plot()
#    subject.plot_events(70)
#    
#    plt.figure(subject.name+" EDA")
#    plt.clf()
#    subject.signals["Electrodermal Activity"].plot()
#    subject.plot_events(10)
#        
#    plt.figure(subject.name+" RSP")
#    plt.clf()
#    subject.signals["RSP"].cleandata.plot()
#    plt.plot(subject.signals["RSP"].peaks.times, subject.signals["RSP"].peaks.values/np.sqrt(2))
#    subject.signals["RespRMS"].rawdata.plot()
#    subject.plot_events(2)
#   
#    plt.figure(subject.name+" PPG")
#    plt.clf()
#    subject.signals["PPG"].plot()
#    subject.plot_events(10)
#     
#    plt.figure(subject.name+" IBI")
#    plt.clf()
#    subject.signals["IBI"].rawdata.plot()
#    subject.signals["IBI"].cleandata.plot()
#    subject.plot_events(0)    
#
###############################################################################


###############################################################################
# .csv output
#  
file = open('./results.csv','w', newline='')
wr = csv.writer(file)
wr.writerow(["== Direct Test Results =="])
for test in tests:
    wr.writerow([])
    wr.writerow([test])
    wr.writerow([" "]+list(tests[test].scores["Subject A"].keys()))
    for subject in subjects:
        wr.writerow([subject]+list(tests[test].scores[subject].values()))

wr.writerow([])
wr.writerow([])
wr.writerow(["== Question Summed Test Results =="])
for test in test_scores:
    wr.writerow([])
    wr.writerow([test])
    wr.writerow([" "]+list(test_scores[test]["Subject A"].keys()))
    for subject in subjects:
        wr.writerow([subject]+list(test_scores[test][subject].values()))

wr.writerow([])
wr.writerow([])
wr.writerow(["== All Test Aggregate Results =="])
wr.writerow([])
wr.writerow([test])
wr.writerow([" "]+list(aggregate["Subject A"].keys()))
for subject in subjects:
    wr.writerow([subject]+list(aggregate[subject].values()))

file.flush()
file.close()
#
###############################################################################

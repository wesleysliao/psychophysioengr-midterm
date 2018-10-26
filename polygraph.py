import numpy as np
from scipy import fftpack, signal, interpolate

class Data:
    def __init__(self, dataseries, timestamps=None, samplerate=1.0, starttime=0.0, units=None):
        self.values = dataseries
        if timestamps is not None and (len(dataseries) == len(timestamps)):
            self.times = timestamps
        else:
            self.times = np.arange(starttime, samplerate*len(dataseries), samplerate)
                
class SensorData:
    def __init__(self, name, rawdata):
        self.rawdata = rawdata
        self.cleandata = self.process(self.rawdata)
        
    def process(self, data):
        return data


class PeriodicSensorData(SensorData):
    def __init__(self, name, rawdata):
        super().__init__(self, name, rawdata)
        
        self.fft = fftpack.fftshift(fftpack.fft(self.cleandata))
        
        self.peaks = self.find_peaks(self.cleandata)
    
    def find_peaks(self,data):
        signal.argrelmax(data.values)
        
        return 
    
class Parameter():
    def __init__(self, SensorData):
        self.data = self.extract(SensorData.cleandata)
        self.interp = interpolate.interp1d(self.cleandata, )
        
    def extract(self, data):
        return np.diff(data)
    
    
class Test():
    def __init__(self, Trials):
        pass
    
    def procedure():
        pass



class Trial:
     
    def __init__(self, subjectname, datafilename, delimiter=',', skipheader=0):
        self.subjectname = subjectname
        self.rawdata = np.genfromtxt(datafilename,
                         delimiter=delimiter,
                         skip_header=skipheader)
        
        self.sensors = []
        self.parameters = []
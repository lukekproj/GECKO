"""
exam_load.py (KINARM file reader)

Provenance:
- This file is provided by / derived from KINARM tooling (vendor/lab-standard).
- It implements low-level parsing of .kinarm zip archives into Trial objects.

Maintenance policy:
- DO NOT modify parsing logic unless you have a strong reason and test coverage.
- Other modules (e.g., KinarmDataExplorer) depend on the exact structure:
    - ExamLoad.trials: Dict[str, Trial]
    - Trial.kinematics[label].values
    - Trial.positions[label].values
    - ExamLoad internal ordering list: _ExamLoad__exam_data (used to preserve file order)

If you need new features, prefer writing wrappers/utilities elsewhere rather than
changing this file.
"""

from zipfile import ZipFile
from os.path import split
from struct import unpack
from array import array

from sys import byteorder

from typing import List, Set, Dict, Tuple


class ExamEvent:
    def __init__(self, input):
        self.label:str = read_string(input)
        self.description:str = read_string(input)
        self.time:float = read_float(input)

class Kinematics:
    def __init__(self, zip_file, fname, trial):
        with zip_file.open(fname, 'r') as kinematicsIn:
            version = read_int(kinematicsIn)
            if version != 1:
                raise ValueError('Only version 1 kinematics are supported, not %d' % version)
        
            self.label:str = read_string(kinematicsIn)
            self.description:str = read_string(kinematicsIn)
            self.unit:str = read_string(kinematicsIn)
            count = read_int(kinematicsIn)
            self.values:array[float] = read_float_arr(kinematicsIn, count)
            trial.kinematics[self.label] = self

class Positions:
    def __init__(self, zip_file, fname, trial):
        with zip_file.open(fname, 'r') as kinematicsIn:
            version = read_int(kinematicsIn)
            if version != 1:
                raise ValueError('Only version 1 position data is supported, not %d' % version)

            self.label:str = read_string(kinematicsIn)
            self.description:str = read_string(kinematicsIn)
            count = read_int(kinematicsIn)
            values = read_float_arr(kinematicsIn, count*2)

            self.values:List[Tuple[float, float]] = []
            for n in range(0, count * 2, 2):
                self.values.append((values[n], values[n+1]))

            trial.positions[self.label] = self

class Trial:
    frame_count:int = 0
    kinematics_count:int = 0
    position_count:int = 0
    event_count:int = 0
    ack_count:int = 0
    frame_rate:int = 0
    save_name:str = ''

    def __init__(self, name:str):
        self.name:str = name
        self.events:List[ExamEvent] = []   # A list of exam event objects
        self.ack_times:List[Tuple[float, float]] = [] # a list of tuples where each tuple is (frame send time, frame ack time)
        self.kinematics:Dict[str, Kinematics] = {} # a dictionary of kinematics objects by the name of the data channel
        self.positions:Dict[str, Positions] = {} # a dictionary of position data by the name of the arm
        self.parameters = {} # a dictionary of dictionaries. Group name -> parameter names
        self.kinematics_names:List[str] = []
        self.position_names:List[str] = []

class ExamLoad:
    """
    This class will load a .kinarm file given the path to the file. There is no extra optimization done
    here to read only the parts of a file that you are interested. When the constructor is complete this
    object will contain Trials accessible from

        this.trials

    Each Trial will contain:

    Kinematics
    Positions
    ExamEvents
    parameters
    """

    def __init__(self, srcName: str):
        with ZipFile(srcName) as zip_file:
            self.__zip_contents = zip_file.namelist()
            # The "raw" folder contains all of the trials
            self.__exam_data:List[str] = [f for f in self.__zip_contents if f.startswith("raw")]
            self.trials:Dict[str, Trial] = {}

            # every folder in the "raw" folder is for a trial. Except the "common" folder, that is common parameter data.
            # This bit of code parses apart all of the file names to optain the folder names and therefore the trial names.
            trial_name:Set[str] = set()
            for exam_file in self.__exam_data:
                name_parts = split(exam_file)
                root_name = name_parts[0][4:]  # remove raw/ to get only the trial name
                trial_name.add(root_name)

            for name in trial_name:
                self.trials[name] = Trial(name)

            # Each trial folder will contain:
            # header.bin - general header information about the number of frames in a trial, kinematics names, etc
            # parameters.bin - all of the general parameter, ex. The target table, robot type, calibration params, etc
            #
            # Trials may contain any of the following
            # examevents.bin - If you use the event_codes input on the DataLogging block then this will contain your events by name and timestamp.
            # videoack.bin - This contains the Video frame acknowledgements information.
            # *.kinematics - This file extension indicates a collected channel of data
            # *.position - This file extension indicates the collected positions for an arm.
            for exam_file in self.__exam_data:
                name_parts = split(exam_file)
                root_name = name_parts[0][4:] # remove raw/ to get only the trial name
                sub_name = name_parts[1]

                if sub_name == 'header.bin':
                    self.__read_header(zip_file, exam_file, self.trials[root_name])
                elif sub_name == 'parameters.bin':
                    self.__read_parameters(zip_file, exam_file, self.trials[root_name])
                elif sub_name == 'examevents.bin':
                    self.__read_exam_events(zip_file, exam_file, self.trials[root_name])
                elif sub_name == 'videoack.bin':
                    self.__read_video_ack(zip_file, exam_file, self.trials[root_name])
                elif sub_name.endswith('.kinematics'):
                    Kinematics(zip_file, exam_file, self.trials[root_name])
                elif sub_name.endswith('.position'):
                    Positions(zip_file, exam_file, self.trials[root_name])

    @staticmethod
    def __read_header(zip_file:ZipFile, fname:str, trial:Trial):
        with zip_file.open(fname, 'r') as headerIn:
            # every file starts with a version code so that we can safely change the information stored in the file
            version = read_int(headerIn)
            if version != 3:
                raise ValueError('Only version 3 headers are supported, not %d' % version)

            trial.frame_count = read_int(headerIn)
            trial.kinematics_count = read_int(headerIn)
            trial.position_count = read_int(headerIn)
            trial.event_count = read_int(headerIn)
            trial.ack_count = read_int(headerIn)
            trial.frame_rate = read_float(headerIn)
            trial.save_name = read_string(headerIn)

            trial.kinematics_names = []
            for n in range(trial.kinematics_count):
                trial.kinematics_names.append(read_string(headerIn))

            trial.position_names = []
            for n in range(trial.position_count):
                trial.position_names.append(read_string(headerIn))

    @staticmethod
    def __read_exam_events(zip_file:ZipFile, fname:str, trial:Trial):
        with zip_file.open(fname, 'r') as eventsIn:
            # every file starts with a version code so that we can safely change the information stored in the file
            version = read_int(eventsIn)
            if version != 3:
                raise ValueError('Only version 3 events are supported, not %d' % version)

            count = read_int(eventsIn)

            for n in range(count):
                trial.events.append(ExamEvent(eventsIn))

    @staticmethod
    def __read_video_ack(zip_file:ZipFile, fname:str, trial:Trial):
        with zip_file.open(fname, 'r') as ackIn:
            version = read_int(ackIn)
            # every file starts with a version code so that we can safely change the information stored in the file
            if version != 3:
                raise ValueError('Only version 3 video ACK data is supported, not %d' % version)

            count = read_int(ackIn)

            for n in range(count):
                # read in each send time and ack time for a video frame
                trial.ack_times.append((read_float(ackIn), read_float(ackIn)))

    @staticmethod
    def __read_parameters(zip_file:ZipFile, fname:str, trial:Trial):
        with zip_file.open(fname, 'r') as paramIn:
            version = read_int(paramIn)
            # every file starts with a version code so that we can safely change the information stored in the file
            if version != 3:
                raise ValueError('Only version 3 parameter files are supported, not %d' % version)

            group = ''

            while paramIn.readable():
                rec_type = read_int(paramIn)

                if rec_type == 0:
                    break
                elif rec_type  == 1:
                    group = read_string(paramIn)
                    group_description = read_string(paramIn)
                elif rec_type == 2:
                    label = read_string(paramIn)
                    description = read_string(paramIn)
                    data_type = read_int(paramIn)
                    value_count = read_int(paramIn)

                    values = []

                    if data_type == 0: #String
                        for n in range(value_count):
                            values.append(read_string(paramIn))

                    elif data_type == 1: #byte
                        values = paramIn.read(value_count)

                    elif data_type == 2: # int
                        values = read_int_arr(paramIn, value_count)

                    elif data_type == 3: # float
                        values = read_float_arr(paramIn, value_count)

                    trial.parameters[group + ":" + label] = values


######
# All of the data reading functions below are specialized to handle little-endian data.
#####

def read_int(fin) -> int:
    data = fin.read(4)
    return 0 if len(data) != 4 else unpack('<i', data)[0]

def read_int_arr(fin, count):
    values = array('i')
    values.fromfile(fin, count)
    # for systems where the native byte order is 'big endian' we will need to swap bytes as we read.
    if byteorder == 'big':
        values.byteswap()
    return values

def read_float(fin) -> float:
    data = fin.read(4)
    return 0 if len(data) != 4 else unpack('<f', data)[0]

def read_float_arr(fin, count):
    values = array('f')
    values.fromfile(fin, count)
    # for systems where the native byte order is 'big endian' we will need to swap bytes as we read.
    if byteorder == 'big':
        values.byteswap()
    return values

def read_string(fin) -> str:
    size = read_int(fin)
    return str(fin.read(size * 2), 'utf-16-le')
# ECG Based Authentication

## Description

This project provides a robust authentication method based on existing ECG data which can be found on [Physionet](https://physionet.org). The ECG signals for different users that's attained from the database goes through 6 steps. 

__Step 1: Read ECG Signal__

Get the path to each person and extract the raw data from the database.
<br/><br/>
__Step 2: Filter Signal__

The ECG signal contains noise, so it gets filtered to remove various noises from the ECG signal.
<br/><br/>
__Step 3: Segmentation__

After filtering, we attain a long ECG signal. Depending on which method you use (i.e. window or RR), the signal will be trimmed in a certain area. For window, the ECG signal will be trimmed before and after the r-peak to get the heartbeat. For RR, the ECG signal will be trimmed at the r-peaks for two consecutive hearbeats. 
<br/><br/>
__Step 4: Feature Extraction__

80% of the segmented ECG signal will then go through a feature extraction where we store decomposition signal level, amplitude and time difference of between each fiducial point, and slope and distance of two correspodning fiducial points. 
<br/><br/>
__Step 5: Template Creation__

The templates are created by taking the average of the extracted fiducial points.
<br/><br/>
__Step 6: Authentication__

Steps 1 through 4 are performed similarly during the authentication mode except we're segmenting the last 20% and using that to match with the template generated earlier. We compare the two templates and find a similarity score using euclidean distance. If the similarity score is good the user is authenticated, else access is denied. 

## Database Used

The following list shows which database was used in order to attain the ECG values, and where this data can be found online. The one currently used in this program is the `ECG-ID Database`. 

* [ECG-ID Database](https://physionet.org/content/ecgiddb/1.0.0/)
    * Data can also be found within `ecg_authentication/src/data/ecg-id-database-1.0.0` of the project folder.

* [QT Database](https://physionet.org/content/qtdb/1.0.0/)

* [PTB Diagnostic ECG Database](https://www.physionet.org/content/ptbdb/1.0.0/)

* [Fantasia Database](https://physionet.org/content/fantasia/1.0.0/)

* [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)

* [CEBS Database](https://www.physionet.org/content/cebsdb/1.0.0/)

* [CYBHi](https://zenodo.org/record/2381823#.XvpCSy2z1N1)

## Requirements to Run the Project

* Latest version of Python 3

## Installation

The following list of libraries must be installed onto your machine and can be done by running the following command: 
    
* `pip3 install numpy`
    
* `pip3 install matplotlib`

* `pip3 install PyWavelets`

* `pip3 install scipy`

* `pip3 install filterpy`

* `pip3 install wfdb`

* `pip3 install math`

* `pip3 install glob3`

## Running

To run the program, navigate to the `ecg_authentication` directory and run the following command:

* `python3 src/authentication_framework.py`

If there are any parameters you want to change it will be done in the `authentication_framework.py` file. Here you can change which feature method you would like to use, which segmentation method to use, which database you want to grab data from, and the number of users you want to pass in. Keywords for changing parameters will be denoted below:

* Feature Method:
    * fiducial
    * non-fiducial

* Segmentation Method
    * window
    * RR

* Database (currently only `ecg_id` works)
    * ecg_id_csv
    * qt_db
    * ecg_id
    * ecg_ptb
    * fantasia
    * mitdb
    * cebsdb
    * CYBHi

* Number of Users
    * Integer value (max for using ecg_id is 90)
# Predict-Decline
Dimensionality reduction of vertex-wise CT measures to predict cognitive decline. Continuation of previous work by Nikhil (Bhagwat, Nikhil, Joseph D. Viviano, Aristotle N. Voineskos, M. Mallar Chakravarty, and Alzheimer’s Disease Neuroimaging Initiative. 2018. “Modeling and Prediction of Clinical Symptom Trajectories in Alzheimer’s Disease Using Longitudinal Data.” PLoS Computational Biology 14 (9): e1006376.)

Feature selection methods:
- Automated Anatomical Labeling (baseline)
- Principal Component Analysis
- Recursive Feature Elimination
- Regularized Logistic Regression (Moradi et al., 2015)
- Hierarchical Clustering Analysis

Machine learning classification methods:
- Artificial Neural Network
- Logistic Regression
- Random Forests
- Support Vector Machines
- Longitudinal Siamese Network (Bhagwat et al., 2018)

Set up module dependencies:
```module load anaconda/5.1.0-python3```

List of packages in conda virtual machine:
```
_tflow_select             2.3.0                       mkl  
absl-py                   0.6.1                    py36_0  
astor                     0.7.1                    py36_0  
blas                      1.0                         mkl  
c-ares                    1.15.0               h7b6447c_1  
ca-certificates           2018.03.07                    0  
certifi                   2018.11.29               py36_0  
cycler                    0.10.0                   py36_0  
dbus                      1.13.2               h714fa37_1  
expat                     2.2.6                he6710b0_0  
fontconfig                2.13.0               h9420a91_0  
freetype                  2.9.1                h8a8886c_1  
gast                      0.2.0                    py36_0  
glib                      2.56.2               hd408876_0  
grpcio                    1.16.1           py36hf8bcb03_1  
gst-plugins-base          1.14.0               hbbd80ab_1  
gstreamer                 1.14.0               hb453b48_1  
h5py                      2.8.0            py36h989c5e5_3  
hdf5                      1.10.2               hba1933b_1  
icu                       58.2                 h9c2bf20_1  
intel-openmp              2019.1                      144  
jpeg                      9b                   h024ee3a_2  
keras-applications        1.0.6                    py36_0  
keras-preprocessing       1.0.5                    py36_0  
kiwisolver                1.0.1            py36hf484d3e_0  
libedit                   3.1.20170329         h6b74fdf_2  
libffi                    3.2.1                hd88cf55_4  
libgcc-ng                 8.2.0                hdf63c60_1  
libgfortran-ng            7.3.0                hdf63c60_0  
libpng                    1.6.35               hbc83047_0  
libprotobuf               3.6.1                hd408876_0  
libstdcxx-ng              8.2.0                hdf63c60_1  
libuuid                   1.0.3                h1bed415_2  
libxcb                    1.13                 h1bed415_1  
libxml2                   2.9.8                h26e45fe_1  
markdown                  3.0.1                    py36_0  
matplotlib                3.0.2            py36h5429711_0  
mkl                       2019.1                      144  
mkl_fft                   1.0.6            py36hd81dba3_0  
mkl_random                1.0.2            py36hd81dba3_0  
ncurses                   6.1                  he6710b0_1  
numpy                     1.15.4           py36h7e9f1db_0  
numpy-base                1.15.4           py36hde5b4d6_0  
openssl                   1.1.1a               h7b6447c_0  
pandas                    0.23.4           py36h04863e7_0  
patsy                     0.5.1                    py36_0  
pcre                      8.42                 h439df22_0  
pip                       18.1                     py36_0  
protobuf                  3.6.1            py36he6710b0_0  
pyparsing                 2.3.0                    py36_0  
pyqt                      5.9.2            py36h05f1152_2  
python                    3.6.7                h0371630_0  
python-dateutil           2.7.5                    py36_0  
pytz                      2018.7                   py36_0  
qt                        5.9.7                h5867ecd_1  
readline                  7.0                  h7b6447c_5  
scikit-learn              0.20.1           py36hd81dba3_0  
scipy                     1.1.0            py36h7c811a0_2  
seaborn                   0.9.0                    py36_0  
setuptools                40.6.3                   py36_0  
sip                       4.19.8           py36hf484d3e_0  
six                       1.12.0                   py36_0  
sqlite                    3.26.0               h7b6447c_0  
statsmodels               0.9.0            py36h035aef0_0  
tensorboard               1.12.1           py36he6710b0_0  
tensorflow                1.12.0          mkl_py36h69b6ba0_0  
tensorflow-base           1.12.0          mkl_py36h3c3e929_0  
termcolor                 1.1.0                    py36_1  
tk                        8.6.8                hbc83047_0  
tornado                   5.1.1            py36h7b6447c_0  
werkzeug                  0.14.1                   py36_0  
wheel                     0.32.3                   py36_0  
xz                        5.2.4                h14c3975_4  
zlib                      1.2.11               h7b6447c_3 
```

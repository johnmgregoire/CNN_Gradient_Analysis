# CNN_Gradient_Analysis
Train a convolutional neural network and analyze the gradients in the model to identify relationships in high dimensional spaces.

Code primarily written by Mitustaro Umehara with conttributions from Dan Guevarra and Helge Stein. Algorithm development by Mitustaro Umehara, Dan Guevarra, Helge Stein, and John Gregoire. For use in scientific publications, please cite the journal article describing the algorithm and its performance:
<TBD>



## Instructions for model traning and analysis in Jupyter notebook

Code has been tested on 64-bit Windows with Python 3.6, using a virtual environment created with conda 4.5.11 from the open source Anaconda Distribution available at: https://www.anaconda.com/

1. After Anaconda installation, launch an "Anaconda Prompt" and clone a local copy of the git repository.

    `git clone https://github.com/johnmgregoire/CNN_Gradient_Analysis.git`

2. Run the following command to create a clean environment with the required packages.

    `conda create -n myenv python=3 numpy matplotlib pandas scipy scikit-learn tensorflow keras jupyter`

3. Activate the virtual environment.

    `conda activate myenv`

4. Launch a local (by default) Jupyter server instance to view and run the `code_v5.ipynb` notebook file located in the repository folder.

    `jupyter notebook`



### List of installed package versions in environment
```
# Name                    Version                   Build  Channel
_tflow_select             2.2.0                     eigen  
absl-py                   0.6.1                    py36_0  
astor                     0.7.1                    py36_0  
backcall                  0.1.0                    py36_0  
blas                      1.0                         mkl  
bleach                    3.0.2                    py36_0  
ca-certificates           2018.03.07                    0  
certifi                   2018.11.29               py36_0  
colorama                  0.4.1                    py36_0  
cycler                    0.10.0           py36h009560c_0  
decorator                 4.3.0                    py36_0  
entrypoints               0.2.3                    py36_2  
freetype                  2.9.1                ha9979f8_1  
gast                      0.2.1                    py36_0  
grpcio                    1.16.1           py36h351948d_1  
h5py                      2.9.0            py36h5e291fa_0  
hdf5                      1.10.4               h7ebc959_0  
icc_rt                    2019.0.0             h0cc432a_1  
icu                       58.2                 ha66f8fd_1  
intel-openmp              2019.1                      144  
ipykernel                 5.1.0            py36h39e3cac_0  
ipython                   7.2.0            py36h39e3cac_0  
ipython_genutils          0.2.0            py36h3c5d0ee_0  
ipywidgets                7.4.2                    py36_0  
jedi                      0.13.2                   py36_0  
jinja2                    2.10                     py36_0  
jpeg                      9b                   hb83a4c4_2  
jsonschema                2.6.0            py36h7636477_0  
jupyter                   1.0.0                    py36_7  
jupyter_client            5.2.4                    py36_0  
jupyter_console           6.0.0                    py36_0  
jupyter_core              4.4.0                    py36_0  
keras                     2.2.4                         0  
keras-applications        1.0.6                    py36_0  
keras-base                2.2.4                    py36_0  
keras-preprocessing       1.0.5                    py36_0  
kiwisolver                1.0.1            py36h6538335_0  
libpng                    1.6.36               h2a8f88b_0  
libprotobuf               3.6.1                h7bd577a_0  
libsodium                 1.0.16               h9d3ae62_0  
m2w64-gcc-libgfortran     5.3.0                         6  
m2w64-gcc-libs            5.3.0                         7  
m2w64-gcc-libs-core       5.3.0                         7  
m2w64-gmp                 6.1.0                         2  
m2w64-libwinpthread-git   5.0.0.4634.697f757               2  
markdown                  3.0.1                    py36_0  
markupsafe                1.1.0            py36he774522_0  
matplotlib                3.0.2            py36hc8f65d3_0  
mistune                   0.8.4            py36he774522_0  
mkl                       2019.1                      144  
mkl_fft                   1.0.10           py36h14836fe_0  
mkl_random                1.0.2            py36h343c172_0  
msys2-conda-epoch         20160418                      1  
nbconvert                 5.3.1                    py36_0  
nbformat                  4.4.0            py36h3a5bc1b_0  
notebook                  5.7.4                    py36_0  
numpy                     1.15.4           py36h19fb1c0_0  
numpy-base                1.15.4           py36hc3f5095_0  
openssl                   1.1.1a               he774522_0  
pandas                    0.23.4           py36h830ac7b_0  
pandoc                    2.2.3.2                       0  
pandocfilters             1.4.2                    py36_1  
parso                     0.3.1                    py36_0  
pickleshare               0.7.5                    py36_0  
pip                       18.1                     py36_0  
prometheus_client         0.5.0                    py36_0  
prompt_toolkit            2.0.7                    py36_0  
protobuf                  3.6.1            py36h33f27b4_0  
pygments                  2.3.1                    py36_0  
pyparsing                 2.3.0                    py36_0  
pyqt                      5.9.2            py36h6538335_2  
pyreadline                2.1                      py36_1  
python                    3.6.8                h9f7ef89_0  
python-dateutil           2.7.5                    py36_0  
pytz                      2018.7                   py36_0  
pywinpty                  0.5.5                 py36_1000  
pyyaml                    3.13             py36hfa6e2cd_0  
pyzmq                     17.1.2           py36hfa6e2cd_0  
qt                        5.9.7            vc14h73c81de_0  
qtconsole                 4.4.3                    py36_0  
scikit-learn              0.20.2           py36h343c172_0  
scipy                     1.1.0            py36h29ff71c_2  
send2trash                1.5.0                    py36_0  
setuptools                40.6.3                   py36_0  
sip                       4.19.8           py36h6538335_0  
six                       1.12.0                   py36_0  
sqlite                    3.26.0               he774522_0  
tensorboard               1.12.2           py36h33f27b4_0  
tensorflow                1.12.0          eigen_py36h67ac661_0  
tensorflow-base           1.12.0          eigen_py36h45df0d8_0  
termcolor                 1.1.0                    py36_1  
terminado                 0.8.1                    py36_1  
testpath                  0.4.2                    py36_0  
tornado                   5.1.1            py36hfa6e2cd_0  
traitlets                 4.3.2            py36h096827d_0  
vc                        14.1                 h0510ff6_4  
vs2015_runtime            14.15.26706          h3a45250_0  
wcwidth                   0.1.7            py36h3d5aa90_0  
webencodings              0.5.1                    py36_1  
werkzeug                  0.14.1                   py36_0  
wheel                     0.32.3                   py36_0  
widgetsnbextension        3.4.2                    py36_0  
wincertstore              0.2              py36h7fe50ca_0  
winpty                    0.4.3                         4  
yaml                      0.1.7                hc54c509_2  
zeromq                    4.2.5                he025d50_1  
zlib                      1.2.11               h62dcd97_3  
```
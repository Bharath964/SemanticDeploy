python -m venv .\3.10venv
.\3.10venv\Scripts\Activate
python -m pip install --upgrade pip (only once)
pip install -r requirements.txt
 python training_aerial_imagery.py (optional)
 python 229_prediction_aerial_imagery_using_smooth_blending.py  (to 1st check)
 streamlit run app.py  


pip install efficientnet==1.1.1(if err)



pip install -r requirements.txt
pip install efficientnet==1.1.1(if err)
python training_aerial_imagery.py
Use tensor flow 2.10.0 for best and no errs


absl-py==2.2.2
astunparse==1.6.3
cachetools==5.5.2
certifi==2025.1.31
charset-normalizer==3.4.1
colorama==0.4.6
contourpy==1.3.1
cycler==0.12.1
efficientnet==1.0.0
flatbuffers==25.2.10
fonttools==4.57.0
gast==0.4.0
google-auth==2.38.0
google-auth-oauthlib==0.4.6
google-pasta==0.2.0
grpcio==1.71.0
h5py==3.13.0
idna==3.10
image-classifiers==1.0.0
imageio==2.37.0
joblib==1.4.2
keras==2.10.0
Keras-Applications==1.0.8
Keras-Preprocessing==1.1.2
kiwisolver==1.4.8
lazy_loader==0.4
libclang==18.1.1
Markdown==3.7
MarkupSafe==3.0.2
matplotlib==3.10.1
networkx==3.4.2
numpy==1.26.4
oauthlib==3.2.2
opencv-python==4.11.0.86
opt_einsum==3.4.0
packaging==24.2
patchify==0.2.3
pillow==11.1.0
protobuf==3.19.6
pyasn1==0.6.1
pyasn1_modules==0.4.2
pyparsing==3.2.3
python-dateutil==2.9.0.post0
requests==2.32.3
requests-oauthlib==2.0.0
rsa==4.9
scikit-image==0.25.2
scikit-learn==1.6.1
scipy==1.10.1
segmentation-models==1.0.1
six==1.17.0
tensorboard==2.10.1
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.1
tensorflow==2.10.0
tensorflow-estimator==2.10.0
tensorflow-io-gcs-filesystem==0.31.0
termcolor==3.0.1
threadpoolctl==3.6.0
tifffile==2025.3.30
tqdm==4.67.1
typing_extensions==4.13.1
urllib3==2.3.0
Werkzeug==3.1.3
wrapt==1.17.2



⚡ ~/Semantic git clone https://github.com/Bharath964/Semantic.git
Cloning into 'Semantic'...
remote: Enumerating objects: 183, done.        
⚡ ~/Semantic pip install -r requirements.txt
ERROR: Could not open requirements file: [Errno 2] No such file or directory: 'requirements.txt'
⚡ ~/Semantic 
⚡ ~/Semantic pip install -r requirements.txt
Requirement already satisfied: absl-py==2.1.0 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from -r requirements.txt (line 1)) (2.1.0)
Collecting astunparse==1.6.3 (from -r requirements.txt (line 2))
  Downloading astunparse-1.6.3-py2.py3-none-any.whl.metadata (4.4 kB)
Collecting certifi==2024.8.30 (from -r requirements.txt (line 3))
  Downloading certifi-2024.8.30-py3-none-any.whl.metadata (2.2 kB)
Collecting charset-normalizer==3.4.0 (from -r requirements.txt (line 4))
  Downloading charset_normalizer-3.4.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (34 kB)
Requirement already satisfied: contourpy==1.3.1 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from -r requirements.txt (line 5)) (1.3.1)
Requirement already satisfied: cycler==0.12.1 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from -r requirements.txt (line 6)) (0.12.1)
Collecting flatbuffers==24.3.25 (from -r requirements.txt (line 8))
  Downloading flatbuffers-24.3.25-py2.py3-none-any.whl.metadata (850 bytes)
Collecting fonttools==4.55.0 (from -r requirements.txt (line 9))
  Downloading fonttools-4.55.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (164 kB)
Collecting gast==0.6.0 (from -r requirements.txt (line 10))
  Downloading gast-0.6.0-py3-none-any.whl.metadata (1.3 kB)
Collecting google-pasta==0.2.0 (from -r requirements.txt (line 11))
  Downloading google_pasta-0.2.0-py3-none-any.whl.metadata (814 bytes)
Collecting grpcio==1.68.1 (from -r requirements.txt (line 12))
  Downloading grpcio-1.68.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.9 kB)
Collecting h5py==3.12.1 (from -r requirements.txt (line 13))
  Downloading h5py-3.12.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.5 kB)
Requirement already satisfied: idna==3.10 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from -r requirements.txt (line 14)) (3.10)
Collecting image-classifiers==1.0.0 (from -r requirements.txt (line 15))
  Downloading image_classifiers-1.0.0-py3-none-any.whl.metadata (8.6 kB)
Collecting imageio==2.36.1 (from -r requirements.txt (line 16))
  Downloading imageio-2.36.1-py3-none-any.whl.metadata (5.2 kB)
Requirement already satisfied: joblib==1.4.2 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from -r requirements.txt (line 17)) (1.4.2)
Collecting keras==3.7.0 (from -r requirements.txt (line 18))
  Downloading keras-3.7.0-py3-none-any.whl.metadata (5.8 kB)
Collecting Keras-Applications==1.0.8 (from -r requirements.txt (line 19))
  Downloading Keras_Applications-1.0.8-py3-none-any.whl.metadata (1.7 kB)
Collecting kiwisolver==1.4.7 (from -r requirements.txt (line 20))
  Downloading kiwisolver-1.4.7-cp310-cp310-manylinux_2_12_x86_64.manylinux2010_x86_64.whl.metadata (6.3 kB)
Collecting lazy_loader==0.4 (from -r requirements.txt (line 21))
  Downloading lazy_loader-0.4-py3-none-any.whl.metadata (7.6 kB)
Collecting libclang==18.1.1 (from -r requirements.txt (line 22))
  Downloading libclang-18.1.1-py2.py3-none-manylinux2010_x86_64.whl.metadata (5.2 kB)
Requirement already satisfied: Markdown==3.7 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from -r requirements.txt (line 23)) (3.7)
Requirement already satisfied: markdown-it-py==3.0.0 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from -r requirements.txt (line 24)) (3.0.0)
Requirement already satisfied: MarkupSafe==3.0.2 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from -r requirements.txt (line 25)) (3.0.2)
Collecting matplotlib==3.9.3 (from -r requirements.txt (line 26))
  Downloading matplotlib-3.9.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (11 kB)
Requirement already satisfied: mdurl==0.1.2 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from -r requirements.txt (line 27)) (0.1.2)
Collecting ml-dtypes==0.4.1 (from -r requirements.txt (line 28))
  Downloading ml_dtypes-0.4.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (20 kB)
Collecting namex==0.0.8 (from -r requirements.txt (line 29))
  Downloading namex-0.0.8-py3-none-any.whl.metadata (246 bytes)
Requirement already satisfied: networkx==3.4.2 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from -r requirements.txt (line 30)) (3.4.2)
Requirement already satisfied: numpy==1.26.4 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from -r requirements.txt (line 31)) (1.26.4)
Collecting opencv-python==4.10.0.84 (from -r requirements.txt (line 32))
  Downloading opencv_python-4.10.0.84-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (20 kB)
Collecting opt_einsum==3.4.0 (from -r requirements.txt (line 33))
  Downloading opt_einsum-3.4.0-py3-none-any.whl.metadata (6.3 kB)
Collecting optree==0.13.1 (from -r requirements.txt (line 34))
  Downloading optree-0.13.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (47 kB)
Requirement already satisfied: packaging==24.2 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from -r requirements.txt (line 35)) (24.2)
Collecting patchify==0.2.3 (from -r requirements.txt (line 36))
  Downloading patchify-0.2.3-py3-none-any.whl.metadata (3.0 kB)
Collecting pillow==11.0.0 (from -r requirements.txt (line 37))
  Downloading pillow-11.0.0-cp310-cp310-manylinux_2_28_x86_64.whl.metadata (9.1 kB)
Collecting protobuf==5.29.0 (from -r requirements.txt (line 38))
  Downloading protobuf-5.29.0-cp38-abi3-manylinux2014_x86_64.whl.metadata (592 bytes)
Collecting Pygments==2.18.0 (from -r requirements.txt (line 39))
  Downloading pygments-2.18.0-py3-none-any.whl.metadata (2.5 kB)
Collecting pyparsing==3.2.0 (from -r requirements.txt (line 40))
  Downloading pyparsing-3.2.0-py3-none-any.whl.metadata (5.0 kB)
Requirement already satisfied: python-dateutil==2.9.0.post0 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from -r requirements.txt (line 41)) (2.9.0.post0)
Requirement already satisfied: requests==2.32.3 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from -r requirements.txt (line 42)) (2.32.3)
Requirement already satisfied: rich==13.9.4 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from -r requirements.txt (line 43)) (13.9.4)
Collecting scikit-image==0.24.0 (from -r requirements.txt (line 44))
  Downloading scikit_image-0.24.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (14 kB)
Collecting scikit-learn==1.5.2 (from -r requirements.txt (line 45))
  Downloading scikit_learn-1.5.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (13 kB)
Collecting scipy==1.14.1 (from -r requirements.txt (line 46))
  Downloading scipy-1.14.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (60 kB)
Collecting segmentation-models==1.0.1 (from -r requirements.txt (line 47))
  Downloading segmentation_models-1.0.1-py3-none-any.whl.metadata (938 bytes)
Collecting six==1.16.0 (from -r requirements.txt (line 48))
  Downloading six-1.16.0-py2.py3-none-any.whl.metadata (1.8 kB)
Collecting tensorboard==2.18.0 (from -r requirements.txt (line 49))
  Downloading tensorboard-2.18.0-py3-none-any.whl.metadata (1.6 kB)
Requirement already satisfied: tensorboard-data-server==0.7.2 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from -r requirements.txt (line 50)) (0.7.2)
Collecting termcolor==2.5.0 (from -r requirements.txt (line 54))
  Downloading termcolor-2.5.0-py3-none-any.whl.metadata (6.1 kB)
Collecting threadpoolctl==3.5.0 (from -r requirements.txt (line 55))
  Downloading threadpoolctl-3.5.0-py3-none-any.whl.metadata (13 kB)
Collecting tifffile==2024.9.20 (from -r requirements.txt (line 56))
  Downloading tifffile-2024.9.20-py3-none-any.whl.metadata (32 kB)
Requirement already satisfied: typing_extensions==4.12.2 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from -r requirements.txt (line 57)) (4.12.2)
Collecting urllib3==2.2.3 (from -r requirements.txt (line 58))
  Downloading urllib3-2.2.3-py3-none-any.whl.metadata (6.5 kB)
Requirement already satisfied: Werkzeug==3.1.3 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from -r requirements.txt (line 59)) (3.1.3)
Collecting wrapt==1.17.0 (from -r requirements.txt (line 60))
  Downloading wrapt-1.17.0-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.4 kB)
Requirement already satisfied: wheel<1.0,>=0.23.0 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from astunparse==1.6.3->-r requirements.txt (line 2)) (0.45.1)
Collecting efficientnet==1.0.0 (from segmentation-models==1.0.1->-r requirements.txt (line 47))
  Downloading efficientnet-1.0.0-py3-none-any.whl.metadata (6.1 kB)
Requirement already satisfied: setuptools>=41.0.0 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from tensorboard==2.18.0->-r requirements.txt (line 49)) (75.8.0)
WARNING: The candidate selected for download or install is a yanked version: 'protobuf' candidate (version 5.29.0 at https://files.pythonhosted.org/packages/ee/2e/cc46181ddce0940647d21a8341bf2eddad247a5d030e8c30c7a342793978/protobuf-5.29.0-cp38-abi3-manylinux2014_x86_64.whl (from https://pypi.org/simple/protobuf/) (requires-python:>=3.8))
Reason for being yanked: https://github.com/protocolbuffers/protobuf/issues/19430#issuecomment-2518458119
Downloading astunparse-1.6.3-py2.py3-none-any.whl (12 kB)
Downloading certifi-2024.8.30-py3-none-any.whl (167 kB)
Downloading charset_normalizer-3.4.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (144 kB)
Downloading flatbuffers-24.3.25-py2.py3-none-any.whl (26 kB)
Downloading fonttools-4.55.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.6/4.6 MB 83.1 MB/s eta 0:00:00
Downloading gast-0.6.0-py3-none-any.whl (21 kB)
Downloading google_pasta-0.2.0-py3-none-any.whl (57 kB)
Downloading grpcio-1.68.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (5.9 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.9/5.9 MB 137.1 MB/s eta 0:00:00
Downloading h5py-3.12.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (5.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.3/5.3 MB 97.5 MB/s eta 0:00:00
Downloading image_classifiers-1.0.0-py3-none-any.whl (19 kB)
Downloading imageio-2.36.1-py3-none-any.whl (315 kB)
Downloading keras-3.7.0-py3-none-any.whl (1.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 36.8 MB/s eta 0:00:00
Downloading Keras_Applications-1.0.8-py3-none-any.whl (50 kB)
Downloading kiwisolver-1.4.7-cp310-cp310-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (1.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.6/1.6 MB 74.3 MB/s eta 0:00:00
Downloading lazy_loader-0.4-py3-none-any.whl (12 kB)
Downloading libclang-18.1.1-py2.py3-none-manylinux2010_x86_64.whl (24.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 24.5/24.5 MB 218.0 MB/s eta 0:00:00
Downloading matplotlib-3.9.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (8.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.3/8.3 MB 143.2 MB/s eta 0:00:00
Downloading ml_dtypes-0.4.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.2/2.2 MB 91.5 MB/s eta 0:00:00
Downloading namex-0.0.8-py3-none-any.whl (5.8 kB)
Downloading opencv_python-4.10.0.84-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (62.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 62.5/62.5 MB 138.0 MB/s eta 0:00:00
Downloading opt_einsum-3.4.0-py3-none-any.whl (71 kB)
Downloading optree-0.13.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (381 kB)
Downloading patchify-0.2.3-py3-none-any.whl (6.6 kB)
Downloading pillow-11.0.0-cp310-cp310-manylinux_2_28_x86_64.whl (4.4 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.4/4.4 MB 97.9 MB/s eta 0:00:00
Downloading protobuf-5.29.0-cp38-abi3-manylinux2014_x86_64.whl (319 kB)
Downloading pygments-2.18.0-py3-none-any.whl (1.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 61.6 MB/s eta 0:00:00
Downloading pyparsing-3.2.0-py3-none-any.whl (106 kB)
Downloading scikit_image-0.24.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (14.9 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 14.9/14.9 MB 151.1 MB/s eta 0:00:00
Downloading scikit_learn-1.5.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (13.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 13.3/13.3 MB 130.2 MB/s eta 0:00:00
Downloading scipy-1.14.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (41.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 41.2/41.2 MB 148.5 MB/s eta 0:00:00
Downloading segmentation_models-1.0.1-py3-none-any.whl (33 kB)
Downloading six-1.16.0-py2.py3-none-any.whl (11 kB)
Downloading tensorboard-2.18.0-py3-none-any.whl (5.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.5/5.5 MB 123.3 MB/s eta 0:00:00
Downloading termcolor-2.5.0-py3-none-any.whl (7.8 kB)
Downloading threadpoolctl-3.5.0-py3-none-any.whl (18 kB)
Downloading tifffile-2024.9.20-py3-none-any.whl (228 kB)
Downloading urllib3-2.2.3-py3-none-any.whl (126 kB)
Downloading wrapt-1.17.0-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (82 kB)
Downloading efficientnet-1.0.0-py3-none-any.whl (17 kB)
Installing collected packages: namex, libclang, flatbuffers, wrapt, urllib3, tifffile, threadpoolctl, termcolor, six, scipy, pyparsing, Pygments, protobuf, pillow, patchify, optree, opt_einsum, opencv-python, ml-dtypes, lazy_loader, kiwisolver, h5py, grpcio, gast, fonttools, charset-normalizer, certifi, tensorboard, scikit-learn, Keras-Applications, imageio, google-pasta, astunparse, scikit-image, matplotlib, keras, image-classifiers, efficientnet, segmentation-models
  Attempting uninstall: urllib3
    Found existing installation: urllib3 2.3.0
    Uninstalling urllib3-2.3.0:
      Successfully uninstalled urllib3-2.3.0
  Attempting uninstall: threadpoolctl
    Found existing installation: threadpoolctl 3.6.0
    Uninstalling threadpoolctl-3.6.0:
      Successfully uninstalled threadpoolctl-3.6.0
  Attempting uninstall: six
    Found existing installation: six 1.17.0
    Uninstalling six-1.17.0:
      Successfully uninstalled six-1.17.0
  Attempting uninstall: scipy
    Found existing installation: scipy 1.11.4
    Uninstalling scipy-1.11.4:
      Successfully uninstalled scipy-1.11.4
  Attempting uninstall: pyparsing
    Found existing installation: pyparsing 3.2.1
    Uninstalling pyparsing-3.2.1:
      Successfully uninstalled pyparsing-3.2.1
  Attempting uninstall: Pygments
    Found existing installation: Pygments 2.19.1
    Uninstalling Pygments-2.19.1:
      Successfully uninstalled Pygments-2.19.1
  Attempting uninstall: protobuf
    Found existing installation: protobuf 4.23.4
    Uninstalling protobuf-4.23.4:
      Successfully uninstalled protobuf-4.23.4
  Attempting uninstall: pillow
    Found existing installation: pillow 11.1.0
    Uninstalling pillow-11.1.0:
      Successfully uninstalled pillow-11.1.0
  Attempting uninstall: kiwisolver
    Found existing installation: kiwisolver 1.4.8
    Uninstalling kiwisolver-1.4.8:
      Successfully uninstalled kiwisolver-1.4.8
  Attempting uninstall: grpcio
    Found existing installation: grpcio 1.71.0
    Uninstalling grpcio-1.71.0:
      Successfully uninstalled grpcio-1.71.0
  Attempting uninstall: fonttools
    Found existing installation: fonttools 4.56.0
    Uninstalling fonttools-4.56.0:
      Successfully uninstalled fonttools-4.56.0
  Attempting uninstall: charset-normalizer
    Found existing installation: charset-normalizer 3.4.1
    Uninstalling charset-normalizer-3.4.1:
      Successfully uninstalled charset-normalizer-3.4.1
  Attempting uninstall: certifi
    Found existing installation: certifi 2025.1.31
    Uninstalling certifi-2025.1.31:
      Successfully uninstalled certifi-2025.1.31
  Attempting uninstall: tensorboard
    Found existing installation: tensorboard 2.15.1
    Uninstalling tensorboard-2.15.1:
      Successfully uninstalled tensorboard-2.15.1
  Attempting uninstall: scikit-learn
    Found existing installation: scikit-learn 1.3.2
    Uninstalling scikit-learn-1.3.2:
      Successfully uninstalled scikit-learn-1.3.2
  Attempting uninstall: matplotlib
    Found existing installation: matplotlib 3.8.2
    Uninstalling matplotlib-3.8.2:
      Successfully uninstalled matplotlib-3.8.2
Successfully installed Keras-Applications-1.0.8 Pygments-2.18.0 astunparse-1.6.3 certifi-2024.8.30 charset-normalizer-3.4.0 efficientnet-1.0.0 flatbuffers-24.3.25 fonttools-4.55.0 gast-0.6.0 google-pasta-0.2.0 grpcio-1.68.1 h5py-3.12.1 image-classifiers-1.0.0 imageio-2.36.1 keras-3.7.0 kiwisolver-1.4.7 lazy_loader-0.4 libclang-18.1.1 matplotlib-3.9.3 ml-dtypes-0.4.1 namex-0.0.8 opencv-python-4.10.0.84 opt_einsum-3.4.0 optree-0.13.1 patchify-0.2.3 pillow-11.0.0 protobuf-5.29.0 pyparsing-3.2.0 scikit-image-0.24.0 scikit-learn-1.5.2 scipy-1.14.1 segmentation-models-1.0.1 six-1.16.0 tensorboard-2.18.0 termcolor-2.5.0 threadpoolctl-3.5.0 tifffile-2024.9.20 urllib3-2.2.3 wrapt-1.17.0
⚡ ~/Semantic pip uninstall segmentation-models efficientnet keras-applications image-classifiers -y
egmentation-models-tf==1.0.2Found existing installation: segmentation-models 1.0.1
Uninstalling segmentation-models-1.0.1:
  Successfully uninstalled segmentation-models-1.0.1
Found existing installation: efficientnet 1.0.0
Uninstalling efficientnet-1.0.0:
  Successfully uninstalled efficientnet-1.0.0
Found existing installation: Keras-Applications 1.0.8
Uninstalling Keras-Applications-1.0.8:
  Successfully uninstalled Keras-Applications-1.0.8
Found existing installation: image-classifiers 1.0.0
Uninstalling image-classifiers-1.0.0:
  Successfully uninstalled image-classifiers-1.0.0
⚡ ~/Semantic pip install segmentation-models-tf==1.0.2
ERROR: Could not find a version that satisfies the requirement segmentation-models-tf==1.0.2 (from versions: none)
ERROR: No matching distribution found for segmentation-models-tf==1.0.2
⚡ ~/Semantic pip uninstall keras                      
Found existing installation: keras 3.7.0
Uninstalling keras-3.7.0:
  Would remove:
    /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages/keras-3.7.0.dist-info/*
    /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages/keras/*
Proceed (Y/n)? y
Your response ('pip install tensorflow==2.9.1y') was not one of the expected responses: y, n, 
Proceed (Y/n)? y
  Successfully uninstalled keras-3.7.0
⚡ ~/Semantic pip install tensorflow==2.10.0
Collecting tensorflow==2.10.0
  Downloading tensorflow-2.10.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.1 kB)
Requirement already satisfied: absl-py>=1.0.0 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from tensorflow==2.10.0) (2.1.0)
Requirement already satisfied: astunparse>=1.6.0 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from tensorflow==2.10.0) (1.6.3)
Requirement already satisfied: flatbuffers>=2.0 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from tensorflow==2.10.0) (24.3.25)
Collecting gast<=0.4.0,>=0.2.1 (from tensorflow==2.10.0)
  Downloading gast-0.4.0-py3-none-any.whl.metadata (1.1 kB)
Requirement already satisfied: google-pasta>=0.1.1 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from tensorflow==2.10.0) (0.2.0)
Requirement already satisfied: grpcio<2.0,>=1.24.3 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from tensorflow==2.10.0) (1.68.1)
Requirement already satisfied: h5py>=2.9.0 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from tensorflow==2.10.0) (3.12.1)
Collecting keras<2.11,>=2.10.0 (from tensorflow==2.10.0)
  Downloading keras-2.10.0-py2.py3-none-any.whl.metadata (1.3 kB)
Collecting keras-preprocessing>=1.1.1 (from tensorflow==2.10.0)
  Downloading Keras_Preprocessing-1.1.2-py2.py3-none-any.whl.metadata (1.9 kB)
Requirement already satisfied: libclang>=13.0.0 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from tensorflow==2.10.0) (18.1.1)
Requirement already satisfied: numpy>=1.20 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from tensorflow==2.10.0) (1.26.4)
Requirement already satisfied: opt-einsum>=2.3.2 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from tensorflow==2.10.0) (3.4.0)
Requirement already satisfied: packaging in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from tensorflow==2.10.0) (24.2)
Collecting protobuf<3.20,>=3.9.2 (from tensorflow==2.10.0)
  Downloading protobuf-3.19.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (787 bytes)
Requirement already satisfied: setuptools in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from tensorflow==2.10.0) (75.8.0)
Requirement already satisfied: six>=1.12.0 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from tensorflow==2.10.0) (1.16.0)
Collecting tensorboard<2.11,>=2.10 (from tensorflow==2.10.0)
  Downloading tensorboard-2.10.1-py3-none-any.whl.metadata (1.9 kB)
Collecting tensorflow-io-gcs-filesystem>=0.23.1 (from tensorflow==2.10.0)
  Downloading tensorflow_io_gcs_filesystem-0.37.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (14 kB)
Collecting tensorflow-estimator<2.11,>=2.10.0 (from tensorflow==2.10.0)
  Downloading tensorflow_estimator-2.10.0-py2.py3-none-any.whl.metadata (1.3 kB)
Requirement already satisfied: termcolor>=1.1.0 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from tensorflow==2.10.0) (2.5.0)
Requirement already satisfied: typing-extensions>=3.6.6 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from tensorflow==2.10.0) (4.12.2)
Requirement already satisfied: wrapt>=1.11.0 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from tensorflow==2.10.0) (1.17.0)
Requirement already satisfied: wheel<1.0,>=0.23.0 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from astunparse>=1.6.0->tensorflow==2.10.0) (0.45.1)
Requirement already satisfied: google-auth<3,>=1.6.3 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from tensorboard<2.11,>=2.10->tensorflow==2.10.0) (2.38.0)
Collecting google-auth-oauthlib<0.5,>=0.4.1 (from tensorboard<2.11,>=2.10->tensorflow==2.10.0)
  Downloading google_auth_oauthlib-0.4.6-py2.py3-none-any.whl.metadata (2.7 kB)
Requirement already satisfied: markdown>=2.6.8 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from tensorboard<2.11,>=2.10->tensorflow==2.10.0) (3.7)
Requirement already satisfied: requests<3,>=2.21.0 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from tensorboard<2.11,>=2.10->tensorflow==2.10.0) (2.32.3)
Collecting tensorboard-data-server<0.7.0,>=0.6.0 (from tensorboard<2.11,>=2.10->tensorflow==2.10.0)
  Downloading tensorboard_data_server-0.6.1-py3-none-manylinux2010_x86_64.whl.metadata (1.1 kB)
Collecting tensorboard-plugin-wit>=1.6.0 (from tensorboard<2.11,>=2.10->tensorflow==2.10.0)
  Downloading tensorboard_plugin_wit-1.8.1-py3-none-any.whl.metadata (873 bytes)
Requirement already satisfied: werkzeug>=1.0.1 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from tensorboard<2.11,>=2.10->tensorflow==2.10.0) (3.1.3)
Requirement already satisfied: cachetools<6.0,>=2.0.0 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from google-auth<3,>=1.6.3->tensorboard<2.11,>=2.10->tensorflow==2.10.0) (5.5.2)
Requirement already satisfied: pyasn1-modules>=0.2.1 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from google-auth<3,>=1.6.3->tensorboard<2.11,>=2.10->tensorflow==2.10.0) (0.4.1)
Requirement already satisfied: rsa<5,>=3.1.4 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from google-auth<3,>=1.6.3->tensorboard<2.11,>=2.10->tensorflow==2.10.0) (4.9)
Requirement already satisfied: requests-oauthlib>=0.7.0 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.11,>=2.10->tensorflow==2.10.0) (2.0.0)
Requirement already satisfied: charset-normalizer<4,>=2 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorboard<2.11,>=2.10->tensorflow==2.10.0) (3.4.0)
Requirement already satisfied: idna<4,>=2.5 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorboard<2.11,>=2.10->tensorflow==2.10.0) (3.10)
Requirement already satisfied: urllib3<3,>=1.21.1 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorboard<2.11,>=2.10->tensorflow==2.10.0) (2.2.3)
Requirement already satisfied: certifi>=2017.4.17 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorboard<2.11,>=2.10->tensorflow==2.10.0) (2024.8.30)
Requirement already satisfied: MarkupSafe>=2.1.1 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from werkzeug>=1.0.1->tensorboard<2.11,>=2.10->tensorflow==2.10.0) (3.0.2)
Requirement already satisfied: pyasn1<0.7.0,>=0.4.6 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.11,>=2.10->tensorflow==2.10.0) (0.6.1)
Requirement already satisfied: oauthlib>=3.0.0 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.11,>=2.10->tensorflow==2.10.0) (3.2.2)
Downloading tensorflow-2.10.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (578.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 578.0/578.0 MB 102.9 MB/s eta 0:00:00
Downloading gast-0.4.0-py3-none-any.whl (9.8 kB)
Downloading keras-2.10.0-py2.py3-none-any.whl (1.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.7/1.7 MB 87.7 MB/s eta 0:00:00
Downloading Keras_Preprocessing-1.1.2-py2.py3-none-any.whl (42 kB)
Downloading protobuf-3.19.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.1/1.1 MB 57.1 MB/s eta 0:00:00
Downloading tensorboard-2.10.1-py3-none-any.whl (5.9 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.9/5.9 MB 88.1 MB/s eta 0:00:00
Downloading tensorflow_estimator-2.10.0-py2.py3-none-any.whl (438 kB)
Downloading tensorflow_io_gcs_filesystem-0.37.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (5.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.1/5.1 MB 187.9 MB/s eta 0:00:00
Downloading google_auth_oauthlib-0.4.6-py2.py3-none-any.whl (18 kB)
Downloading tensorboard_data_server-0.6.1-py3-none-manylinux2010_x86_64.whl (4.9 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.9/4.9 MB 82.7 MB/s eta 0:00:00
Downloading tensorboard_plugin_wit-1.8.1-py3-none-any.whl (781 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 781.3/781.3 kB 37.6 MB/s eta 0:00:00
Installing collected packages: tensorboard-plugin-wit, keras, tensorflow-io-gcs-filesystem, tensorflow-estimator, tensorboard-data-server, protobuf, keras-preprocessing, gast, google-auth-oauthlib, tensorboard, tensorflow
  Attempting uninstall: tensorboard-data-server
    Found existing installation: tensorboard-data-server 0.7.2
    Uninstalling tensorboard-data-server-0.7.2:
      Successfully uninstalled tensorboard-data-server-0.7.2
  Attempting uninstall: protobuf
    Found existing installation: protobuf 5.29.0
    Uninstalling protobuf-5.29.0:
      Successfully uninstalled protobuf-5.29.0
  Attempting uninstall: gast
    Found existing installation: gast 0.6.0
    Uninstalling gast-0.6.0:
      Successfully uninstalled gast-0.6.0
  Attempting uninstall: google-auth-oauthlib
    Found existing installation: google-auth-oauthlib 1.2.1
    Uninstalling google-auth-oauthlib-1.2.1:
      Successfully uninstalled google-auth-oauthlib-1.2.1
  Attempting uninstall: tensorboard
    Found existing installation: tensorboard 2.18.0
    Uninstalling tensorboard-2.18.0:
      Successfully uninstalled tensorboard-2.18.0
Successfully installed gast-0.4.0 google-auth-oauthlib-0.4.6 keras-2.10.0 keras-preprocessing-1.1.2 protobuf-3.19.6 tensorboard-2.10.1 tensorboard-data-server-0.6.1 tensorboard-plugin-wit-1.8.1 tensorflow-2.10.0 tensorflow-estimator-2.10.0 tensorflow-io-gcs-filesystem-0.37.1
⚡ ~/Semantic ^V⚡ ~
zsh: command not found: ⚡
⚡ ~/Semantic ^Mpip uninstall segmentation-models -y
pipgit+https://github.com/qubvel/segmentation_modezsh: command not found: 
ls.git                                                                                                                                  
⚡ ~/Semantic pip install -U git+https://github.com/qubvel/segmentation_models.git
Collecting git+https://github.com/qubvel/segmentation_models.git
  Cloning https://github.com/qubvel/segmentation_models.git to /tmp/pip-req-build-ixycjkq9
  Running command git clone --filter=blob:none --quiet https://github.com/qubvel/segmentation_models.git /tmp/pip-req-build-ixycjkq9

  Resolved https://github.com/qubvel/segmentation_models.git to commit 5d24bbfb28af6134e25e2c0b79e7727f6c0491d0
  Running command git submodule update --init --recursive -q
  Preparing metadata (setup.py) ... done
Collecting keras_applications<=1.0.8,>=1.0.7 (from segmentation_models==1.0.1)
  Using cached Keras_Applications-1.0.8-py3-none-any.whl.metadata (1.7 kB)
Collecting image-classifiers==1.0.0 (from segmentation_models==1.0.1)
  Using cached image_classifiers-1.0.0-py3-none-any.whl.metadata (8.6 kB)
Collecting efficientnet==1.1.1 (from segmentation_models==1.0.1)
  Downloading efficientnet-1.1.1-py3-none-any.whl.metadata (6.4 kB)
Requirement already satisfied: scikit-image in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from efficientnet==1.1.1->segmentation_models==1.0.1) (0.24.0)
Requirement already satisfied: numpy>=1.9.1 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from keras_applications<=1.0.8,>=1.0.7->segmentation_models==1.0.1) (1.26.4)
Requirement already satisfied: h5py in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from keras_applications<=1.0.8,>=1.0.7->segmentation_models==1.0.1) (3.12.1)
Requirement already satisfied: scipy>=1.9 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from scikit-image->efficientnet==1.1.1->segmentation_models==1.0.1) (1.14.1)
Requirement already satisfied: networkx>=2.8 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from scikit-image->efficientnet==1.1.1->segmentation_models==1.0.1) (3.4.2)
Requirement already satisfied: pillow>=9.1 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from scikit-image->efficientnet==1.1.1->segmentation_models==1.0.1) (11.0.0)
Requirement already satisfied: imageio>=2.33 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from scikit-image->efficientnet==1.1.1->segmentation_models==1.0.1) (2.36.1)
Requirement already satisfied: tifffile>=2022.8.12 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from scikit-image->efficientnet==1.1.1->segmentation_models==1.0.1) (2024.9.20)
Requirement already satisfied: packaging>=21 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from scikit-image->efficientnet==1.1.1->segmentation_models==1.0.1) (24.2)
Requirement already satisfied: lazy-loader>=0.4 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from scikit-image->efficientnet==1.1.1->segmentation_models==1.0.1) (0.4)
Downloading efficientnet-1.1.1-py3-none-any.whl (18 kB)
Using cached image_classifiers-1.0.0-py3-none-any.whl (19 kB)
Using cached Keras_Applications-1.0.8-py3-none-any.whl (50 kB)
Building wheels for collected packages: segmentation_models
  Building wheel for segmentation_models (setup.py) ... done
  Created wheel for segmentation_models: filename=segmentation_models-1.0.1-py3-none-any.whl size=33854 sha256=3e09564002763fe229b895a778b716c0617d1ab285260db358a66a40b9829097
  Stored in directory: /tmp/pip-ephem-wheel-cache-wc7j0afc/wheels/c3/d2/cc/c5ab5def97531c2b2c36b2bf00be51a18a031015c747fcce5d
Successfully built segmentation_models
Installing collected packages: keras_applications, image-classifiers, efficientnet, segmentation_models
Successfully installed efficientnet-1.1.1 image-classifiers-1.0.0 keras_applications-1.0.8 segmentation_models-1.0.1
⚡ ~/Semantic 
⚡ ~/Semantic pip uninstall segmentation-models -y
ls.git
Found existing installation: segmentation_models 1.0.1
Uninstalling segmentation_models-1.0.1:
  Successfully uninstalled segmentation_models-1.0.1
⚡ ~/Semantic pip install -U git+https://github.com/qubvel/segmentation_models.git
Collecting git+https://github.com/qubvel/segmentation_models.git
  Cloning https://github.com/qubvel/segmentation_models.git to /tmp/pip-req-build-7vh7gx18
  Running command git clone --filter=blob:none --quiet https://github.com/qubvel/segmentation_models.git /tmp/pip-req-build-7vh7gx18

  Resolved https://github.com/qubvel/segmentation_models.git to commit 5d24bbfb28af6134e25e2c0b79e7727f6c0491d0
  Running command git submodule update --init --recursive -q
  Preparing metadata (setup.py) ... done
Requirement already satisfied: keras_applications<=1.0.8,>=1.0.7 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from segmentation_models==1.0.1) (1.0.8)
Requirement already satisfied: image-classifiers==1.0.0 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from segmentation_models==1.0.1) (1.0.0)
Requirement already satisfied: efficientnet==1.1.1 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from segmentation_models==1.0.1) (1.1.1)
Requirement already satisfied: scikit-image in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from efficientnet==1.1.1->segmentation_models==1.0.1) (0.24.0)
Requirement already satisfied: numpy>=1.9.1 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from keras_applications<=1.0.8,>=1.0.7->segmentation_models==1.0.1) (1.26.4)
Requirement already satisfied: h5py in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from keras_applications<=1.0.8,>=1.0.7->segmentation_models==1.0.1) (3.12.1)
Requirement already satisfied: scipy>=1.9 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from scikit-image->efficientnet==1.1.1->segmentation_models==1.0.1) (1.14.1)
Requirement already satisfied: networkx>=2.8 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from scikit-image->efficientnet==1.1.1->segmentation_models==1.0.1) (3.4.2)
Requirement already satisfied: pillow>=9.1 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from scikit-image->efficientnet==1.1.1->segmentation_models==1.0.1) (11.0.0)
Requirement already satisfied: imageio>=2.33 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from scikit-image->efficientnet==1.1.1->segmentation_models==1.0.1) (2.36.1)
Requirement already satisfied: tifffile>=2022.8.12 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from scikit-image->efficientnet==1.1.1->segmentation_models==1.0.1) (2024.9.20)
Requirement already satisfied: packaging>=21 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from scikit-image->efficientnet==1.1.1->segmentation_models==1.0.1) (24.2)
Requirement already satisfied: lazy-loader>=0.4 in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (from scikit-image->efficientnet==1.1.1->segmentation_models==1.0.1) (0.4)
Building wheels for collected packages: segmentation_models
  Building wheel for segmentation_models (setup.py) ... done
  Created wheel for segmentation_models: filename=segmentation_models-1.0.1-py3-none-any.whl size=33854 sha256=b9d7b089a23e9490a1127aeb18949dbb78dec7604ebefa89d62b6a20e4767f7c
  Stored in directory: /tmp/pip-ephem-wheel-cache-fkmsj6uv/wheels/c3/d2/cc/c5ab5def97531c2b2c36b2bf00be51a18a031015c747fcce5d
Successfully built segmentation_models
Installing collected packages: segmentation_models
Successfully installed segmentation_models-1.0.1
⚡ ~/Semantic python training_aerial_imagery.py
  File "/teamspace/studios/this_studio/Semantic/training_aerial_imagery.py", line 529
    pip uninstall segmentation-models -y
        ^^^^^^^^^
SyntaxError: invalid syntax
⚡ ~/Semantic python training_aerial_imagery.py
Model save directory: /teamspace/studios/this_studio/Semantic/models
2025-04-09 19:54:17.674425: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F AVX512_VNNI AVX512_BF16 AVX_VNNI AMX_TILE AMX_INT8 AMX_BF16 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-04-09 19:54:17.909868: I tensorflow/core/util/util.cc:169] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-04-09 19:54:17.936088: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages/cv2/../../lib64:
2025-04-09 19:54:17.936125: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
2025-04-09 19:54:17.967962: E tensorflow/stream_executor/cuda/cuda_blas.cc:2981] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2025-04-09 19:54:18.615830: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer.so.7'; dlerror: libnvinfer.so.7: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages/cv2/../../lib64:
2025-04-09 19:54:18.616087: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer_plugin.so.7'; dlerror: libnvinfer_plugin.so.7: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages/cv2/../../lib64:
2025-04-09 19:54:18.616103: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Cannot dlopen some TensorRT libraries. If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly.
Segmentation Models: using `keras` framework.
Now patchifying image: Semantic segmentation dataset/Tile 8/images/image_part_006.jpg
Now patchifying image: Semantic segmentation dataset/Tile 8/images/image_part_005.jpg
Now patchifying image: Semantic segmentation dataset/Tile 8/images/image_part_001.jpg
Now patchifying image: Semantic segmentation dataset/Tile 8/images/image_part_003.jpg
Now patchifying image: Semantic segmentation dataset/Tile 8/images/image_part_007.jpg
Now patchifying image: Semantic segmentation dataset/Tile 8/images/image_part_002.jpg
Now patchifying image: Semantic segmentation dataset/Tile 8/images/image_part_004.jpg
Now patchifying image: Semantic segmentation dataset/Tile 8/images/image_part_009.jpg
Now patchifying image: Semantic segmentation dataset/Tile 8/images/image_part_008.jpg
Now patchifying image: Semantic segmentation dataset/Tile 7/images/image_part_006.jpg
Now patchifying image: Semantic segmentation dataset/Tile 7/images/image_part_005.jpg
Now patchifying image: Semantic segmentation dataset/Tile 7/images/image_part_001.jpg
Now patchifying image: Semantic segmentation dataset/Tile 7/images/image_part_003.jpg
Now patchifying image: Semantic segmentation dataset/Tile 7/images/image_part_007.jpg
Now patchifying image: Semantic segmentation dataset/Tile 7/images/image_part_002.jpg
Now patchifying image: Semantic segmentation dataset/Tile 7/images/image_part_004.jpg
Now patchifying image: Semantic segmentation dataset/Tile 7/images/image_part_009.jpg
Now patchifying image: Semantic segmentation dataset/Tile 7/images/image_part_008.jpg
Now patchifying image: Semantic segmentation dataset/Tile 1/images/image_part_006.jpg
Now patchifying image: Semantic segmentation dataset/Tile 1/images/image_part_005.jpg
Now patchifying image: Semantic segmentation dataset/Tile 1/images/image_part_001.jpg
Now patchifying image: Semantic segmentation dataset/Tile 1/images/image_part_003.jpg
Now patchifying image: Semantic segmentation dataset/Tile 1/images/image_part_007.jpg
Now patchifying image: Semantic segmentation dataset/Tile 1/images/image_part_002.jpg
Now patchifying image: Semantic segmentation dataset/Tile 1/images/image_part_004.jpg
Now patchifying image: Semantic segmentation dataset/Tile 1/images/image_part_009.jpg
Now patchifying image: Semantic segmentation dataset/Tile 1/images/image_part_008.jpg
Now patchifying image: Semantic segmentation dataset/Tile 4/images/image_part_006.jpg
Now patchifying image: Semantic segmentation dataset/Tile 4/images/image_part_005.jpg
Now patchifying image: Semantic segmentation dataset/Tile 4/images/image_part_001.jpg
Now patchifying image: Semantic segmentation dataset/Tile 4/images/image_part_003.jpg
Now patchifying image: Semantic segmentation dataset/Tile 4/images/image_part_007.jpg
Now patchifying image: Semantic segmentation dataset/Tile 4/images/image_part_002.jpg
Now patchifying image: Semantic segmentation dataset/Tile 4/images/image_part_004.jpg
Now patchifying image: Semantic segmentation dataset/Tile 4/images/image_part_009.jpg
Now patchifying image: Semantic segmentation dataset/Tile 4/images/image_part_008.jpg
Now patchifying image: Semantic segmentation dataset/Tile 2/images/image_part_006.jpg
Now patchifying image: Semantic segmentation dataset/Tile 2/images/image_part_005.jpg
Now patchifying image: Semantic segmentation dataset/Tile 2/images/image_part_001.jpg
Now patchifying image: Semantic segmentation dataset/Tile 2/images/image_part_003.jpg
Now patchifying image: Semantic segmentation dataset/Tile 2/images/image_part_007.jpg
Now patchifying image: Semantic segmentation dataset/Tile 2/images/image_part_002.jpg
Now patchifying image: Semantic segmentation dataset/Tile 2/images/image_part_004.jpg
Now patchifying image: Semantic segmentation dataset/Tile 2/images/image_part_009.jpg
Now patchifying image: Semantic segmentation dataset/Tile 2/images/image_part_008.jpg
Now patchifying image: Semantic segmentation dataset/Tile 3/images/image_part_006.jpg
Now patchifying image: Semantic segmentation dataset/Tile 3/images/image_part_005.jpg
Now patchifying image: Semantic segmentation dataset/Tile 3/images/image_part_001.jpg
Now patchifying image: Semantic segmentation dataset/Tile 3/images/image_part_003.jpg
Now patchifying image: Semantic segmentation dataset/Tile 3/images/image_part_007.jpg
Now patchifying image: Semantic segmentation dataset/Tile 3/images/image_part_002.jpg
Now patchifying image: Semantic segmentation dataset/Tile 3/images/image_part_004.jpg
Now patchifying image: Semantic segmentation dataset/Tile 3/images/image_part_009.jpg
Now patchifying image: Semantic segmentation dataset/Tile 3/images/image_part_008.jpg
Now patchifying image: Semantic segmentation dataset/Tile 5/images/image_part_006.jpg
Now patchifying image: Semantic segmentation dataset/Tile 5/images/image_part_005.jpg
Now patchifying image: Semantic segmentation dataset/Tile 5/images/image_part_001.jpg
Now patchifying image: Semantic segmentation dataset/Tile 5/images/image_part_003.jpg
Now patchifying image: Semantic segmentation dataset/Tile 5/images/image_part_007.jpg
Now patchifying image: Semantic segmentation dataset/Tile 5/images/image_part_002.jpg
Now patchifying image: Semantic segmentation dataset/Tile 5/images/image_part_004.jpg
Now patchifying image: Semantic segmentation dataset/Tile 5/images/image_part_009.jpg
Now patchifying image: Semantic segmentation dataset/Tile 5/images/image_part_008.jpg
Now patchifying image: Semantic segmentation dataset/Tile 6/images/image_part_006.jpg
Now patchifying image: Semantic segmentation dataset/Tile 6/images/image_part_005.jpg
Now patchifying image: Semantic segmentation dataset/Tile 6/images/image_part_001.jpg
Now patchifying image: Semantic segmentation dataset/Tile 6/images/image_part_003.jpg
Now patchifying image: Semantic segmentation dataset/Tile 6/images/image_part_007.jpg
Now patchifying image: Semantic segmentation dataset/Tile 6/images/image_part_002.jpg
Now patchifying image: Semantic segmentation dataset/Tile 6/images/image_part_004.jpg
Now patchifying image: Semantic segmentation dataset/Tile 6/images/image_part_009.jpg
Now patchifying image: Semantic segmentation dataset/Tile 6/images/image_part_008.jpg
Now patchifying mask: Semantic segmentation dataset/Tile 8/masks/image_part_005.png
Now patchifying mask: Semantic segmentation dataset/Tile 8/masks/image_part_009.png
Now patchifying mask: Semantic segmentation dataset/Tile 8/masks/image_part_008.png
Now patchifying mask: Semantic segmentation dataset/Tile 8/masks/image_part_006.png
Now patchifying mask: Semantic segmentation dataset/Tile 8/masks/image_part_007.png
Now patchifying mask: Semantic segmentation dataset/Tile 8/masks/image_part_001.png
Now patchifying mask: Semantic segmentation dataset/Tile 8/masks/image_part_004.png
Now patchifying mask: Semantic segmentation dataset/Tile 8/masks/image_part_003.png
Now patchifying mask: Semantic segmentation dataset/Tile 8/masks/image_part_002.png
Now patchifying mask: Semantic segmentation dataset/Tile 7/masks/image_part_005.png
Now patchifying mask: Semantic segmentation dataset/Tile 7/masks/image_part_009.png
Now patchifying mask: Semantic segmentation dataset/Tile 7/masks/image_part_008.png
Now patchifying mask: Semantic segmentation dataset/Tile 7/masks/image_part_006.png
Now patchifying mask: Semantic segmentation dataset/Tile 7/masks/image_part_007.png
Now patchifying mask: Semantic segmentation dataset/Tile 7/masks/image_part_001.png
Now patchifying mask: Semantic segmentation dataset/Tile 7/masks/image_part_004.png
Now patchifying mask: Semantic segmentation dataset/Tile 7/masks/image_part_003.png
Now patchifying mask: Semantic segmentation dataset/Tile 7/masks/image_part_002.png
Now patchifying mask: Semantic segmentation dataset/Tile 1/masks/image_part_005.png
Now patchifying mask: Semantic segmentation dataset/Tile 1/masks/image_part_009.png
Now patchifying mask: Semantic segmentation dataset/Tile 1/masks/image_part_008.png
Now patchifying mask: Semantic segmentation dataset/Tile 1/masks/image_part_006.png
Now patchifying mask: Semantic segmentation dataset/Tile 1/masks/image_part_007.png
Now patchifying mask: Semantic segmentation dataset/Tile 1/masks/image_part_001.png
Now patchifying mask: Semantic segmentation dataset/Tile 1/masks/image_part_004.png
Now patchifying mask: Semantic segmentation dataset/Tile 1/masks/image_part_003.png
Now patchifying mask: Semantic segmentation dataset/Tile 1/masks/image_part_002.png
Now patchifying mask: Semantic segmentation dataset/Tile 4/masks/image_part_005.png
Now patchifying mask: Semantic segmentation dataset/Tile 4/masks/image_part_009.png
Now patchifying mask: Semantic segmentation dataset/Tile 4/masks/image_part_008.png
Now patchifying mask: Semantic segmentation dataset/Tile 4/masks/image_part_006.png
Now patchifying mask: Semantic segmentation dataset/Tile 4/masks/image_part_007.png
Now patchifying mask: Semantic segmentation dataset/Tile 4/masks/image_part_001.png
Now patchifying mask: Semantic segmentation dataset/Tile 4/masks/image_part_004.png
Now patchifying mask: Semantic segmentation dataset/Tile 4/masks/image_part_003.png
Now patchifying mask: Semantic segmentation dataset/Tile 4/masks/image_part_002.png
Now patchifying mask: Semantic segmentation dataset/Tile 2/masks/image_part_005.png
Now patchifying mask: Semantic segmentation dataset/Tile 2/masks/image_part_009.png
Now patchifying mask: Semantic segmentation dataset/Tile 2/masks/image_part_008.png
Now patchifying mask: Semantic segmentation dataset/Tile 2/masks/image_part_006.png
Now patchifying mask: Semantic segmentation dataset/Tile 2/masks/image_part_007.png
Now patchifying mask: Semantic segmentation dataset/Tile 2/masks/image_part_001.png
Now patchifying mask: Semantic segmentation dataset/Tile 2/masks/image_part_004.png
Now patchifying mask: Semantic segmentation dataset/Tile 2/masks/image_part_003.png
Now patchifying mask: Semantic segmentation dataset/Tile 2/masks/image_part_002.png
Now patchifying mask: Semantic segmentation dataset/Tile 3/masks/image_part_005.png
Now patchifying mask: Semantic segmentation dataset/Tile 3/masks/image_part_009.png
Now patchifying mask: Semantic segmentation dataset/Tile 3/masks/image_part_008.png
Now patchifying mask: Semantic segmentation dataset/Tile 3/masks/image_part_006.png
Now patchifying mask: Semantic segmentation dataset/Tile 3/masks/image_part_007.png
Now patchifying mask: Semantic segmentation dataset/Tile 3/masks/image_part_001.png
Now patchifying mask: Semantic segmentation dataset/Tile 3/masks/image_part_004.png
Now patchifying mask: Semantic segmentation dataset/Tile 3/masks/image_part_003.png
Now patchifying mask: Semantic segmentation dataset/Tile 3/masks/image_part_002.png
Now patchifying mask: Semantic segmentation dataset/Tile 5/masks/image_part_005.png
Now patchifying mask: Semantic segmentation dataset/Tile 5/masks/image_part_009.png
Now patchifying mask: Semantic segmentation dataset/Tile 5/masks/image_part_008.png
Now patchifying mask: Semantic segmentation dataset/Tile 5/masks/image_part_006.png
Now patchifying mask: Semantic segmentation dataset/Tile 5/masks/image_part_007.png
Now patchifying mask: Semantic segmentation dataset/Tile 5/masks/image_part_001.png
Now patchifying mask: Semantic segmentation dataset/Tile 5/masks/image_part_004.png
Now patchifying mask: Semantic segmentation dataset/Tile 5/masks/image_part_003.png
Now patchifying mask: Semantic segmentation dataset/Tile 5/masks/image_part_002.png
Now patchifying mask: Semantic segmentation dataset/Tile 6/masks/image_part_005.png
Now patchifying mask: Semantic segmentation dataset/Tile 6/masks/image_part_009.png
Now patchifying mask: Semantic segmentation dataset/Tile 6/masks/image_part_008.png
Now patchifying mask: Semantic segmentation dataset/Tile 6/masks/image_part_006.png
Now patchifying mask: Semantic segmentation dataset/Tile 6/masks/image_part_007.png
Now patchifying mask: Semantic segmentation dataset/Tile 6/masks/image_part_001.png
Now patchifying mask: Semantic segmentation dataset/Tile 6/masks/image_part_004.png
Now patchifying mask: Semantic segmentation dataset/Tile 6/masks/image_part_003.png
Now patchifying mask: Semantic segmentation dataset/Tile 6/masks/image_part_002.png
60
Unique labels in label dataset are:  [0 1 2 3 4 5]
^CTraceback (most recent call last):
  File "/teamspace/studios/this_studio/Semantic/training_aerial_imagery.py", line 211, in <module>
    X_train, X_test, y_train, y_test = train_test_split(image_dataset, labels_cat, test_size = 0.20, random_state = 42)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/sklearn/utils/_param_validation.py", line 213, in wrapper
    return func(*args, **kwargs)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/sklearn/model_selection/_split.py", line 2810, in train_test_split
    return list(
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/sklearn/model_selection/_split.py", line 2812, in <genexpr>
    (_safe_indexing(a, train), _safe_indexing(a, test)) for a in arrays
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/sklearn/utils/_indexing.py", line 267, in _safe_indexing
    return _array_indexing(X, indices, indices_dtype, axis=axis)
KeyboardInterrupt

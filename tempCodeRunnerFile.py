import os #文件读写模块
import mne
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import xml.etree.ElementTree
from xml.etree.ElementTree import parse
from sklearn.decomposition import PCA, FastICA
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)
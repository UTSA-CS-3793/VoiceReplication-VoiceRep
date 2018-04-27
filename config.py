"""
config.py

Several constants used in the project
"""
ENCODING = 'UTF-8'
NUM_EPOCHS = 8
VOCAB_SIZE = 32
VOCAB = "_$ abcdefghijklmnopqrstuvwxyz'.?"  # _ = padding, $ = ending

EMBED_SIZE = 256
NUM_HIGHWAY_LAYERS = 4
DROPOUT_RATE = 0.5
REDUCTION_FACTOR = 2    # This value can be changed. In the Tacotron paper,
                        # the number 2 was used. The paper said, however,
                        # that numbers as large as 5 worked well

# directories
LOG_DIR = 'C:\\Users\\Sabrina\\Documents\\UTSA\\Intro to AI\\Group Project\\data_lj_1'
MODEL_NAME = 'model-3254.meta'
DATA_PATH = 'data/LJ/LJSpeech-1.1'
TEST_DATA = 'C:\\Users\\Sabrina\\Documents\\UTSA\\Intro to AI\\Group Project\\test_data.txt'
SAVE_DIR = 'C:\\Users\\Sabrina\\Documents\\UTSA\\Intro to AI\\Group Project\\synth_lj_1'
DEVICE = '/cpu:0'

# Signal Processing
SR = 22050                              # Sample rate.
N_MELS = 80                             # 80 band mel scale spectrogram
N_FFT = 2048                            # fft points (samples)
MAX_DB = 100
REF_DB = 20
FRAME_SHIFT = 0.0125                    # seconds
FRAME_LENGTH = 0.05                     # seconds
HOP_LENGTH = int(SR*FRAME_SHIFT)        # samples.
WIN_LENGTH = int(SR*FRAME_LENGTH)       # window length
POWER = 1.2                             # Exponent for amplifying the predicted magnitude
N_ITER = 50                             # Number of inversion iterations
PREEMPHASIS = .97                       # or None
BATCH_SIZE = 32
CHECK_VALS = 1

DATASET_DIR = '../Data/dataset/'
AUDIO_DIR = '../Data/dataset/'

# The length of shortest wav
N_SAMPLES = 800
# MINIMUIN_LENGTH = 400

# Parameters for fbank features
NUM_PREVIOUS_FRAME = 9
#NUM_PREVIOUS_FRAME = 13
NUM_NEXT_FRAME = 23

NUM_FRAMES = NUM_PREVIOUS_FRAME + NUM_NEXT_FRAME
USE_LOGSCALE = True
USE_DELTA = False
USE_SCALE = True
USE_ENERGY = False
SAMPLE_RATE = 16000
TRUNCATE_SOUND_FIRST_SECONDS = 0.5
FILTER_BANK = 64

TDNN_FBANK_FILTER = 23
CMVN = 'cmvnw'
VAD = True
NORMALIZE = True

# Paramters for Spectrogram
NUM_FFT = 512
NUM_FRAMES_SPECT = 300

# Parameters for VAD
VAD_ENERGY_THRESHOLD = 5.5
VAD_ENERGY_MEAN_SCALE = 0.5
VAD_PROPORTION_THRESHOLD = 0.12
VAD_FRAMES_CONTEXT = 2

#
DNN_FILTER = [0.00754443, 0.01200533, 0.01642858, 0.01972508, 0.02147102,
              0.0226888, 0.02469071, 0.02506242, 0.02584714, 0.026544,
              0.02744689, 0.02778561, 0.0288272, 0.02941703, 0.02986392,
              0.02941785, 0.02964678, 0.02946718, 0.03002279, 0.02958207,
              0.02994801, 0.02939642, 0.02953483, 0.02839124, 0.02855285,
              0.02796266, 0.02814863, 0.02748642, 0.02801614, 0.02749056,
              0.02747193, 0.02631736, 0.02615773, 0.0255265, 0.0258321,
              0.02544096, 0.02596661, 0.02555359, 0.02552891, 0.02462734,
              0.0247531, 0.02457665, 0.02512659, 0.02495774, 0.02543424,
              0.02510019, 0.0251669, 0.0244995, 0.02457104, 0.02447885,
              0.02504115, 0.02461394, 0.02525825, 0.02497334, 0.02470482,
              0.02371456, 0.02343372, 0.02306277, 0.02339805, 0.02276554,
              0.02315823, 0.02268941, 0.02241172, 0.02143822, 0.02147482,
              0.02109371, 0.02139363, 0.02079251, 0.02117723, 0.0205691,
              0.02046507, 0.01975423, 0.01990366, 0.01977469, 0.02047094,
              0.0199114, 0.02047737, 0.02007335, 0.02003213, 0.01925786,
              0.019457, 0.019261, 0.019761, 0.01930143, 0.01955818,
              0.01895796, 0.01901542, 0.0180656, 0.0182367, 0.01800973,
              0.01860644, 0.01828756, 0.0189565, 0.01867137, 0.01885242,
              0.01791786, 0.01838151, 0.01811256, 0.01885726, 0.01862201,
              0.01964463, 0.01930191, 0.01989467, 0.01923947, 0.01988874,
              0.01977481, 0.02087282, 0.02053996, 0.02158896, 0.02121622,
              0.02174318, 0.02079145, 0.0215099, 0.02139695, 0.02233053,
              0.02206058, 0.02311639, 0.02285545, 0.0233136, 0.02231173,
              0.02337892, 0.02316566, 0.02410961, 0.02363132, 0.0246225,
              0.02399185, 0.02440539, 0.02356056, 0.02411388, 0.02374618,
              0.02446622, 0.02371911, 0.02471249, 0.02384905, 0.02423068,
              0.02286158, 0.02348385, 0.02281719, 0.02363081, 0.0226122,
              0.02335403, 0.02261329, 0.02296318, 0.02155136, 0.02207833,
              0.02116796, 0.02185807, 0.0204712, 0.02089286, 0.01970987,
              0.01964203, 0.01815534, 0.01857068, 0.01795201, 0.01873097,
              0.01793163, 0.01839271, 0.01749192, 0.01760474, 0.01468716,
              0.01088171]

# cener lores8
VOX_FILTER = [0.00466257, 0.00654789, 0.00863268, 0.01038483, 0.01028049,
              0.01048799, 0.01147182, 0.01169276, 0.01109474, 0.01105516,
              0.01109578, 0.01100038, 0.01073753, 0.01053618, 0.01028662,
              0.00995912, 0.00952664, 0.00922783, 0.00898764, 0.00883661,
              0.00848279, 0.00821643, 0.00796168, 0.00776756, 0.00750316,
              0.00736067, 0.00731447, 0.00732651, 0.00723964, 0.00723305,
              0.00720487, 0.00708541, 0.00687802, 0.0067943, 0.00680173,
              0.00674361, 0.00663911, 0.00659226, 0.00660621, 0.00665272,
              0.00659965, 0.00668223, 0.00680516, 0.00694296, 0.00693447,
              0.00702487, 0.00716135, 0.007218, 0.00716053, 0.00722057,
              0.00722981, 0.00722492, 0.0071189, 0.00713749, 0.00715846,
              0.00714881, 0.00695501, 0.00690282, 0.00687081, 0.00683976,
              0.00667255, 0.00662325, 0.00659407, 0.00652735, 0.00636878,
              0.00633986, 0.00632793, 0.00628727, 0.00615812, 0.00611552,
              0.00609743, 0.00610578, 0.0060375, 0.00608204, 0.00613821,
              0.00617251, 0.00609853, 0.00612716, 0.00614064, 0.00612289,
              0.00603522, 0.00605713, 0.0060365, 0.00601076, 0.00590654,
              0.00589889, 0.00587772, 0.00582076, 0.00566793, 0.00559029,
              0.00553206, 0.00545556, 0.00531212, 0.00525949, 0.00522503,
              0.00518711, 0.00505899, 0.00503567, 0.00502568, 0.00501381,
              0.0049255, 0.00491598, 0.00492826, 0.00493222, 0.00484929,
              0.00487881, 0.00490062, 0.00489815, 0.00481222, 0.0048158,
              0.00487603, 0.00490445, 0.00482612, 0.00487571, 0.00490599,
              0.0049316, 0.00485377, 0.00489318, 0.00491304, 0.0049463,
              0.00488138, 0.00493125, 0.00495534, 0.00502799, 0.00498185,
              0.00501798, 0.00500804, 0.00502666, 0.00490449, 0.00491775,
              0.00489912, 0.0049565, 0.00487797, 0.00487588, 0.00486807,
              0.00484497, 0.00472198, 0.00469652, 0.00465172, 0.00462139,
              0.004491, 0.00445085, 0.00441527, 0.00439442, 0.00427149,
              0.00423693, 0.00424081, 0.00422304, 0.00414158, 0.00413231,
              0.00410051, 0.00409625, 0.00395227, 0.0038867, 0.00389537,
              0.00402454, 0.0040729, 0.00417478, 0.00428524, 0.00444082,
              0.00323236]

TIMIT_FIlTER = [0.00438871, 0.01160961, 0.00957214, 0.01141842, 0.01001372,
                0.01003338, 0.00932224, 0.00917608, 0.00769265, 0.00793429,
                0.00655623, 0.00723177, 0.00651865, 0.00641746, 0.00593072,
                0.00692516, 0.0062521, 0.00656339, 0.00585565, 0.00647662,
                0.00551753, 0.00578098, 0.0053743, 0.00613742, 0.00546302,
                0.00605668, 0.00547354, 0.0060053, 0.00521483, 0.00586652,
                0.0053714, 0.00593993, 0.00526677, 0.00576459, 0.00511302,
                0.00556865, 0.0049147, 0.00545784, 0.0049947, 0.00563289,
                0.00513415, 0.00573856, 0.00520307, 0.00579096, 0.00519562,
                0.00577502, 0.00528196, 0.00583518, 0.00539886, 0.00589751,
                0.00527434, 0.00575686, 0.0053225, 0.00560838, 0.0052121,
                0.00584799, 0.00545013, 0.00592394, 0.00546769, 0.00610077,
                0.00549983, 0.00599317, 0.005605, 0.00624946, 0.0058025,
                0.00640206, 0.00580106, 0.00640739, 0.00589833, 0.0062511,
                0.00573257, 0.00637377, 0.00604299, 0.0064972, 0.00593189,
                0.00647808, 0.00603195, 0.00629374, 0.00580848, 0.00644126,
                0.00609148, 0.0064778, 0.0058325, 0.00647128, 0.00583835,
                0.00614009, 0.00572108, 0.00642075, 0.00600638, 0.00635413,
                0.00572564, 0.00643374, 0.00588593, 0.00608729, 0.00566962,
                0.00626002, 0.00591399, 0.00637217, 0.00575651, 0.00624367,
                0.00572105, 0.0061001, 0.00571627, 0.00630855, 0.00588308,
                0.00644777, 0.00587655, 0.00656378, 0.00594684, 0.00628707,
                0.00591668, 0.00659803, 0.0062303, 0.00676784, 0.00605407,
                0.0067257, 0.00608058, 0.0065976, 0.00617009, 0.0068267,
                0.00654335, 0.00717557, 0.00646952, 0.00714041, 0.00646003,
                0.00702034, 0.00655731, 0.00710213, 0.00664681, 0.00714033,
                0.00646539, 0.00701774, 0.0064717, 0.00685962, 0.00653114,
                0.00703848, 0.00654586, 0.00698199, 0.0063195, 0.00677797,
                0.00628652, 0.00662829, 0.00633496, 0.00677041, 0.00621359,
                0.00677706, 0.00623519, 0.00651487, 0.00581428, 0.00598001,
                0.00562069, 0.0060354, 0.00554316, 0.00575794, 0.00506065,
                0.00522674, 0.00458464, 0.00532818, 0.00480734, 0.00446642,
                0.00266636]

# TIMIT_FIlTER = [0.00023661, 0.01669825, 0.02125655, 0.02627175, 0.01751978,
#                 0.02091466, 0.01601569, 0.01508592, 0.01005613, 0.01154809,
#                 0.00501273, 0.00709499, 0.00669917, 0.00819075, 0.00385591,
#                 0.0049142, 0.00560719, 0.00704817, 0.00608785, 0.00547906,
#                 0.00556113, 0.00545453, 0.00400463, 0.00511243, 0.00450075,
#                 0.003991, 0.00323734, 0.00452368, 0.00330588, 0.00429099,
#                 0.00241071, 0.00295584, 0.00204205, 0.00223633, 0.00126788,
#                 0.00162129, 0.00137238, 0.0021811, 0.00122526, 0.00174958,
#                 0.0016026, 0.00217564, 0.00218022, 0.00297162, 0.00273184,
#                 0.00339905, 0.0026015, 0.00352932, 0.00322493, 0.00314407,
#                 0.00233214, 0.00304625, 0.00295879, 0.00339754, 0.00292505,
#                 0.00360743, 0.003755, 0.00369849, 0.00373958, 0.00433877,
#                 0.00439785, 0.00507664, 0.00467313, 0.00523916, 0.00531578,
#                 0.00523321, 0.00491371, 0.00513935, 0.0048346, 0.00539471,
#                 0.0053772, 0.00593623, 0.00624347, 0.00592501, 0.00565248,
#                 0.00595077, 0.00608898, 0.0059869, 0.00564812, 0.00582707,
#                 0.00632508, 0.00601213, 0.00570106, 0.00581659, 0.0059829,
#                 0.00599994, 0.00574064, 0.00559464, 0.00603581, 0.00579266,
#                 0.00546322, 0.00548659, 0.00573457, 0.00576936, 0.00549532,
#                 0.00513439, 0.00560857, 0.00543168, 0.00498894, 0.00494378,
#                 0.00515174, 0.00564291, 0.00541211, 0.00495206, 0.00528614,
#                 0.00560756, 0.00553618, 0.00554504, 0.00552215, 0.00593071,
#                 0.00605203, 0.00582979, 0.0061642, 0.0063893, 0.0059803,
#                 0.00582385, 0.00613368, 0.00695021, 0.00729953, 0.00688198,
#                 0.00729623, 0.008301, 0.00835363, 0.00808782, 0.00781258,
#                 0.00879345, 0.00953409, 0.00860236, 0.00864163, 0.00972668,
#                 0.01001589, 0.00976134, 0.00942435, 0.01032801, 0.01045107,
#                 0.00999452, 0.00996147, 0.01058146, 0.01009149, 0.00976301,
#                 0.00882807, 0.00978413, 0.01081803, 0.01038582, 0.00894065,
#                 0.00899766, 0.00845226, 0.00926698, 0.00817182, 0.00765828,
#                 0.00768802, 0.00539208, 0.00614423, 0.00657554, 0.00587436,
#                 0.00346217, 0.0020781, 0.00061639, 0.00148055, 0.00293487,
#                 0.00092614]

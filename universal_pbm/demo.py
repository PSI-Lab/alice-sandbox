import numpy as np
from model_utils import UPbmModel, UPbmEnsemble

filename = 'result/BHLHE40/Mus_musculus|M00251_1.94d|Badis09|Bhlhb2_1274.3=v2.fold_0.h5'
# model = load_model(filename)


model = UPbmModel(filename, 36)

print model.predict_sequence('ACTATTCTGGCTTACTGTATCGGCGACCGCATGTTG')
print model.predict_sequence('TAGTCACGCGACATATCTCAATGAATGTAACGGGAA')

print model.predict_sequence('TAGTCACGCGACATATCTCAATGAATGTAACGGGAAACGT')

print model.predict_sequence('ACTATTCTGGCTTACTATCGGCGACCGCATGTTG')

ens_model = UPbmEnsemble('BHLHE40', 'Mus_musculus|M00251_1.94d|Badis09|Bhlhb2_1274.3=v2', np.median)

print ens_model.predict_sequence('ACTATTCTGGCTTACTGTATCGGCGACCGCATGTTG')

print ens_model.predict_sequence('TAGTCACGCGACATATCTCAATGAATGTAACGGGAAACGT')

print ens_model.predict_sequence('ACTATTCTGGCTTACTATCGGCGACCGCATGTTG')


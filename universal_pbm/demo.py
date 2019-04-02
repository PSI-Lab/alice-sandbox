import numpy as np
from model_utils import UPbmModel, UPbmEnsemble

# TODO check that keras version is >=2.2.4

# TODO make model_utils.py

filename = 'result/BHLHE40/Mus_musculus|M00251_1.94d|Badis09|Bhlhb2_1274.3=v2.fold_0.h5'
# model = load_model(filename)
#
# d = np.random.rand(1, 36, 4)
# print model.predict(d)
#
# d = np.random.rand(1, 36, 4)
# print model.predict(d)

model = UPbmModel(filename, 36)

print model.predict_sequence('ACTATTCTGGCTTACTGTATCGGCGACCGCATGTTG')
print model.predict_sequence('TAGTCACGCGACATATCTCAATGAATGTAACGGGAA')

print model.predict_sequence('TAGTCACGCGACATATCTCACTATTCTGGCTTACTGTATCGGCGACCGCATGTTGAATGAATGTAACGGGAA')


ens_model = UPbmEnsemble('BHLHE40', 'Mus_musculus|M00251_1.94d|Badis09|Bhlhb2_1274.3=v2', np.median)

print ens_model.predict_sequence('ACTATTCTGGCTTACTGTATCGGCGACCGCATGTTG')

print ens_model.predict_sequence('TAGTCACGCGACATATCTCACTATTCTGGCTTACTGTATCGGCGACCGCATGTTGAATGAATGTAACGGGAA')


# TODO demo that short sequence will fail, minimum sequence length required

# TODO mention that model might not generalize well to super long sequence

# TODO demo scanning window


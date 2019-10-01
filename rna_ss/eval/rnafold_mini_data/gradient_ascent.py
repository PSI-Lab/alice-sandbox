#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt 
import numpy as np
import tensorflow as tf
import keras
import keras.backend as kb
import logomaker
import pandas as pd


# In[30]:


import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
sns.set(color_codes=True)
import cufflinks as cf
cf.go_offline()
cf.set_config_file(theme='ggplot')


# In[2]:


model = keras.models.load_model('input_data/model.hdf5', custom_objects={'kb': kb})


# In[3]:


DNA_ENCODING = np.asarray([[0, 0, 0, 0],
                           [1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])


# In[4]:


# def encode_seq(seq):
#     def _encode_seq(seq):
#         seq = seq.upper().replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4').replace('U', '4').replace('N', '0')
#         x = np.asarray(map(int, list(seq)))
#         x = DNA_ENCODING[x.astype('int8')]
#         return x
#     seq_rev = seq[::-1]
#     x1 = _encode_seq(seq)
#     x2 = _encode_seq(seq_rev)
#     return x1, x2


def encode_seq(seq):
    def _encode_seq(seq):
        seq = seq.upper().replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4').replace('U', '4').replace('N', '0')
        x = np.asarray(map(int, list(seq)))
        x = DNA_ENCODING[x.astype('int8')]
        return x
    return _encode_seq(seq)


# In[5]:


def w_to_df(w):
    assert len(w.shape) == 2
    assert w.shape[1] == 4
    data = []
    for i in range(w.shape[0]):
        data.append({
            'A': w[i, 0],
            'C': w[i, 1],
            'G': w[i, 2],
            'U': w[i, 3],
        })
    return pd.DataFrame(data)


# In[6]:


# test seq
seq_org = 'GAAUGGGUUAAAAGGGGGGCGCAUUGGUACCUGCUAUUAGGGAUCAAUCGG'


# In[7]:


# current output
x1 = encode_seq(seq_org)
x1 = x1[np.newaxis, :, :]
pred_org = model.predict(x1)[0, :, 0]


# In[8]:


print pred_org.max(), pred_org.argmax()


# In[33]:


df_plot = pd.DataFrame({
    'prediction': pred_org,
#     'text': list(seq_org),
})
df_plot.iplot(kind='line', xTitle='base position', yTitle='prediction')


# In[9]:


# test idx - make sure to pick one that makes sense!
new_idx = 10
assert new_idx != pred_org.argmax()
print pred_org[new_idx], new_idx


# In[10]:


layer_output = model.layers[-1].output


# In[11]:


loss = kb.mean(layer_output[:,new_idx, :])


# In[12]:



input_node = model.layers[0].input


# In[13]:


print input_node


# In[14]:



grads = kb.gradients(loss, input_node)[0]


# In[15]:


grads /= (kb.sqrt(kb.mean(kb.square(grads))) + 1e-5)


# In[16]:


iterate = kb.function([input_node], [loss, grads])


# In[ ]:





# In[17]:


# we start from a gray image with some noise
x1_new = x1.astype(np.float32)
# add a little bit noise
x1_new += 1e-2
# normalize
x1_new = x1_new/np.sum(x1_new, axis=-1)[:, :, np.newaxis]
# run gradient ascent
for i in range(200):
    loss_value, grads_value = iterate([x1_new])
    x1_new += grads_value * 0.01
    # re-normalize
    x1_new = x1_new/np.sum(x1_new, axis=-1)[:, :, np.newaxis]
#     print x1_new

    # plot every 10 iteration
    if i % 10 == 0:
        df = w_to_df(x1_new[0, :, :])
        filter_logo = logomaker.Logo(df)
        filter_logo.ax.set_title("Gradient ascent iteration {}".format(i))
        filter_logo.fig.show()

    pred_new = model.predict(x1_new)[0, :, 0]
    print pred_new[new_idx]


# In[26]:


# TODO after gradient ascent, take argmax -> new sequence
# check prediction
# run RNAfold look at structure
nt_dict = ['A', 'C', 'G', 'U']
seq_new = []
for i in range(x1_new.shape[1]):
    seq_new.append(nt_dict[np.argmax(x1_new[0, i, :])])
seq_new = ''.join(seq_new)
print seq_new


_x1 = encode_seq(seq_new)
_x1 = _x1[np.newaxis, :, :]
_pred = model.predict(_x1)[0, :, 0]

print np.argmax(_pred), np.max(_pred)
print _pred[new_idx]


# In[ ]:





# In[27]:


seq_org


# In[28]:


seq_new


# In[34]:


df_plot = pd.DataFrame({
    'prediction': _pred,
#     'text': list(seq_org),
})
df_plot.iplot(kind='line', xTitle='base position', yTitle='prediction')


# In[ ]:





# In[18]:


grads_value[0, :5, :]


# In[ ]:





# In[19]:


x1_new[:, :5, :]


# In[ ]:





# In[20]:


# TODO normalize so input makes sense?


# In[21]:


model.summary()


# In[ ]:





# -*- coding: utf-8 -*-
"""
Created on Wed May 18 17:29:24 2016

@author: Tianwei Xing
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 04 15:13:58 2016

@author: Tianwei Xing
"""

import os
import numpy as np
file_name='GoodModel.npz'
model=np.load(file_name)
file_list=model.files
listID=0
##for list_name in file_list:
#for j in range(0,len(file_list)):
#    list_name='arr_'+str(j)
#    print(list_name)
#    current_list=model[list_name]
#    list_shape=current_list.shape
#    list_shape_len=len(list_shape)
#    dimension=1
#    for i in range(0, list_shape_len):
#        dimension=dimension*list_shape[i]
#    new_list=current_list.reshape(dimension)
#    file_path=os.path.join('model_save','list_save_'+'%02d'%int(listID)+'.txt')
#    np.savetxt(file_path,new_list,fmt='%1.4e')
#    listID=listID+1
# #   print(listID)


for j in range(0, len(file_list)):
    list_name='arr_'+str(j)
    print(list_name)
    current_list=model[list_name]
    list_shape=current_list.shape
    print(list_shape)
    
#arr_0 ----1.1 kernel param
#(128L, 3L, 3L, 3L)
#arr_6 ----1.2 kernel param
#(128L, 128L, 3L, 3L)
#arr_12 ----2.1 kernel param
#(256L, 128L, 3L, 3L)
#arr_18 ----2.2 kernel param
#(256L, 256L, 3L, 3L)
#arr_24 ----3.1 kernel param
#(512L, 256L, 3L, 3L)
#arr_30 ----3.2 kernel param
#(512L, 512L, 3L, 3L)
#arr_36 ----4.1 denselayer param
#(8192L, 1024L)
#arr_42 ----4.2 denselayer param
#(1024L, 1024L)
#arr_48 ----4.3 denselayer param
#(1024L, 10L)

print("Saving trained kernels valuse to txt file")
name_list=['arr_0', 'arr_6', 'arr_12', 'arr_18', 'arr_24', 'arr_30', 'arr_36', 'arr_42', 'arr_48']
for name in name_list:
    current_list=model[name]
    list_shape=current_list.shape
    list_shape_len=len(list_shape)
    dimension=1
    for i in range(0, list_shape_len):
        dimension=dimension*list_shape[i]
    new_list=current_list.reshape(dimension)
    file_path=os.path.join('model_save','list_save_'+ name +'.txt')
    np.savetxt(file_path,new_list,fmt='%.3f')
    listID=listID+1
 #   print(listID)
print("File saved")
    

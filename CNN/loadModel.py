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
#for list_name in file_list:
for j in range(0,len(file_list)):
    list_name='arr_'+str(j)
    print(list_name)
    current_list=model[list_name]
    list_shape=current_list.shape
    list_shape_len=len(list_shape)
    dimension=1
    for i in range(0, list_shape_len):
        dimension=dimension*list_shape[i]
    new_list=current_list.reshape(dimension)
    file_path=os.path.join('model_save','list_save_'+'%02d'%int(listID)+'.txt')
    np.savetxt(file_path,new_list,fmt='%1.4e')
    listID=listID+1
 #   print(listID)

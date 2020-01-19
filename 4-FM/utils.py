import tensorflow as tf
import pickle
import numpy as np 
import os
import pandas as pd 
from collections import Counter

class Process:
    def __init__(self, file,opt, chunksize=10000):
        self.opt = opt
        self.sample_index = 0
        self.file = file
        self.data = pd.read_csv(self.file,chunksize=chunksize)
        self.one_hot_express()
        if not hasattr(self,'sp_shape'):
            setattr(self,'sp_shape',np.array([1,self.feature_length], dtype=np.int64))

    def one_hot_express(self):
        '''
        获取每一个特征每一数值所代表one-hot的index
        '''
        # 直接进行one-hot的特征
        click = set()
        hour = set()
        C1 = set()
        banner_pos = set()
        site_category = set()
        app_category = set()
        device_type = set()
        device_conn_type = set()
        C15 = set()
        C16 = set()
        C18 = set()
        C20 = set()

        hours = set(range(24))

        # 通过频数来编码one-hot特征
        C14 = dict()
        C17 = dict()
        C19 = dict()
        C21 = dict()
        site_id = dict()
        site_domain = dict()
        app_id = dict()
        app_domain = dict()
        device_model = dict()
        device_id = dict()
        device_ip = dict()

        for kk, data in enumerate(self.data):
            if kk > 100:
                break

            click_v = set(data['click'].values)
            click = click | click_v

            C1_v = set(data['C1'].values)
            C1 = C1 | C1_v

            C15_v = set(data['C15'].values)
            C15 = C15 | C15_v

            C16_v = set(data['C16'].values)
            C16 = C16 | C16_v

            C18_v = set(data['C18'].values)
            C18 = C18 | C18_v

            C20_v = set(data['C20'].values)
            C20 = C20 | C20_v

            banner_pos_v = set(data['banner_pos'].values)
            banner_pos = banner_pos | banner_pos_v

            site_category_v = set(data['site_category'].values)
            site_category = site_category | site_category_v

            app_category_v = set(data['app_category'].values)
            app_category = app_category | app_category_v

            device_type_v = set(data['device_type'].values)
            device_type = device_type | device_type_v

            device_conn_type_v = set(data['device_conn_type'].values)
            device_conn_type = device_conn_type | device_conn_type_v


            #------------------------------------------------------------
            C14_list = data['C14'].values
            for k,v in Counter(C14_list).items():
                if k in C14.keys():
                    C14[k] += v
                else:
                    C14[k] = v

            C17_list = data['C17'].values
            for k,v in Counter(C17_list).items():
                if k in C17.keys():
                    C17[k] += v
                else:
                    C17[k] = v

            C19_list = data['C19'].values
            for k,v in Counter(C19_list).items():
                if k in C19.keys():
                    C19[k] += v
                else:
                    C19[k] = v

            C21_list = data['C21'].values
            for k,v in Counter(C21_list).items():
                if k in C21.keys():
                    C21[k] += v
                else:
                    C21[k] = v

            site_id_list = data['site_id'].values
            for k,v in Counter(site_id_list).items():
                if k in site_id.keys():
                    site_id[k] += v
                else:
                    site_id[k] = v

            site_domain_list = data['site_domain'].values
            for k,v in Counter(site_domain_list).items():
                if k in site_domain.keys():
                    site_domain[k] += v
                else:
                    site_domain[k] = v

            app_id_list = data['app_id'].values
            for k,v in Counter(app_id_list).items():
                if k in app_id.keys():
                    app_id[k] += v
                else:
                    app_id[k] = v

            app_domain_list = data['app_domain'].values
            for k,v in Counter(app_domain_list).items():
                if k in app_domain.keys():
                    app_domain[k] += v
                else:
                    app_domain[k] = v

            device_model_list = data['device_model'].values
            for k,v in Counter(device_model_list).items():
                if k in device_model.keys():
                    device_model[k] += v
                else:
                    device_model[k] = v

            device_id_list = data['device_id'].values
            for k,v in Counter(device_id_list).items():
                if k in device_id.keys():
                    device_id[k] += v
                else:
                    device_id[k] = v

            device_ip_list = data['device_ip'].values
            for k,v in Counter(device_ip_list).items():
                if k in device_ip.keys():
                    device_ip[k] += v
                else:
                    device_ip[k] = v
            
            print('\rhave process : %06d'%kk,end='',flush=True)
        direct_encoding_fields = ['hour', 'C1', 'C15', 'C16', 'C18', 'C20',
                          'banner_pos',  'site_category','app_category',
                          'device_type','device_conn_type']

        frequency_encoding_fields = ['C14','C17', 'C19', 'C21',
                             'site_id','site_domain','app_id','app_domain',
                              'device_model', 'device_id']
        print()
        #为特征匹配one-hot的index
        ind = 0
        self.fields_dict = {}
        for kk,field in enumerate(direct_encoding_fields):
            # value to one-hot-encoding index dict
            field_dict = {}
            field_sets = eval(field)
            for value in list(field_sets):
                field_dict[value] = ind
                ind += 1
            self.fields_dict[field] = field_dict
            print('\rhave process : %06d'%kk,end='',flush=True)
        print()
        for kk,field in enumerate(frequency_encoding_fields):
            # value to one-hot-encoding index dict
            field2count = eval(field)
            index_rare = None
            for k,count in field2count.items():
                if count < 10:
                    if index_rare == None:
                        field_dict[k] = ind
                        index_rare = ind
                        ind += 1
                    else:
                        field_dict[k] = index_rare
                else:
                    field_dict[k] = ind
                    ind += 1
            self.fields_dict[field] = field_dict
            print('\rhave process : %06d'%kk,end='',flush=True)
        print()
        # one-hot 长度

        self.feature_length = max(field_dict.values())
                
    def get_data(self):
        
        dataset = tf.data.Dataset.from_generator(self._sample_data, output_types=(tf.int64,tf.float32,tf.int32))
        dataset = dataset.map(self._parse_function,num_parallel_calls=8)
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(self.opt.batch_size)
        dataset = dataset.repeat(self.opt.num_epochs)
        iterator = dataset.make_one_shot_iterator()
        batch_input, batch_label = iterator.get_next()

        return batch_input,batch_label

    def _sample_data(self):
        for kk,data in enumerate(self.data):
            if kk < 100:
                for i in range(len(data)):
                    sample = data.iloc[i,:]
                    click = sample['click']
                    if click == 0:
                        label = 0
                    else:
                        label = 1
                    index,sp_value = self._one_hot_representation(sample)
                    
                    yield (index,sp_value,label)
                

    def _one_hot_representation(self,sample):
        """
        One hot presentation for every sample data
        :param fields_dict: fields value to array index
        :param sample: sample data, type of pd.series
        :param isample: sample index
        :return: sample index
        """
        index = []
        for field in self.fields_dict:
            # get index of array
            if field == 'hour':
                field_value = int(str(sample[field])[-2:])
            else:
                field_value = sample[field]
            ind = self.fields_dict[field][field_value]
            index.append([0,ind])
        

        
        sp_value = np.ones([len(index)], dtype=np.float32)
        return np.array(index),sp_value
                    
    def _parse_function(self,index_list,sp_value,label):
        
                

        inputs = tf.SparseTensor(indices=index_list,values=sp_value,dense_shape=self.sp_shape)
        inputs = tf.sparse_reshape(inputs, [-1])
        return inputs,label
 
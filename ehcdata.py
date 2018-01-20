# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

class data_manager:
    def __init__(self):
        self.__read_file()
        self.__user_dict()
        self.__cur_index_tr = 0
        self.__cur_index_te = 0
        self.__cur_month = 6
        
    def __read_file(self):
        self.v = pd.read_csv('view_step1.csv', parse_dates=[0])
        self.o = pd.read_csv('order_step1.csv')
        
    def __user_dict(self):
        users = self.v['userid'].unique()
        self.u2id = dict(zip(users, range(45000)))
        self.id2u = dict(zip(range(45000), users))
    def p_mat(self, vo='o'):
        p_mat = np.zeros((45000, 13))
        if vo == 'o':
            x = self.o
        elif vo == 'v':
            x = self.v
        for i in range(x.shape[0]):
            u = self.u2id[x['userid'][i]]
            s = x['storeid'][i]-1
            p_mat[u, s] = 1
        return p_mat
        
    def c_mat(self, alpha, eps, vo='o') :
        c_mat = np.zeros((45000, 13))
        if vo == 'o':
            x = self.o
        elif vo == 'v':
            x = self.v
        for i in range(x.shape[0]):
            u = self.u2id[x['userid'][i]]
            s = x['storeid'][i]-1
            c_mat[u, s] += 1
        c_mat = 1 + alpha * np.log(1 + 1 / eps * c_mat)
        return c_mat
        
    def get_train_batch(self, batch_size=100):
        if self.__cur_month == 6:
            if (self.__cur_index_tr + batch_size) != self.x6.shape[0]:
                self.__cur_index_tr += batch_size
                return self.x6[(self.__cur_index_tr-batch_size):self.__cur_index_tr, :, :, :], \
                       self.y7[(self.__cur_index_tr-batch_size):self.__cur_index_tr, :]
            else:
                self.__cur_index_tr = 0
                self.__cur_month = 7
        if self.__cur_month == 7:
            if (self.__cur_index_tr + batch_size) != self.x7.shape[0]:
                self.__cur_index_tr += batch_size
                return self.x7[(self.__cur_index_tr-batch_size):self.__cur_index_tr, :, :, :], \
                       self.y8[(self.__cur_index_tr-batch_size):self.__cur_index_tr, :]
            else:
                self.__cur_index_tr = 0
                self.__cur_month = 8
        if self.__cur_month == 8:
            if (self.__cur_index_tr + batch_size) != self.x8.shape[0]:
                self.__cur_index_tr += batch_size
                return self.x8[(self.__cur_index_tr-batch_size):self.__cur_index_tr, :, :, :], \
                       self.y9[(self.__cur_index_tr-batch_size):self.__cur_index_tr, :]
            else:
                self.__cur_index_tr = 0
                self.__cur_month = 6
                
    def get_test(self, batch_size=45000):
        return self.x9
        
    def get_x(self, month):
        if month==6:
            return self.x6
        if month==7:
            return self.x7
        if month==8:
            return self.x8
        if month==9:
            return self.x9
            
    def get_y(self, month):
        if month==7:
            return self.y7
        if month==8:
            return self.y8
        if month==9:
            return self.y9
        
    def get_x_cat_10day(self, month):
        if month==6:
            return self.x6_cat_10day
        if month==7:
            return self.x7_cat_10day
        if month==8:
            return self.x8_cat_10day
        if month==9:
            return self.x9_cat_10day
            
    def gen_x_cat_10day(self):
        print('Generating x_cat_10day data...')
        self.x6_cat_10day = np.zeros((45000, 3, 116*13, 2))
        self.x7_cat_10day = np.zeros((45000, 3, 116*13, 2))
        self.x8_cat_10day = np.zeros((45000, 3, 116*13, 2))
        self.x9_cat_10day = np.zeros((45000, 3, 116*13, 2))
        cat1 = self.v.catid_1.unique() 
        self.cat1id = dict(zip(cat1, range(cat1.size)))
        self.id2cat1 = dict(zip(range(cat1.size), cat1))
        cat2 = self.v.catid_2.unique()
        self.cat2id = dict(zip(cat2, range(cat2.size)))
        self.id2cat2 = dict(zip(range(cat2.size), cat2))
        self._view2x_cat_10day(self.x6_cat_10day, 6)
        self._view2x_cat_10day(self.x7_cat_10day, 7)
        self._view2x_cat_10day(self.x8_cat_10day, 8)
        self._view2x_cat_10day(self.x9_cat_10day, 9)
        self._order2x_cat_10day(self.x6_cat_10day, 6)
        self._order2x_cat_10day(self.x7_cat_10day, 7)
        self._order2x_cat_10day(self.x8_cat_10day, 8)
        self._order2x_cat_10day(self.x9_cat_10day, 9)
        print('Finished!')
        
    def _view2x_cat_10day(self, x, month):
        t = self.v.loc[self.v['dates'].dt.month == month, ['dates', 'userid', 'storeid', 'catid_1', 'catid_2']]
        t = t.reset_index()
        u = np.zeros(t.shape[0])
        for i in range(t.shape[0]):
            u[i] = self.u2id[t.userid[i]]
        d = np.array(t['dates'].dt.day) // 10
        d[d>2] = 2
        s = (t['storeid'] - 1) * 116
        c1 = t.catid_1
        c2 = t.catid_2
        for i in range(t.shape[0]):
            sid = int(s[i])
            cid1 = int(self.cat1id[c1[i]] + sid + 1)
            cid2 = int(self.cat2id[c2[i]] + sid + 31)
            uid = int(u[i])
            did = int(d[i])
            x[uid, did, sid, 0] += 1
            x[uid, did, cid1, 0] += 1
            x[uid, did, cid2, 0] += 1
            
    def _order2x_cat_10day(self, x, month):
        t = self.o.loc[self.o.datestime // 100000000 % 100 == month, ['datestime', 'userid', 'storeid', 'catid_1', 'catid_2']]
        t = t.reset_index()
        u = np.zeros(t.shape[0])
        for i in range(t.shape[0]):
            u[i] = self.u2id[t.userid[i]]
        d = np.array(t['datestime'] // 1000000 % 100) // 10
        d[d>2] = 2
        s = (t['storeid'] - 1) * 116
        c1 = t.catid_1
        c2 = t.catid_2
        for i in range(t.shape[0]):
            sid = int(s[i])
            cid1 = int(self.cat1id[c1[i]] + sid + 1)
            cid2 = int(self.cat2id[c2[i]] + sid + 31)
            uid = int(u[i])
            did = int(d[i])
            x[uid, did, sid, 1] += 1
            x[uid, did, cid1, 1] += 1
            x[uid, did, cid2, 1] += 1
            
    def gen_xy(self):
        print(">>>>> Generating training and testing data...")
        self.x6 = np.zeros((45000, 24*31, 13, 2))
        self.y7 = np.zeros((45000, 13))
        self.x7 = np.zeros((45000, 24*31, 13, 2))
        self.y8 = np.zeros((45000, 13))
        self.x8 = np.zeros((45000, 24*31, 13, 2))
        self.y9 = np.zeros((45000, 13))
        self.x9 = np.zeros((45000, 24*31, 13, 2))
        self.__view2x(self.v, self.x6, 6)
        self.__order2x(self.o, self.x6, 6)
        self.__view2x(self.v, self.x7, 7)
        self.__order2x(self.o, self.x7, 7)
        self.__view2x(self.v, self.x8, 8)
        self.__order2x(self.o, self.x8, 8)
        self.__view2x(self.v, self.x9, 9)
        self.__order2x(self.o, self.x9, 9)
        self.__order2y(self.o, self.y7, 7)
        self.__order2y(self.o, self.y8, 8)
        self.__order2y(self.o, self.y9, 9)
        print(">>>>> Finished.")
        
    def __view2x(self, v, x, month):
        t = v.loc[v['dates'].dt.month == month, ['dates', 'time', 'userid', 'storeid']]
        t = t.reset_index()
        u = np.zeros(t.shape[0])
        for i in range(t.shape[0]):
            u[i] = self.u2id[t.userid[i]]
        d = t['dates'].dt.day
        h = pd.to_datetime(t['time'], format='%H:%M:%S').dt.hour
        dh = (d-1)*24 + h
        s = t['storeid'] - 1
        for i, v in enumerate(zip(u, dh, s)):
            x[int(v[0]), int(v[1]), int(v[2]), 0] += 1
            
    def __order2x(self, o, x, month):
        t = o.loc[o.datestime // 100000000 % 100 == month, ['datestime', 'userid', 'storeid']]
        t = t.reset_index()
        u = np.zeros(t.shape[0])
        for i in range(t.shape[0]):
            u[i] = self.u2id[t.userid[i]]
        d = t['datestime'] // 1000000 % 100
        h = t['datestime'] // 10000 % 100
        dh = (d-1)*24+h
        s = t['storeid']-1
        for i, v in enumerate(zip(u, dh, s)):
            x[int(v[0]), int(v[1]), int(v[2]), 1] += 1
            
    def __order2y(self, o, y, month):
        t = o.loc[o.datestime // 100000000 % 100 == month, ['userid', 'storeid']]
        t = t.reset_index()
        u = np.zeros(t.shape[0])
        for i in range(t.shape[0]):
            u[i] = self.u2id[t.userid[i]]
        s = t['storeid'] - 1
        for i, v in enumerate(zip(u, s)):
            y[int(v[0]), int(v[1])] += 1

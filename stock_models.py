# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 13:49:37 2019

@author: Nate
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy as copy
from sklearn import svm
from sklearn import preprocessing
import random as random
import urllib.request
import datetime
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import to_categorical 
import pywt as pwt
import core as aecore
import math as math
import permutations as perm
import itertools as it


####### Helper Functions
def numpy2df(numpy_array):
    out = {}
    for i in range(len(numpy_array[0])):
        out.update({'Column'+str(i) : numpy_array[:,i]})
    return pd.DataFrame(out)

def make_matrix(vector1, vector2):
    out = []
    for i in vector1:
        out.append([])
        for j in vector2:
            out[-1].append([i,j])
    return out


#### Data Structures
    
class data_set:
    def __init__(self, X, Y):
        self.XData = X
        self.YData = Y
        self.XY = []
        for i in range(len(self.XData)):
             self.XY.append([self.XData[i] , self.YData[i]])
       
        self.XDataOrig = copy.deepcopy(self.XData)
        self.YDataOrig = copy.deepcopy(self.YData)
        
    def normalize(self, span = (-1, +1), YFlag = 'N'):
        self.Xtrans = self.XData.transpose()
        self.Xnormtrans = []
        for i in range(len(self.Xtrans)):
            self.Xnormtrans.append(np.interp(self.Xtrans[i], (self.Xtrans[i].min(), self.Xtrans[i].max()), (span[0], span[1])))
        self.XDataNT = np.asarray(self.Xnormtrans)
        self.XData = self.XDataNT.transpose()
        
    def reset_data(self):
        self.XData = copy.deepcopy(self.XDataOrig)
        self.YData = copy.deepcopy(self.YDataOrig)
        
    def shuffle(self):
        self.XYShuffle = copy.deepcopy(self.XY)
        self.XRandom = []
        self.YRandom = []
        np.random.shuffle(self.XYShuffle)
        for i in range(len(self.XYShuffle)):
            self.XRandom.append(self.XYShuffle[i][0])
            self.YRandom.append(self.XYShuffle[i][1])
        self.XData = np.asarray(self.XRandom)
        self.YData = np.asarray(self.YRandom)
        
    def autoencode(self, epochs = 500, plot = 'N', divisor = 2, path = None):
        print("Encoding Data to Dim: "+str(int(len(self.XData[0])/divisor)))
        self.autoencoder = aecore.Xae(self.XData, arch = 2*[int(len(self.XData[0])/divisor)], epochs = epochs, latent_dim = int(len(self.XData[0])/divisor))
        self.encoded_rep = self.autoencoder.encode()
        self.encoded_data = data_set(self.encoded_rep, self.YData)
        self.decoded_rep = self.autoencoder.decode(self.encoded_rep)
        if plot == 'Y':
            print("Plotting Original vs. Recon")
            fig, ax = plt.subplots()
            self.plot_data = []
            for i in range(len(self.encoded_rep)-2):
                self.plot_data.append(np.interp(self.encoded_rep[i], (self.encoded_rep[i].min(), self.encoded_rep[i].max()), (-1, +1)))
            self.prod_plot = np.asarray(self.plot_data).transpose()
            self.color_range = np.arange(0.5,1.0,1.0/len(self.prod_plot))
            for i in range(len(self.color_range)):
                plt.plot(self.prod_plot[i], linewidth = 2,color=(0.0,np.flip(self.color_range)[i],self.color_range[i]))
            plt.plot(self.XData[:,0], linewidth = 4,color=(0.66,0.33,0.33,1.0))
            if path != None:
                fig.savefig(path)
            plt.show()
            
    def wavelet_denoise(self, wavelet = 'db1', plot = 'Y', level = 1):
        ### only built for Level 1 decomp
        self.Xtrans = self.XData.transpose()
        self.Xnormtrans = []
        self.wavecoef = []
        self.thresh_wavecoef = []
        self.denoisedXtrans = []
        for i in range(len(self.Xtrans)):     
            self.maxwave = pwt.dwt_max_level(len(self.Xtrans[i]), wavelet)
            if level != 'Auto':
                self.level = level
            self.wavecoef.append(pwt.wavedec(self.Xtrans[i],wavelet, level = self.level))
            self.thresh_A = (np.average(np.abs(self.wavecoef[i][0]))) - (np.std(np.abs(self.wavecoef[i][0])))
            self.thresh_D =(np.average(np.abs(self.wavecoef[i][1]))) - (np.std(np.abs(self.wavecoef[i][1])))
            self.thresh_wavecoef.append((pwt.threshold(self.wavecoef[i][0], self.thresh_A), (pwt.threshold(self.wavecoef[i][1], self.thresh_D))))
            self.denoisedXtrans.append(pwt.idwt(self.thresh_wavecoef[-1][0], self.thresh_wavecoef[-1][1], wavelet))
        self.XDataNT = np.asarray(self.denoisedXtrans)
        self.XData = self.XDataNT.transpose()
        if plot != 'N':                
            plt.plot(self.Xtrans[plot])
            plt.plot(self.denoisedXtrans[plot])
            plt.show()
            

    def svd(self, degree = 0, plot = 'N'):
            self.m, self.n = (len(self.XData), len(self.XData[0]))
            self.U, self.s, self.Vh = np.linalg.svd(self.XData)
            self.sigma = np.zeros((self.m,self.n))
            for i in range(min(self.m, self.n)):
                self.sigma[i, i] = self.s[i]
            self.s0 = copy.deepcopy(self.s)
            for i in range(1,degree+1):
                self.s0[-i] = 0.0
            self.sigma0 = np.zeros((self.m,self.n))
            for i in range(min(self.m, self.n)):
                self.sigma0[i, i] = self.s0[i]
            self.denoised_data = np.dot(self.U, np.dot(self.sigma0, self.Vh))
            if plot == 'Y':
                plt.plot(self.XData[:,0], color=(0.9,0.05,0.05,1.0))
                plt.plot(self.XData[:,-1], color=(0.05,0.9,0.05,1.0))
                plt.plot(self.XData[:,-2], color=(0.05,0.05,0.9,1.0))
                plt.plot(self.denoised_data[:,0], color=(0.5,0.25,0.25,0.8))
                plt.plot(self.denoised_data[:,-1],color=(0.25,0.5,0.25,0.8))
                plt.plot(self.denoised_data[:,-2],color=(0.25,0.25,0.5,0.8))
                plt.show()
            else:
                pass
            self.XData = self.denoised_data
        
               
    def test_and_train(self, percent_train, rand = 'Y', plot = 'N'):
            self.split_pointX = int(percent_train*len(self.XData))
            self.split_pointY = int(percent_train*len(self.YData))
            self.X_train = self.XData[:self.split_pointX]
            self.Y_train = self.YData[:self.split_pointY]
            self.X_test = self.XData[self.split_pointX:]
            self.Y_test = self.YData[self.split_pointY:]
            return self.X_train, self.Y_train, self.X_test, self.Y_test
            
    
    
    def prepare_LSTM(self, time_window = 'Auto'):
            self.num_features = int(self.XData.shape[1])
            self.time_steps = self.XData.shape[0]
            self.test_window = 3
            if time_window == 'Auto':
                while self.time_steps%self.test_window != 0:
                    self.test_window += 1
                    if (self.time_steps % self.test_window) == 0:
                        print(self.test_window)
                        break
                self.time_window = self.test_window
            else:
                self.time_window = time_window
            self.samples = int(self.time_steps/self.time_window)
            self.XData = self.XData.reshape((self.samples,self.time_window,self.num_features))
            
            self.YData = self.YData[(self.time_window - 1)::self.time_window].reshape((1,self.samples))[0]
            
    def Calc_Hurst(self, plot = 'N'):
        ### See wikipedia rescaled range
        self.Xtrans = self.XData.transpose()
        self.data = []
        self.mean = []
        self.mean_adj_values = []
        self.mean_adj_series = []
        self.cumulative_deviate_series = []
        self.range_series = []
        self.standard_deviation_series = []
        self.RS = []
        self.logs = np.log(np.asarray([i for i in range(0,len(self.Xtrans[0]))]))
        self.log_value = []
        self.Hurst = []
        for i in self.Xtrans:
            self.mean.append(np.average(i))
        for i in range(len(self.Xtrans)):
            self.mean_adj_values.append(self.Xtrans[i] - self.mean[i])
        for j in range(len(self.Xtrans)):
            self.mean_adj_series.append([])
            for i in range(1,len(self.mean_adj_values[0])+1):
                self.mean_adj_series[-1].append(self.mean_adj_values[j][0:i])
        for i in range(len(self.mean_adj_series)):
            self.cumulative_deviate_series.append([])
            for j in range(len(self.mean_adj_series[i])):
                self.cumulative_deviate_series[-1].append(sum(self.mean_adj_series[i][j]))
        for i in range(len(self.cumulative_deviate_series)):
            self.range_series.append([])
            for j in range(len(self.cumulative_deviate_series[i])):
                self.range_series[-1].append((max(self.cumulative_deviate_series[i][0:j+1])) - min(self.cumulative_deviate_series[i][0:j+1]))
        for i in self.Xtrans:
            self.standard_deviation_series.append([])
            for j in range(len(i)):
                self.standard_deviation_series[-1].append(np.std(np.asarray(i[:j+1])))
        for i in range(len(self.range_series)):
            self.RS.append([])
            for j in range(len(self.range_series[i])):
                if self.standard_deviation_series[i][j] != 0.0:
                    self.RS[-1].append((self.range_series[i][j])/(self.standard_deviation_series[i][j]))
                if self.standard_deviation_series[i][j] == 0.0:
                    self.RS[-1].append(self.range_series[i][j])
        for i in range(len(self.Xtrans)):
            self.log_value.append(np.log(np.asarray(self.standard_deviation_series[i])))
        for i in range(len(self.Xtrans)):
            self.Hurst.append(np.polyfit(self.logs[1:], self.log_value[i][1:], 1))
        if plot == 'Y':
            for i in range(len(self.Xtrans)):
                plt.plot(self.logs[1:], self.log_value[i][1:])
            plt.legend([str(i) for i in list(np.asarray(self.Hurst)[:,0])], loc = 'upper right')
            plt.show()
            print("Hurst Values: "+str(np.asarray(self.Hurst)[:,0]))
    
    def histogram(self, num_bins, display = 'N'):
        self.num_bins = num_bins
        self.Xtrans = self.XData.transpose()
        self.bins = []
        for i in range(len(self.Xtrans)):
            self.little_bins = []
            self.Del = (max(self.Xtrans[i]) - min(self.Xtrans[i]))
            self.bin_del = self.Del / self.num_bins
            self.min = min(self.Xtrans[i])
            self.bin_floor = min(self.Xtrans[i])
            while self.bin_floor < max(self.Xtrans[i]):
                self.bin_count = 0
                for k in self.Xtrans[i]:
                    if (k > self.bin_floor) and (k <= (self.bin_floor + self.bin_del)):
                        self.bin_count += 1
                self.little_bins.append(np.asarray([self.bin_floor, self.bin_count]))
                self.bin_floor += self.bin_del
            self.bins.append(np.asarray(self.little_bins))
        self.bins_array = np.asarray(self.bins)
        self.bins_prob = []
        for i in range(len(self.bins_array)):
            self.feat_prob = []
            self.total = np.sum(self.bins_array[i][:,1])
            for j in range(len(self.bins_array[i][:,1])):
                self.feat_prob.append(self.bins_array[i][:,1][j]/self.total)
            self.bins_prob.append(self.feat_prob)
        self.S_Shannon = []
        self.S = 0.0
        for i in range(len(self.bins_prob)):
            self.S = 0.0
            for j in self.bins_prob[i]:
                if j != 0.0:
                    self.S += j*np.log(j)
            self.S_Shannon.append(-self.S)
        if display == 'Y':
            print("Shannon Entropy of Features : "+str(self.S_Shannon))
            
    def embed(self,time_window = 4):
        self.num_features = int(self.XData.shape[1])
        self.time_steps = self.XData.shape[0]
        
        self.time_window = time_window
        
        self.embed_data = self.XData[(self.time_steps%self.time_window):]
        self.samples = int(self.time_steps/self.time_window)
        self.embed_data = self.embed_data.reshape((self.samples,self.time_window,self.num_features))
        self.CG_rep = []
        for i in self.embed_data:
            self.CG_rep.append([np.average(i)])
            
    def permutation(self, tech = 'standard'):
        ### Standard method uses Bandt and Pompe (2002) permutation entropy approach
        self.permutation_bin = []        
        self.permutations = []
        if tech == 'standard':
            self.perm_types = list(it.permutations(range(0,self.embed_data.shape[1])))
            for i in range(len(list(self.embed_data))):
                self.tri = perm.embed_vector(self.embed_data[i])
                self.tri.permutation()
                self.permutations.append(tuple(self.tri.perm))
        for i in self.perm_types:
            self.permutation_bin.append([i, 0])
            for j in self.permutations:
                if j == i:
                    self.permutation_bin[-1][-1] += 1
        self.total = 0
        for i in self.permutation_bin:
            self.total += i[-1]
        self.perm_probs = []
        for i in self.permutation_bin:
            self.perm_probs.append([i[0], i[1]/self.total])
        self.PE = 0.0
        for i in self.perm_probs:
            if i[-1] != 0.0:
                self.PE += -(i[-1])*np.log(i[-1])

        
            
                
        
                
            
    
        
            


    def prepare(self):
        pass





class stock_day:
    def __init__(self, atts = { 'index' : 0, 'date' : 'N/A', 'close' : 0.0, 'volume' : 0.0, 'open' : 0.0, 'high' : 0.0, 'low' : 0.0}):
        self.atts = atts
        self.atts.update( { 'M' : (self.atts['close'] + self.atts['high'] + self.atts['low'])/3})
        
    def inner_list(self, stock_list):
        self.stock_list = stock_list
        
    def past(self, n):      
        self.past_range = []
        self.past_indices = [(j + 1) for j in range(n)]
        self.past_indices.reverse()
        self.past_length = n
        for i in self.past_indices:
            self.past_range.append(self.stock_list.data[self.atts['index'] - i])
        self.past_range.reverse()
        
    def next_day(self):
        self.future_range = []
        self.future_range.append(self.stock_list.data[self.atts['index'] + 1 ])
        
    def next_day_class(self):
        self.atts.update( { 'future_class' : self.future_range[0].atts['class']})
        self.atts.update( { 'future_close' : self.future_range[0].atts['close']})


    def stoch_k(self, n):
        ### stochastic K, %K -- S.B. Achelis, Technical Analysis from A to Z, Probus Publishing, Chicago, 1995. (Ref 1)
        ###      C_t - LL_t
        ### ---------------------   where C_t is closing at t, LL is lowest low, HH is highest high in last n days
        ### HH_(t-n) - LL_(t - n)
        self.past_lows = []
        self.past_highs = []        
        for i in self.past_range:
            self.past_lows.append(i.atts['low'])
            self.past_highs.append(i.atts['high'])
        self.atts.update( {'lowest_low' : min(self.past_lows)} )
        self.atts.update( {'highest_high' : max(self.past_highs)} )        
        self.atts.update( {'%k' : (self.atts['close'] - self.atts['lowest_low'])/(self.atts['highest_high'] - self.atts['lowest_low'])*100 } )        
   
    def stoch_d(self, n):
        ### stochastic D, %D -- (Ref 1)
        ### Moving average of %K over window length n      
        self.past_perK = []
        for i in self.past_range:
            self.past_perK.append(i.atts['%k'])
        self.atts.update({ '%D' : np.average(self.past_perK)})
        
    def slow_stoch_d(self, n):
        ### Slow Stochastic D, slow%D -- E. GiEord, Investor’s Guide to Technical Analysis: Predicting Price Action in the Markets, Pitman
        ### Publishing, London, 1995. (Ref 2)
        ### Moving average of %D        
        self.past_perD = []
        for i in self.past_range:
            self.past_perD.append(i.atts['%D'])
        self.atts.update( { 'slow%D' : np.average(self.past_perD) } )
        
    def momentum(self):
        ### Momentum, momentum -- J. Chang, Y. Jung, K. Yeon, J. Jun, D. Shin, H. Kim, Technical Indicators and Analysis Methods,
        ### Jinritamgu Publishing, Seoul, 1996. (Ref 3)
        ### C_t - C_(t-n)
        ### Metric of first difference
        self.atts.update({ 'momentum' : (self.atts['close'] - self.past_range[-1].atts['close']) })
        
    def price_ROC(self):
        ### Price Rate-of-Change, ROC -- J.J. Murphy, Technical Analysis of the Futures Markets: A Comprehensive Guide to Trading Methods
        ### and Applications, Prentice-Hall, New York, 1986. (Ref 4)
        ###  C_t
        ### ----- x 100
        ### C_t-n
        ### Rate of change relative to a point n days in the past
        self.atts.update( { 'ROC' : (self.atts['close']/self.past_range[-1].atts['close'])*100 } )
        
    def will_per_R(self):
         ### Williams' %R, %R 
         ###  H_n - C_t
         ### ----------- x 100  
         ###  H_n - L_n
         self.atts.update( { '%R' : 100*( (self.past_range[-1].atts['high'] - self.atts['close']) / (self.past_range[-1].atts['high'] - self.past_range[-1].atts['low']) ) })
         
    def ad_oscillator(self):
        ### AD Oscillator, AD
        ###  H_t - C_t-1
        ### ----------- x 100  
        ###  H_t - L_t
        self.atts.update( { 'AD' : ( self.atts['high'] - self.past_range[0].atts['close'] ) / ( self.atts['high'] - self.atts['low']) } )


    def w5_avg(self):
       ### 5 Average, 5Mu
       ### Rolling average of last 5 values
       self.values = []
       for i in range(0,5):
           self.values.append(self.past_range[i].atts['close'])          
       self.atts.update( { '5Mu' : np.average(self.values)})
       
    def w10_avg(self):
       ### 10 Average, 5Mu
       ### Rolling average of last 5 values
       self.values = []
       for i in range(0,10):
           self.values.append(self.past_range[i].atts['close'])          
       self.atts.update( { '10Mu' : np.average(self.values)})
       
    def oscp(self):
        ### Price Oscillator, OSCP
        ### 5Mu - 10Mu
        ### ----------
        ###    5Mu
        self.atts.update( { 'OSCP' : ((self.atts['5Mu'] - self.atts['10Mu']) / self.atts['5Mu'])})
        
    def cci(self):
        ### Commedy channel index, CCI
        
        self.Ms = []
        for i in self.past_range:
            self.Ms.append(i.atts['M'])
        self.Ms.append(self.atts['M'])            
        self.atts.update( { 'SM' : np.average(self.Ms)/(len(self.past_range)+1) } )
        self.Ds = []
        for i in self.past_range:
            self.Ds.append( abs(i.atts['M'] - self.atts['SM']) )
        self.Ds.append( abs(self.atts['M'] - self.atts['SM']))
        self.atts.update( {'D' : np.average(self.Ds)})
        self.atts.update( { 'cci' : (self.atts['M'] - self.atts['SM'])/(0.015 * self.atts['D']) } )
        
    def growth_loss(self):
        ### Determine growth or loss
        self.atts.update({ 'delta' : (self.atts['close'] - self.past_range[0].atts['close'])})
        if self.atts['delta'] > 0:
            self.atts.update( {'growth' : self.atts['delta']})
            self.atts.update( {'loss' : 0.0 })
            self.atts.update( {'class' : 1 })
        elif self.atts['delta'] < 0:
            self.atts.update( { 'growth' : 0.0 } )
            self.atts.update( { 'loss' : abs(self.atts['delta'])})
            self.atts.update( {'class' : 0 })
        else:
            self.atts.update( { 'growth' : 0.0 } )
            self.atts.update( { 'loss' : 0.0 } )
            self.atts.update( {'class' : 0 })
        
    def rsi(self):
        ### Relative strength index
        self.growths = []
        self.losses = []
        for i in self.past_range:
            self.growths.append(i.atts['growth'])
            self.losses.append(i.atts['loss'])
        self.losses.append(self.atts['loss'])
        self.growths.append(self.atts['growth'])
        self.atts.update({ 'rs' : (np.average(self.growths)/np.average(self.losses))})
        self.atts.update( { 'rsi' : 100*(1 - (1/(1+self.atts['rs'])))})
        
                                                                                        
        
        

class stock_list:   
    ### it is important the index 0-N of data be chronological (oldest -> 0, most recent -> N).     
    ### 'data' should be in chronological order
    ### I do not think there is a strong reason to believe
    ### financial physics would obey time-reversal symmetry
    def __init__(self, data = []):
        self.data = []
        for i in data:
            self.data.append(i)
        for i in data:
            i.inner_list(self)
        self.size = len(self.data)
    
    def initialize_past(self, n):
        ### initializes the past_range variables for all stock days up to n days in the past
        for i in range(n, self.size):
            self.data[i].past(n)
    
    def initialize_metrics(self, n):
        ### 1st Order Difference Metrics
        for i in range(n, self.size):           
            self.data[i].stoch_k(n)
            self.data[i].momentum()
            self.data[i].price_ROC()
            self.data[i].will_per_R()
            self.data[i].w5_avg()
            self.data[i].w10_avg()
            self.data[i].ad_oscillator()
            self.data[i].oscp()
            self.data[i].cci()
            self.data[i].growth_loss()
        ### Initialize Future    
        for i in range(n, self.size - 1):
            self.data[i].next_day()
            self.data[i].next_day_class()
            
            
        
        ### 2nd Order Difference Metrics
        ### RSI is not technically 2nd Order but it needs to go here or it won't work
            
        for i in range(2*n, self.size):
            self.data[i].stoch_d(n)
            self.data[i].rsi()
            
        ### 3rd Order Difference Metric
        for i in range(3*n, self.size):
            self.data[i].slow_stoch_d(n)
            
    def slice_dates(self, start_date, end_date):
        self.start_ind = 'NA'
        self.end_ind = 'NA'
        self.start_date = [int(start_date[0:1]), int(start_date[3:4]), int(start_date[6:9])]
        self.end_date = [int(end_date[0:1]), int(end_date[3:4]), int(end_date[6:9])]
        while (self.start_ind == 'NA' or self.end_ind == 'NA'):
            for i in range(len(self.data)):
                if (str(self.start_date[0])+"/"+str(self.start_date[1])+"/"+str(self.start_date[2])) == self.data[i].atts['date']:
                    print(start_date+" "+str(i))
                    self.start_ind = i
                if (str(self.end_date[0])+"/"+str(self.end_date[1])+"/"+str(self.end_date[2])) == self.data[i].atts['date']:
                    print(end_date+" "+str(i))
                    self.end_ind = i
                start_date[1] += 1
                end_date[1] += 1
                
            print(str(self.start_ind)+" "+str(self.end_ind))
        return stock_list(data = self.data[self.start_ind:self.end_ind])
            
            
    def return_production(self,n):
        return self.data[3*n:self.size-1]
    

    
###### Models

class stock_SVM:
    def __init__(self, X_train, Y_train, X_test, Y_test, index=0, C=1.0, gamma=1.0):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.index = index
        self.C = C
        self.gamma = gamma
        self.model = svm.SVC(C=C, kernel='rbf', gamma=self.gamma)
        self.model.fit(self.X_train, self.Y_train)
        self.test_preds = self.model.predict(self.X_test)
        
    def metrics(self):
        self.residues = []
        self.residuals = []
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0
        for i in range(len(self.test_preds)):
            self.residues.append(abs(self.test_preds[i] - self.Y_test[i]))
            self.residuals.append(self.test_preds[i]-self.Y_test[i])
            if self.test_preds[i] == 1 and self.Y_test[i] == 1:
                self.TP += 1
            elif self.test_preds[i] == 1 and self.Y_test[i] == 0:
                self.FP += 1
            elif self.test_preds[i] == 0 and self.Y_test[i] == 0:
                self.TN += 1
            elif self.test_preds[i] == 0 and self.Y_test[i] == 1:
                self.FN += 1
            else:
                pass
        try:
            self.MCC = (self.TP * self.TN - self.FP * self.FN) / ((self.TP + self.FP)*(self.TP + self.FN)*(self.TN + self.FP)*(self.TN + self.FN))**(1/2)
        except:
            self.MCC = (self.TP * self.TN - self.FP * self.FN) 
        self.acc = (1 - (sum(self.residues)/len(self.Y_test)))
        
    def predict(self, XToday):
        self.XToday = XToday
        return self.model.predict(self.XToday)
    
class stock_DNN:
    def __init__(self, X_train, Y_train, X_test, Y_test, arch = [10,10,10], acti = 'relu', epochs = 100):
        self.X_train = X_train
        self.Y_train = to_categorical(Y_train)
        self.X_test = X_test
        self.Y_test = Y_test
        self.arch = arch
        self.acti = acti
        self.model = Sequential()
        self.flag = 0
        self.epochs = 100
        for i in self.arch:
            if self.flag == 0:
                self.model.add(Dense(i, activation = self.acti, input_dim = len(X_train[-1])))
                self.flag = 1
            else:
                self.model.add(Dense(i, activation = self.acti))
        self.model.add(Dense(2, activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(self.X_train, self.Y_train, epochs = self.epochs, verbose = 0)
    def metrics(self):
        self.fuzzy_preds = self.model.predict(self.X_test)
        self.test_preds = []
        for i in self.fuzzy_preds:
            if i[0] > i[1]:
                self.test_preds.append(0)
            else:
                self.test_preds.append(1)
        self.residues = []
        self.residuals = []
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0
        for i in range(len(self.test_preds)):
            self.residues.append(abs(self.test_preds[i] - self.Y_test[i]))
            self.residuals.append(self.test_preds[i]-self.Y_test[i])
            if self.test_preds[i] == 1 and self.Y_test[i] == 1:
                self.TP += 1
            elif self.test_preds[i] == 1 and self.Y_test[i] == 0:
                self.FP += 1
            elif self.test_preds[i] == 0 and self.Y_test[i] == 0:
                self.TN += 1
            elif self.test_preds[i] == 0 and self.Y_test[i] == 1:
                self.FN += 1
            else:
                pass
        try:
            self.MCC = (self.TP * self.TN - self.FP * self.FN) / ((self.TP + self.FP)*(self.TP + self.FN)*(self.TN + self.FP)*(self.TN + self.FN))**(1/2)
        except:
            self.MCC = (self.TP * self.TN - self.FP * self.FN) 
        self.acc = (1 - (sum(self.residues)/len(self.Y_test)))
        
class stock_LSTM:
    def __init__(self, X_train, Y_train, epochs = 100, cells = 8, plot = 'N'):
        self.X_train = X_train
        self.Y_train = Y_train
        self.epochs = epochs
        self.arch = [cells, 1]
        self.model = Sequential()
        self.model.add(LSTM(cells, input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.fit(self.X_train, self.Y_train, epochs=epochs, verbose=2)
        self.train_fit = self.model.predict(self.X_train)
        if plot == 'Y':
            plt.scatter(self.Y_train, self.train_fit)
            plt.show()
            
    def metrics(self, X_test, Y_test, plot = 'N'):
        self.X_test = X_test
        self.Y_test = Y_test
        self.fuzzy_preds = self.model.predict(self.X_test)
        self.test_preds = []
        for i in self.fuzzy_preds:
            if i > 0.5:
                self.test_preds.append(1)
            else:
                self.test_preds.append(0)
        if plot == 'Y':
            plt.title("LSTM Metrics")
            plt.scatter(self.fuzzy_preds, self.Y_test)
        self.residues = []
        self.residuals = []
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0
        for i in range(len(self.test_preds)):
            self.residues.append(abs(self.test_preds[i] - self.Y_test[i]))
            self.residuals.append(self.test_preds[i]-self.Y_test[i])
            if self.test_preds[i] == 1 and self.Y_test[i] == 1:
                self.TP += 1
            elif self.test_preds[i] == 1 and self.Y_test[i] == 0:
                self.FP += 1
            elif self.test_preds[i] == 0 and self.Y_test[i] == 0:
                self.TN += 1
            elif self.test_preds[i] == 0 and self.Y_test[i] == 1:
                self.FN += 1
            else:
                pass
        try:
            self.MCC = (self.TP * self.TN - self.FP * self.FN) / ((self.TP + self.FP)*(self.TP + self.FN)*(self.TN + self.FP)*(self.TN + self.FN))**(1/2)
        except:
            self.MCC = (self.TP * self.TN - self.FP * self.FN) 
        self.acc = (1 - (sum(self.residues)/len(self.Y_test)))
        
    def ordain(self, TodayX):
        self.fuzzy_ordain = self.model.predict(TodayX)
        if self.fuzzy_ordain[0][0] >= 0.5:
            return 1
        else:
            return 0
        

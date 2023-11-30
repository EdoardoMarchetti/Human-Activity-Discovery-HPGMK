#___________________________________________________________________________________#
#   A Novel Skeleton-Based Human Activity Discovery
#   Technique Using Particle Swarm Optimization with
#   Gaussian Mutation
#
#                                                                                   #
#   Author and programmer: Parham Hadikhani, DTC Lai, WH Ong                             #
#                                                                                   #
#   e-Mail:20h8561@ubd.edu.bn, daphne.lai@ubd.edu.bn, weehong.ong@ubd.edu.bn        #   
#___________________________________________________________________________________#


import argparse
import os
import os.path as osp

import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()
from sklearn import metrics
from sklearn.metrics.cluster import adjusted_rand_score
import time
from HPGMK import *
from Evaluation import *
import csv
from feature_extraction import Extraction
from Preprocessing import *
import warnings
warnings.filterwarnings("ignore")



def run(args):

    #dataset=['CAD60','F3D','KARD','UTK','MSR']
    dataset= [args.dataset]
    to_extract = True

    for Datasetsss in dataset:
        
        if Datasetsss=='CAD60':
            maxsub=5
        else:
            maxsub=11
        
        if Datasetsss=='CAD60':
            activities=['brushing teeth','cooking (chopping)','Rinsing mouth with water','Still(standing)','Taking on the couch','Talking on the phone','Wearing contact lenses','Working on computer','writing on whiteboard','Drinking water','Cooking (stirring)','Opening pill container','Random','Relaxing on couch']    
            k=len(activities)
        if Datasetsss=='F3D':
            activities=['wave','drink from a bottle','answer phone','clap','tight lace','sit down','stand up', 'read watch', 'bow']
            k=len(activities)
        if Datasetsss=='KARD':
            activities=['Horizontal arm wave', 	'High arm wave',	'Two hand wave' ,	'Catch Cap' ,	'High throw' ,	'Draw X' ,	'Draw Tick' ,	'Toss Paper' ,	'Forward Kick' ,	'Side Kick' ,	'Take Umbrella' ,	'Bend' ,	'Hand Clap' ,	'Walk' ,	'Phone Call' ,	'Drink' ,	'Sit down' ,	'Stand up']
            k=len(activities)
        if Datasetsss=='UTK':
            activities=['walk', 'sit down', 'stand up', 'pick up', 'carry', 'throw', 'push', 'pull', 'wave hands', 'clap hands']
            k=len(activities)
        if Datasetsss=='MSR':
            activities=['drink', 'eat', 'read book', 'call cellphone', 'write on a paper', 'use laptop', 'use vacuum cleaner', 'cheer up', 'sit still', 'toss paper', 'play game', 'lie down on sofa', 'walk', 'play guitar', 'stand up', 'sit down']
            k=len(activities)
            
        print('Dataset: ****',Datasetsss,'****')
        
        for subject in range(1,maxsub):
            print('subject: /',subject,'\\')
            data = np.load(osp.join(args.root, args.data_folder, args.dataset,f"sub{subject}.npy"))

            data = data[:1000]
            print(data.shape)
            true_label = np.load(osp.join(args.root, args.data_folder, args.dataset,f"label{subject}.npy"))


            results_folder = osp.join(args.root, args.results_folder, args.dataset)
            print(results_folder)
            if not osp.exists(results_folder):
                os.makedirs(results_folder, exist_ok=True)
            
            #Extracted data
            extracted_features = osp.join(args.root, args.data_folder, args.dataset,f'extracted_features_{subject}.npy')
            extracted_keyframes = osp.join(args.root, args.data_folder, args.dataset,f'keyframes_{subject}.npy')
            if osp.exists(extracted_features):
                to_extract = False

            #Output files
            savefile1 = osp.join(results_folder,f'result_sub_{subject}.csv')
            savefile2 = osp.join(results_folder,f'fscore_sub_{subject}.csv')
            matcon = osp.join(results_folder,f'confusion_sub{subject}.png')
            timesave = osp.join(results_folder, f'time_sub{subject}.npy')
            history1plot = osp.join(results_folder, f'history_sub{subject}.npy')

            # data=np.load(r'%s'%save_path+'\Data\%s'%Datasetsss+'\sub%d'%subject+'.npy')
            # true_label=np.load(r'%s'%save_path+'\Data\%s'%Datasetsss+'\label%d'%subject+'.npy')
            if to_extract:
                print('Extracting features: ')
                data,keyframes=Extraction(data)
                print(keyframes.shape)
                data=np.nan_to_num(data)
                np.save(extracted_features, data)
                np.save(extracted_keyframes, keyframes)
            else:
                print('loading extracted data')
                data = np.load(extracted_features)
                keyframes = np.load(extracted_keyframes)

            print(data.shape, keyframes.shape)

            #save the results of clustering
            results_folder = osp.join(args.root, args.results_folder, args.dataset)         
            data=dimentaion_reduction(data)
            true_label=Keyframe(true_label,keyframes)
            data,ind=Sampling(data,15,true_label)
            data = data.reshape(data.shape[0], (data.shape[1]*data.shape[2])) 
            
            _matrix=[]
            _sse=[]
            _accuracy=[]
            _NMI=[]
            _adjusted_rand_score=[]
            _label=[]
            _bestsse=[]
            _time=[]
            _homogeneity_score=[]
            _completeness_score=[]
            _v_measure_score=[]
            _fowlkes_mallows_score=[]
                
            for i in range(30):
                start = time.time()
                HPGMK = ParticleSwarmOptimizedClustering(n_cluster=k, n_particles=20, data=data, hybrid=True)  
                history1,label1,S1=HPGMK.run()
                end = time.time()
                runtime=end - start
                _bestsse.append(S1)
                _sse.append(history1)
                lbl,label=map1(ind,label1)
                _label.append(label)
                _matrix.append(confusion_matrix(lbl, label))
                _NMI.append(NMI(lbl, label))
                _accuracy.append(accuracy(lbl, label))
                _adjusted_rand_score.append(adjusted_rand_score(lbl, label))
                _homogeneity_score.append(metrics.homogeneity_score(lbl, label))
                _completeness_score.append(metrics.completeness_score(lbl, label))
                _v_measure_score.append(metrics.v_measure_score(lbl, label))
                _fowlkes_mallows_score.append(metrics.fowlkes_mallows_score(lbl, label))
                _time.append(runtime)
        
                print('iteration: ',i,'------------------data: %s'%Datasetsss+'----subject: %d'%subject+'---------------------------------------')
            print('Dataset: ',Datasetsss,'****')
            _time=np.array(_time)
            _sse1=np.array(_sse[np.argmax(_accuracy)])
            np.save(timesave,_time)
            np.save(history1plot,_sse1)
            
            
            
            myFile = open(savefile1, 'w')
            with myFile:    
                myFields = ['metric','Mean', 'Min','Max']
                writer = csv.DictWriter(myFile, fieldnames=myFields)    
                writer.writeheader()
                writer.writerow({'metric':'sse','Mean': np.sum(_bestsse)/len(_bestsse), 'Min': np.min(_bestsse),'Max':np.max(_bestsse)})
                writer.writerow({'metric':'accuracy','Mean':np.sum(_accuracy)/len(_accuracy),'Min':np.min(_accuracy),'Max':np.max(_accuracy)})
                writer.writerow({'metric':'NMI','Mean':np.sum(_NMI)/len(_NMI),'Min':np.min(_NMI),'Max':np.max(_NMI)})
                writer.writerow({'metric':'homogeneity_score','Mean':np.sum(_homogeneity_score)/len(_homogeneity_score),'Min':np.min(_homogeneity_score),'Max': np.max(_homogeneity_score)})
                writer.writerow({'metric':'completeness_score','Mean':np.sum(_completeness_score)/len(_completeness_score),'Min':np.min(_completeness_score),'Max': np.max(_completeness_score)})
                writer.writerow({'metric':'v_measure_score','Mean':np.sum(_v_measure_score)/len(_v_measure_score),'Min':np.min(_v_measure_score),'Max': np.max(_v_measure_score)})
                writer.writerow({'metric':'fowlkes_mallows_score','Mean':np.sum(_fowlkes_mallows_score)/len(_fowlkes_mallows_score),'Min':np.min(_fowlkes_mallows_score),'Max': np.max(_fowlkes_mallows_score)})
                writer.writerow({'metric':'adjusted_rand_score','Mean':np.sum(_adjusted_rand_score)/len(_adjusted_rand_score),'Min':np.min(_adjusted_rand_score),'Max':np.max(_adjusted_rand_score)})
            
            flist=F_Score(ind,lbl,_label)
            
            f = open(savefile2, 'w')
            
            with f:
                writer = csv.writer(f)    
                writer.writerow(activities)
                writer.writerow(flist)
            
            conf_mat_draw(_matrix[np.argmax(_accuracy)],activities,matcon)
            
        
        print('-----------------')



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--root', dest='root', help='saving path', default = '.')
    parser.add_argument('--data_folder', dest = 'data_folder', help='folder containig dataset folders', default='Data')
    parser.add_argument('--results_folder', dest='results_folder', help='folder where to save the results', default='Results')

    parser.add_argument('--dataset', dest='dataset', help='dataset name', default='CAD60')


    args = parser.parse_args()

    run(args)
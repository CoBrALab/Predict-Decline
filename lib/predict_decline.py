#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import time
from datetime import datetime
from argparse import ArgumentParser
import os
import pickle

from lsn import *

if __name__ == "__main__":
    # parse user command-input arguments    
    parser = ArgumentParser(description="Wrapper to run the LSN code")
    parser.add_argument("-m", "--method", dest="method", type=str, default="LSN",
                        help="options: ANN, LR, RF, SVM, LSN, default=LSN")
    parser.add_argument("-f", "--features", dest="features", type=str, default="AAL",
                        help="options: AAL, PCA, RFE, RLR, default=AAL")
    parser.add_argument("-t", "--trajectory", dest="trajectory", type=str, default="MMSE",
                        help="options: MMSE, ADAS13, default=MMSE")
    parser.add_argument("-n", "--n_features", dest="n_features", type=int, default=78,
                        help="""number of features to be extracted 
                        (if feature type is not AAL) [default = %(default)s]""")
    parser.add_argument("-i", "--n_iterations", dest="n_iterations", type=int, default=10,
                        help="number of cross-validation iterations")
    parser.add_argument("--no_clinical", action="store_true", default=False)
    parser.add_argument("--no_followup", action="store_true", default=False)
    parser.add_argument("--only_clinical", action="store_true", default=False)
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate of the LSN [default = %(default)s]")
    parser.add_argument("--n_epochs", type=int, default=100,
                        help="number of epochs for training [default=%(default)s]")
    parser.add_argument("--validate_after", type=int, default=10,
                        help="validate after each n epochs [default=%(default)s]")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="size per gradient descent iteration [default=%(default)s]")
    parser.add_argument("--dropout", type=float, default=0.75,
                        help="keep probability during training [default=%(default)s]")
    parser.add_argument("--net_arch", type=int, nargs="+", default=[50, 50, 50, 50, 20, 10],
                        help="list of the sizes of hidden layers in the LSN default=%(default)s")
    parser.add_argument("--save_model", action="store_true", default=False)
    parser.add_argument("--grid_search", action="store_true", default=False)
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", default=False)
    parser.add_argument("-o", "--out_path", dest="out_path", type=str, default="output/")    
    opt = parser.parse_args()
    
    # performance metrics to be measured
    perf = []
    
    for i in range(opt.n_iterations):
        if opt.verbose:
            start_time = time.time()
            print("Cross-validation iteration {}".format(i))
            print("Loading data...")
    
        # load data and train-test splits specifications    
        sub_list = pd.read_csv("Exp_CL1_sub_list.csv")
        df = pd.read_csv("Exp_502_602_combined.csv")
        splits = pd.read_pickle("train_test_splits.pkl")
        train_split = splits["train"][i]
        test_split = splits["test"][i]
        ptid = sub_list["PTID"]
        train_ptid = ptid[train_split]
        test_ptid = ptid[test_split]
        df_train = df[df["PTID"].isin(train_ptid)]
        df_test = df[df["PTID"].isin(test_ptid)]
        print(sub_list.shape)
        print(df.shape)
        print(len(train_split))
        print(len(test_split))
        print(ptid.shape)
        print(train_ptid.shape)
        print(test_ptid.shape)
        print(df_train.shape)
        print(df_test.shape)
        
        # obtain the CT values grouped with ROI atlases
        if opt.features == "AAL":
            loc_bl = df_train.columns.get_loc("ACG.L_CT_bl")
            loc_var_tp = df_train.columns.get_loc("ACG.L_CT_var_tp")
            X_train_baseline = df_train.iloc[:, loc_bl:loc_bl+78].values
            X_train_followup = df_train.iloc[:, loc_var_tp:loc_var_tp+78].values
            X_test_baseline = df_test.iloc[:, loc_bl:loc_bl+78].values
            X_test_followup = df_test.iloc[:, loc_var_tp:loc_var_tp+78].values
            print(X_train_baseline.shape)
            print(X_train_followup.shape)
            print(X_test_baseline.shape)
            print(X_test_followup.shape)
        
        # create the reduced datasets with unsupervised feature selection
        elif opt.features == "PCA" or opt.features == "HCA":
            X_train_baseline = np.load("data/{}_bl_train_cv{}.npy".format(opt.features, i))
            X_train_followup = np.load("data/{}_vartp_train_cv{}.npy".format(opt.features, i))
            X_test_baseline = np.load("data/{}_bl_test_cv{}.npy".format(opt.features, i))
            X_test_followup = np.load("data/{}_vartp_test_cv{}.npy".format(opt.features, i))
            print(X_train_baseline.shape)
            print(X_train_followup.shape)
            print(X_test_baseline.shape)
            print(X_test_followup.shape)
        
        # create the reduced datasets with supervised feature selection
        elif opt.features == "RFE" or opt.features == "RLR":
            X_train_baseline = np.load("data/{}_bl_train_{}_cv{}.npy".format(opt.features, opt.trajectory, i))
            X_train_followup = np.load("data/{}_vartp_train_{}_cv{}.npy".format(opt.features, opt.trajectory, i))
            X_test_baseline = np.load("data/{}_bl_test_{}_cv{}.npy".format(opt.features, opt.trajectory, i))
            X_test_followup = np.load("data/{}_vartp_test_{}_cv{}.npy".format(opt.features, opt.trajectory, i))
            print(X_train_baseline.shape)
            print(X_train_followup.shape)
            print(X_test_baseline.shape)
            print(X_test_followup.shape)
    
        # load trajectory classes and auxiliary data
        if opt.trajectory == "MMSE":
            y_train_vector = df_train["MMSE_2c_traj"]
            y_test_vector = df_test["MMSE_2c_traj"]
            X_aux_train = df_train[["APOE4", "AGE", "MMSE_bl", "MMSE_var_tp"]]
            X_aux_test = df_test[["APOE4", "AGE", "MMSE_bl", "MMSE_var_tp"]]
        elif opt.trajectory == "ADAS13":
            y_train_vector = df_train["ADAS_3c_traj"]
            y_test_vector = df_test["ADAS_3c_traj"]
            X_aux_train = df_train[["APOE4", "AGE", "ADAS13_bl", "ADAS13_var_tp"]]
            X_aux_test = df_test[["APOE4", "AGE", "ADAS13_bl", "ADAS13_var_tp"]]
        print(y_train_vector.shape)
        print(y_test_vector.shape)
        print(X_aux_train.shape)
        print(X_aux_test.shape)
        print(type(X_aux_train))
        print(type(X_aux_test))
            
        if opt.verbose:
            delta_t = time.time() - start_time
            print("Time spent loading data: {}".format(delta_t))
            print("Ready to train the LSN model...")
        
        if opt.method == "LSN":    
            # get ready to train the LSN
            if opt.save_model:
                save_model_path = os.getcwd() + "/TF_trained_models/"
            MR_shape = X_train_baseline.shape[1]
            train_size = X_train_baseline.shape[0]
            test_size = X_test_baseline.shape[0]
            n_classes = int(np.amax(y_train_vector) + 1)
            X_MR_train = np.stack((X_train_baseline, X_train_followup), axis=1)
            X_MR_test = np.stack((X_test_baseline, X_test_followup), axis=1)
            y_train = np.zeros((train_size, n_classes))
            y_train[np.arange(train_size), y_train_vector] = 1
            y_test = np.zeros((test_size, n_classes))
            y_test[np.arange(test_size), y_test_vector] = 1
            X_aux_train = X_aux_train.as_matrix()
            X_aux_test = X_aux_test.as_matrix()
            print(X_MR_train.shape)
            print(X_MR_test.shape)
            print(y_train.shape)
            print(y_test.shape)
            print(X_aux_train.shape)
            print(X_aux_test.shape)
            
            # set the neural network architecture    
            net_arch = {'MR_shape':MR_shape,'n_layers':len(opt.net_arch)-1,'MR_output':opt.net_arch[-2],
                        'use_aux':True,'aux_shape':4,'aux_output':opt.net_arch[-2],'output':n_classes,'reg':0.01}
            for hidden_layer in range(len(opt.net_arch)-1):
                net_arch['l{}'.format(hidden_layer+1)] = opt.net_arch[hidden_layer]
            perf_df = pd.DataFrame(columns=['subject_id','label','pred_prob','pred_label'])
            print(net_arch)
            
            tf.reset_default_graph()
            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            
                # ----------------------Train model    -------------------------------
                data = {'X_MR':X_MR_train,'X_aux':X_aux_train,'y':y_train}    
                if check_data_shapes(data,net_arch):      
                    print('train data <-> net_arch check passed')  
                    lsn = siamese_net(net_arch)        
                    optimizer = tf.train.AdamOptimizer(learning_rate = opt.lr).minimize(lsn.loss)
                    tf.global_variables_initializer().run()
                    saver = tf.train.Saver()
                    
                    cur_time = datetime.time(datetime.now())
                    print('\nStart training time: {}'.format(cur_time))                
                    lsn, train_metrics = train_lsn(sess, lsn, data, optimizer, opt.n_epochs, opt.batch_size, 
                                                   opt.dropout, opt.validate_after, opt.verbose)
                    
                    # Save trained model 
                    if opt.save_model:
                        filename = "{}_{}_{}_{}".format(opt.features, opt.trajectory, MR_shape, i)
                        print('saving model at {}'.format("models/" + filename))
                        saver.save(sess, "models/" + filename)
                    
                    cur_time = datetime.time(datetime.now())
                    print('End training time: {}\n'.format(cur_time))  
                else:
                    print('train data <-> net_arch check failed')
        
                # Test model  (within same session)         
                data = {'X_MR':X_MR_test,'X_aux':X_aux_test,'y':y_test}
                if check_data_shapes(data,net_arch):
                    print('test data <-> net_arch check passed')   
                    _,test_metrics = test_lsn(sess,lsn,data)        
                    # populate perf dataframe
                    perf_df['subject_id']  = test_ptid
                    perf_df['label'] = np.argmax(y_test,1)
                    perf_df['pred_prob'] = list(test_metrics['test_preds'])
                    perf_df['pred_label'] = np.argmax(test_metrics['test_preds'],1)
                else:
                    print('test data <-> net_arch check failed')
                    
            # save model stats (accuracy, loss, performance)
            with open(opt.out_path + "{}_{}_{}_{}_train_metrics.pkl".format(
                        opt.features, opt.trajectory, MR_shape, i), "wb") as f:
                pickle.dump(train_metrics, f, pickle.HIGHEST_PROTOCOL)
            with open(opt.out_path + "{}_{}_{}_{}_test_metrics.pkl".format(
                        opt.features, opt.trajectory, MR_shape, i), "wb") as f:
                pickle.dump(test_metrics, f, pickle.HIGHEST_PROTOCOL)
            perf_df.to_csv(opt.out_path + "{}_{}_{}_{}_perf_df.csv".format(
                            opt.features, opt.trajectory, MR_shape, i))
            print(perf_df.shape)
            
            accuracy = test_metrics["test_acc"]
        
        else:
            # train a baseline classifier
            MR_shape = X_train_baseline.shape[1]
            train_size = X_train_baseline.shape[0]
            test_size = X_test_baseline.shape[0]
            n_classes = int(np.amax(y_train_vector) + 1)
            X_aux_train = X_aux_train.values
            X_aux_test = X_aux_test.values
            print(X_train_baseline.shape)
            print(X_train_followup.shape)
            print(X_aux_train.shape)
            
            if opt.no_followup and opt.no_clinical:
                X_train = X_train_baseline
                X_test = X_test_baseline
            elif opt.no_followup and opt.only_clinical:
                X_train = X_aux_train[:, :3]
                X_test = X_aux_test[:, :3]
            elif opt.no_followup:
                X_train = np.concatenate((X_train_baseline, X_aux_train), axis=1)
                X_test = np.concatenate((X_test_baseline, X_aux_test), axis=1)
            elif opt.no_clinical:
                X_train = np.concatenate((X_train_baseline, X_train_followup), axis=1)
                X_test = np.concatenate((X_test_baseline, X_test_followup), axis=1)
            elif opt.only_clinical:
                X_train = X_aux_train
                X_test = X_aux_test
            else: # if use both followup and clinical attributes
                X_train = np.concatenate((X_train_baseline, X_train_followup, X_aux_train), axis=1)
                X_test = np.concatenate((X_test_baseline, X_test_followup, X_aux_test), axis=1)
            print(X_train.shape)
            print(X_test.shape)
            print(y_train_vector.shape)
            print(y_test_vector.shape)
            print(X_aux_train.shape)
            print(X_aux_test.shape)
            
            if opt.method == "LR":
                lr_classifier = LogisticRegression(solver="lbfgs", multi_class="multinomial", verbose=opt.verbose)
                lr_classifier.fit(X_train, y_train_vector)
                accuracy = lr_classifier.score(X_test, y_test_vector)
                print(lr_classifier.classes_)
                
            if opt.method == "SVM":
                svm_classifier = SVC(gamma="scale", verbose=opt.verbose)
                svm_classifier.fit(X_train, y_train_vector)
                accuracy = svm_classifier.score(X_test, y_test_vector)
                
            if opt.method == "RF":
                rf_classifier = RandomForestClassifier(n_estimators=100, verbose=opt.verbose)
                rf_classifier.fit(X_train, y_train_vector)
                accuracy = rf_classifier.score(X_test, y_test_vector)
                print(rf_classifier.classes_)
                
            if opt.method == "ANN": # same parameters as the LSN
                hidden_layer_sizes = opt.net_arch[:-2]
                print(hidden_layer_sizes)
                ann_classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, verbose=opt.verbose, 
                                               alpha=0.01, learning_rate_init=opt.lr, batch_size=opt.batch_size)
                ann_classifier.fit(X_train, y_train_vector)
                accuracy = ann_classifier.score(X_test, y_test_vector)
                print(ann_classifier.classes_)
                
        perf.append(accuracy)
            
    #save cross-validation model performance
    if not opt.grid_search:
        with open(opt.out_path + "model_performance.csv", "a") as f:
            if os.stat(opt.out_path + "model_performance.csv").st_size == 0:
                f.write("Method,FeatureType,Trajectory,NumberOfFeatures,NoClinical,NoFollowup," +
                        "AccuracyMean,AccuracySD,Accuracy0,Accuracy1,Accuracy2,Accuracy3,Accuracy4," +
                        "Accuracy5,Accuracy6,Accuracy7,Accuracy8,Accuracy9\n")
            f.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    opt.method, opt.features, opt.trajectory, opt.n_features, opt.no_clinical, opt.no_followup, 
                    np.mean(perf), np.std(perf), perf[0], perf[1], perf[2], perf[3], perf[4], 
                    perf[5], perf[6], perf[7], perf[8], perf[9]))
                    
    else:
        with open(opt.out_path + "parameter_grid_search.csv", "a") as f:
            if os.stat(opt.out_path + "parameter_grid_search.csv").st_size == 0:
                f.write("Method,FeatureType,Trajectory,LearningRate,BatchSize,Dropout,Architecture" +
                        "AccuracyMean,AccuracySD,Accuracy0,Accuracy1,Accuracy2,Accuracy3,Accuracy4," +
                        "Accuracy5,Accuracy6,Accuracy7,Accuracy8,Accuracy9\n")
            f.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    opt.method, opt.features, opt.trajectory, opt.lr, opt.batch_size, opt.dropout,
                    " ".join([str(l) for l in opt.net_arch]), np.mean(perf), np.std(perf), perf[0], 
                    perf[1], perf[2], perf[3], perf[4], perf[5], perf[6], perf[7], perf[8], perf[9]))
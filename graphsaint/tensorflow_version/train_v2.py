#import sys
#sys.path.append('$HOME/workspace/GraphSAINT/')

from graphsaint.globals import *
from graphsaint.tensorflow_version.inits_v2 import *
from graphsaint.tensorflow_version.model_v2 import GraphSAINT
from graphsaint.tensorflow_version.minibatch_v2 import Minibatch
from graphsaint.utils import *
from graphsaint.metric import *
from tensorflow.python.client import timeline

import sys, os, random
import tensorflow as tf
#import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
#tf.compat.v1.enable_eager_execution()
import numpy as np
import time
import pdb
import json

import random
import threading
import psutil, GPUtil
import pandas as pd
from numba import cuda

ini_rc, ini_wc, ini_rb, ini_wb = psutil.disk_io_counters()[:4]
ini_bs, ini_br = psutil.net_io_counters()[:2]
interval = 0.01

def sample_metrics(unit="MB"):
    global ini_rc, ini_wc, ini_rb, ini_wb, ini_bs, ini_br
    weight = 1
    if unit == "MB":
        weight = 1024 * 1024
    elif unit == "GB":
        weight = 1024 * 1024 * 1024
    network_stat = psutil.net_io_counters()
    disk_io_stat = psutil.disk_io_counters()
    result = {
        "time": str(datetime.datetime.utcnow()),
        "cpu": psutil.cpu_percent(interval=interval),
        "percpu": np.mean(psutil.cpu_percent(interval=interval, percpu=True)),
        "mem": psutil.virtual_memory().used / weight,
        "ram": psutil.virtual_memory().active / weight,
        "disk": psutil.disk_usage("/").used / weight,
        "disk_io": {
            "rc": disk_io_stat[0] - ini_rc,
            "wc": disk_io_stat[1] - ini_wc,
            "rb": disk_io_stat[2] - ini_rb,
            "wb": disk_io_stat[3] - ini_wb
        },
        "network": {
            "sent": network_stat.bytes_sent - ini_bs,
            "recv": network_stat.bytes_recv - ini_br
        }
    }
    # if self._use_gpu:
    gpus = GPUtil.getGPUs()
    if len(gpus) > 0:
        result["gpu load"] = gpus[0].load * 100
        result["gpu memutil"] = gpus[0].memoryUtil * 100
    return result

def compute_metrics():
    global running
    running = True
    currentProcess = psutil.Process()
    lst = []
    # start loop
    while running:
        # *measure/store all needed metrics*
        lst.append(sample_metrics())
        time.sleep(interval)
    df = pd.DataFrame(lst)
    df.to_csv('demo_voc2007_gcn_metrics_2.csv', index=False)

def start():
    global t
    # create thread and start it
    t = threading.Thread(target=compute_metrics)
    t.start()

def stop():
    global running
    global t
    # use `running` to stop loop in thread so thread will end
    running = False
    # wait for thread's end
    t.join()

class TimeLiner:
    _timeline_dict = None

    def update_timeline(self, chrome_trace):
        # convert crome trace to python dict
        chrome_trace_dict = json.loads(chrome_trace)
        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict
        # for other - update only time consumption, not definitions
        else:
            for event in chrome_trace_dict['traceEvents']:
                # events time consumption started with 'ts' prefix
                if 'ts' in event:
                    self._timeline_dict['traceEvents'].append(event)

    def save(self, f_name):
        with open(f_name, 'w') as f:
            json.dump(self._timeline_dict, f)

def evaluate_full_batch(sess,model,minibatch_iter,many_runs_timeline,mode):
    """
    Full batch evaluation
    NOTE: HERE GCN RUNS THROUGH THE FULL GRAPH. HOWEVER, WE CALCULATE F1 SCORE
        FOR VALIDATION / TEST NODES ONLY. 
    """
    options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
    run_metadata = tf.compat.v1.RunMetadata()
    t1 = time.time()
    num_cls = minibatch_iter.class_arr.shape[-1]
    feed_dict, labels = minibatch_iter.feed_dict(mode)
    if args_global.timeline:
        preds,loss = sess.run([model.preds, model.loss], feed_dict=feed_dict, options=options, run_metadata=run_metadata)
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        many_runs_timeline.append(chrome_trace)
    else:
        preds,loss = sess.run([model.preds, model.loss], feed_dict=feed_dict)
    node_val_test = minibatch_iter.node_val if mode=='val' else minibatch_iter.node_test
    t2 = time.time()
    f1_scores = calc_f1(labels[node_val_test],preds[node_val_test],model.sigmoid_loss)
    return loss, f1_scores[0], f1_scores[1], (t2-t1)



def construct_placeholders(num_classes):
    placeholders = {
        'labels': tf.compat.v1.placeholder(DTYPE, shape=(None, num_classes), name='labels'),
        'node_subgraph': tf.compat.v1.placeholder(tf.int32, shape=(None), name='node_subgraph'),
        'dropout': tf.compat.v1.placeholder(DTYPE, shape=(None), name='dropout'),
        'adj_subgraph' : tf.compat.v1.sparse_placeholder(DTYPE,name='adj_subgraph',shape=(None,None)),
        'adj_subgraph_0' : tf.compat.v1.sparse_placeholder(DTYPE,name='adj_subgraph_0'),
        'adj_subgraph_1' : tf.compat.v1.sparse_placeholder(DTYPE,name='adj_subgraph_1'),
        'adj_subgraph_2' : tf.compat.v1.sparse_placeholder(DTYPE,name='adj_subgraph_2'),
        'adj_subgraph_3' : tf.compat.v1.sparse_placeholder(DTYPE,name='adj_subgraph_3'),
        'adj_subgraph_4' : tf.compat.v1.sparse_placeholder(DTYPE,name='adj_subgraph_4'),
        'adj_subgraph_5' : tf.compat.v1.sparse_placeholder(DTYPE,name='adj_subgraph_5'),
        'adj_subgraph_6' : tf.compat.v1.sparse_placeholder(DTYPE,name='adj_subgraph_6'),
        'adj_subgraph_7' : tf.compat.v1.sparse_placeholder(DTYPE,name='adj_subgraph_7'),
        'dim0_adj_sub' : tf.compat.v1.placeholder(tf.int64,shape=(None),name='dim0_adj_sub'),
        'norm_loss': tf.compat.v1.placeholder(DTYPE,shape=(None),name='norm_loss'),
        'is_train': tf.compat.v1.placeholder(tf.bool, shape=(None), name='is_train')
    }
    return placeholders





#########
# TRAIN #
#########
#@cuda.jit(device=True)
def prepare(train_data,train_params,arch_gcn):
    adj_full,adj_train,feats,class_arr,role = train_data
    adj_full = adj_full.astype(np.int32)
    adj_train = adj_train.astype(np.int32)
    adj_full_norm = adj_norm(adj_full)
    num_classes = class_arr.shape[1]

    placeholders = construct_placeholders(num_classes)
    minibatch = Minibatch(adj_full_norm, adj_train, role, class_arr, placeholders, train_params)
    model = GraphSAINT(num_classes, placeholders,
                feats, arch_gcn, train_params, adj_full_norm, logging=True)
    
    # Initialize session
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(device_count={"CPU":40},inter_op_parallelism_threads=44,intra_op_parallelism_threads=44,log_device_placement=args_global.log_device_placement))
    ph_misc_stat = {'val_f1_micro': tf.compat.v1.placeholder(DTYPE, shape=()),
                    'val_f1_macro': tf.compat.v1.placeholder(DTYPE, shape=()),
                    'train_f1_micro': tf.compat.v1.placeholder(DTYPE, shape=()),
                    'train_f1_macro': tf.compat.v1.placeholder(DTYPE, shape=()),
                    'time_per_epoch': tf.compat.v1.placeholder(DTYPE, shape=()),
                    'size_subgraph': tf.compat.v1.placeholder(tf.int32, shape=())}
    merged = tf.compat.v1.summary.merge_all()

    with tf.compat.v1.name_scope('summary'):
        _misc_val_f1_micro = tf.compat.v1.summary.scalar('val_f1_micro', ph_misc_stat['val_f1_micro'])
        _misc_val_f1_macro = tf.compat.v1.summary.scalar('val_f1_macro', ph_misc_stat['val_f1_macro'])
        _misc_train_f1_micro = tf.compat.v1.summary.scalar('train_f1_micro', ph_misc_stat['train_f1_micro'])
        _misc_train_f1_macro = tf.compat.v1.summary.scalar('train_f1_macro', ph_misc_stat['train_f1_macro'])
        _misc_time_per_epoch = tf.compat.v1.summary.scalar('time_per_epoch',ph_misc_stat['time_per_epoch'])
        _misc_size_subgraph = tf.compat.v1.summary.scalar('size_subgraph',ph_misc_stat['size_subgraph'])

    misc_stats = tf.compat.v1.summary.merge([_misc_val_f1_micro,_misc_val_f1_macro,_misc_train_f1_micro,_misc_train_f1_macro,
                    _misc_time_per_epoch,_misc_size_subgraph])
    summary_writer = tf.compat.v1.summary.FileWriter(log_dir(args_global.train_config,args_global.data_prefix,git_branch,git_rev,timestamp), sess.graph)
    # Init variables
    sess.run(tf.compat.v1.global_variables_initializer())
    return model,minibatch, sess, [merged,misc_stats],ph_misc_stat, summary_writer


#@cuda.jit(device=True)
def train(train_phases,model,minibatch,\
            sess,train_stat,ph_misc_stat,summary_writer):
    import time

    # saver = tf.train.Saver(var_list=tf.trainable_variables())
    saver=tf.compat.v1.train.Saver()

    epoch_ph_start = 0
    f1mic_best, e_best = 0, 0
    time_calc_f1, time_train, time_prepare, EPOCH_TIMES = 0, 0, 0, [] ###
    options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE,report_tensor_allocations_upon_oom=True)
    run_metadata = tf.compat.v1.RunMetadata()
    many_runs_timeline=[]       # only used when TF timeline is enabled
    for ip,phase in enumerate(train_phases):
        # We normally only have a single phase of training (see README for defn of 'phase').
        # On the other hand, our implementation does support multi-phase training. 
        # e.g., you can use smaller subgraphs during initial epochs and larger subgraphs
        #       when closer to convergence. -- This might speed up convergence. 
        minibatch.set_sampler(phase)
        num_batches = minibatch.num_training_batches()
        EPOCH_TIMES.append(str(datetime.datetime.utcnow()))
        printf('START PHASE {:4d}'.format(ip),style='underline')
        for e in range(epoch_ph_start,int(phase['end'])):
            printf('Epoch {:4d}'.format(e),style='bold')
            minibatch.shuffle()
            l_loss_tr, l_f1mic_tr, l_f1mac_tr, l_size_subg = [], [], [], []
            time_train_ep, time_prepare_ep = 0, 0
            while not minibatch.end():
                t0 = time.time()
                feed_dict, labels = minibatch.feed_dict(mode='train')
                t1 = time.time()
                if args_global.timeline:      # profile the code with Tensorflow Timeline
                    _,__,loss_train,pred_train = sess.run([train_stat[0], \
                            model.opt_op, model.loss, model.preds], feed_dict=feed_dict, \
                            options=options, run_metadata=run_metadata)
                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    many_runs_timeline.append(chrome_trace)
                else:
                    _,__,loss_train,pred_train = sess.run([train_stat[0], \
                            model.opt_op, model.loss, model.preds], feed_dict=feed_dict, \
                            options=tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom=True))
                t2 = time.time()
                time_train_ep += t2-t1
                time_prepare_ep += t1-t0
                if not minibatch.batch_num % args_global.eval_train_every:
                    f1_mic,f1_mac = calc_f1(labels,pred_train,model.sigmoid_loss)
                    l_loss_tr.append(loss_train)
                    l_f1mic_tr.append(f1_mic)
                    l_f1mac_tr.append(f1_mac)
                    l_size_subg.append(minibatch.size_subgraph)
            EPOCH_TIMES.append(str(datetime.datetime.utcnow())) ###
            time_train += time_train_ep
            time_prepare += time_prepare_ep
            if args_global.cpu_eval:      # Full batch evaluation using CPU
                # we have to start a new session so that CPU can perform full-batch eval.
                # current model params are communicated to the new session via tmp.chkpt
                saver.save(sess,'./tmp.chkpt')
                with tf.device('/cpu:0'):
                    sess_cpu = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(device_count={'GPU': 0}))
                    sess_cpu.run(tf.compat.v1.global_variables_initializer())
                    saver = tf.compat.v1.train.Saver()
                    saver.restore(sess_cpu, './tmp.chkpt')
                    sess_eval=sess_cpu
            else:
                sess_eval=sess
            loss_val,f1mic_val,f1mac_val,time_eval = \
                evaluate_full_batch(sess_eval,model,minibatch,many_runs_timeline,mode='val')
            printf(' TRAIN (Ep avg): loss = {:.4f}\tmic = {:.4f}\tmac = {:.4f}\ttrain time = {:.4f} sec'.format(f_mean(l_loss_tr),f_mean(l_f1mic_tr),f_mean(l_f1mac_tr),time_train_ep))
            printf(' VALIDATION:     loss = {:.4f}\tmic = {:.4f}\tmac = {:.4f}'.format(loss_val,f1mic_val,f1mac_val),style='yellow')
            if f1mic_val > f1mic_best:
                f1mic_best, e_best = f1mic_val, e
                if not os.path.exists(args_global.dir_log+'/models'):
                    os.makedirs(args_global.dir_log+'/models')
                print('  Saving models ...')
                savepath = saver.save(sess, '{}/models/saved_model_{}.chkpt'.format(args_global.dir_log,timestamp),write_meta_graph=False,write_state=False)
 
            if args_global.tensorboard:
                misc_stat = sess.run([train_stat[1]],feed_dict={\
                                        ph_misc_stat['val_f1_micro']: f1mic_val,
                                        ph_misc_stat['val_f1_macro']: f1mac_val,
                                        ph_misc_stat['train_f1_micro']: f_mean(l_f1mic_tr),
                                        ph_misc_stat['train_f1_macro']: f_mean(l_f1mac_tr),
                                        ph_misc_stat['time_per_epoch']: time_train_ep+time_prepare_ep,
                                        ph_misc_stat['size_subgraph']: f_mean(l_size_subg)})
                # tensorboard visualization
                summary_writer.add_summary(_, e)
                summary_writer.add_summary(misc_stat[0], e)
        epoch_ph_start = int(phase['end'])
    printf("Optimization Finished!",style='yellow')
    timelines = TimeLiner()
    for tl in many_runs_timeline:
        timelines.update_timeline(tl)
    timelines.save('timeline.json')
    saver.restore(sess_eval, '{}/models/saved_model_{}.chkpt'.format(args_global.dir_log,timestamp))
    loss_val, f1mic_val, f1mac_val, duration = evaluate_full_batch(sess_eval,model,minibatch,many_runs_timeline,mode='val')
    printf("Full validation (Epoch {:4d}): \n  F1_Micro = {:.4f}\tF1_Macro = {:.4f}".format(e_best,f1mic_val,f1mac_val),style='red')
    loss_test, f1mic_test, f1mac_test, duration = evaluate_full_batch(sess_eval,model,minibatch,many_runs_timeline,mode='test')
    printf("Full test stats: \n  F1_Micro = {:.4f}\tF1_Macro = {:.4f}".format(f1mic_test,f1mac_test),style='red')
    printf('Total training time: {:6.2f} sec'.format(time_train),style='red')
    print('Epoch timestamps: ', repr(EPOCH_TIMES)) ###
    #ret = {'loss_val_opt':loss_val,'f1mic_val_opt':f1mic_val,'f1mac_val_opt':f1mac_val,\
    #        'loss_test_opt':loss_test,'f1mic_test_opt':f1mic_test,'f1mac_test_opt':f1mac_test,\
    #        'epoch_best':e_best,
    #        'time_train': time_train}
    return      # everything is logged by TF. no need to return anything


########
# MAIN #
########

#@cuda.jit
def train_main(argv=None):
    train_params,train_phases,train_data,arch_gcn = parse_n_prepare(args_global)
    model,minibatch,sess,train_stat,ph_misc_stat,summary_writer = prepare(train_data,train_params,arch_gcn)
    ret = train(train_phases,model,minibatch,sess,train_stat,ph_misc_stat,summary_writer)
    return ret


if __name__ == '__main__':
    #threadsperblock = 32
    #blockspergrid = 300000000 #(an_array.size + (threadsperblock - 1)) // threadsperblock
    start()
    try:
        tf.compat.v1.app.run(main=train_main)#[blockspergrid, threadsperblock])
    finally:
        stop()


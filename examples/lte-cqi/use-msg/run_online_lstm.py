# run_online_lstm.py
# 
# Example for combining ns3-ai with an LSTM predictor to adjust CQI
# in a multi-flow, multi-enb environment (like 5G-V2X) scenario.
# 
# Copyright (c) 2019-2023 Huazhong ...
# Licensed under GPLv2
#
# Modified by: (your name or your lab)
# for demonstration purpose

import numpy as np
import tensorflow as tf
import keras
from keras.layers import *
import sys
import gc
import keras.backend as K
import ns3ai_ltecqi_py as py_binding
from ns3ai_utils import Experiment
import traceback

#-------------------
# delta: 0 => no LSTM
#        >0 => use LSTM
#-------------------
delta = int(sys.argv[1])
not_train = (delta == 0)

def new_print(filename="log", print_screen=False):
    old_print = print

    def print_fun(s):
        if print_screen:
            old_print(s)
        with open(filename, "a+") as f:
            f.write(s)
            f.write('\n')

    return print_fun

old_print = print
print = new_print(filename="log_" + str(delta), print_screen=False)

tf.random.set_seed(0)
np.random.seed(1)

# LSTM hyper-params
input_len = 200
pred_len = 40
batch_size = 20
alpha = 0.6

lstm_input_vec = Input(shape=(input_len, 1), name="input_vec")
dense1 = Dense(30, activation='selu', kernel_regularizer='l1')(lstm_input_vec[:, :, 0])
lstm_l1_mse = tf.keras.ops.expand_dims(dense1, axis=-1)
lstm_mse = LSTM(20)(lstm_l1_mse)
predict_lstm_mse = Dense(1)(lstm_mse)
lstm_model_mse = keras.Model(inputs=lstm_input_vec, outputs=predict_lstm_mse)
lstm_model_mse.compile(optimizer="adam", loss="MSE")

def simple_MSE(y_pred, y_true):
    return ((y_pred - y_true)**2).mean()

def weighted_MSE(y_pred, y_true):
    # example weighting
    return (((y_pred - y_true)**2) * (1 + np.arange(len(y_pred))) / len(y_pred)).mean()

# Data containers
cqi_queue = []
prediction = []
last = []
right = []  # 用於記錄每次預測成功(1) or 失敗(0)
corrected_predict = []
target = []
train_data = []
delay_queue = []

# 啟動 ns3-ai
exp = Experiment("ns3ai_ltecqi_msg", "../../../../../", py_binding, handleFinish=True)
msgInterface = exp.run(show_output=True)

try:
    while True:
        msgInterface.PyRecvBegin()
        if msgInterface.PyGetFinished():
            break
        gc.collect()

        CQI = msgInterface.GetCpp2PyStruct().wbCqi
        msgInterface.PyRecvEnd()

        if CQI > 15:
            # just a safety check
            break

        old_print("get: %d" % CQI)

        #---------------------------------
        # simulate feedback delay delta
        #---------------------------------
        delay_queue.append(CQI)
        if len(delay_queue) < delta:
            CQI_for_prediction = delay_queue[-1]
        else:
            CQI_for_prediction = delay_queue[-delta]

        #---------------------------------
        # If no LSTM, just echo back
        #---------------------------------
        if not_train:
            msgInterface.PySendBegin()
            msgInterface.GetPy2CppStruct().new_wbCqi = CQI_for_prediction
            msgInterface.PySendEnd()
            continue

        #---------------------------------
        # LSTM-based
        #---------------------------------
        cqi_queue.append(CQI_for_prediction)
        if len(cqi_queue) >= input_len + delta:
            target.append(CQI_for_prediction)

        if len(cqi_queue) >= input_len:
            one_data = cqi_queue[-input_len:]
            train_data.append(one_data)
        else:
            # not enough data for LSTM
            msgInterface.PySendBegin()
            msgInterface.GetPy2CppStruct().new_wbCqi = CQI_for_prediction
            msgInterface.PySendEnd()
            old_print("set: %d" % CQI_for_prediction)
            continue

        # predict
        data_to_pred = np.array(one_data).reshape(-1, input_len, 1) / 10.0
        _predict_cqi = lstm_model_mse.predict(data_to_pred)
        predict_cqi_int = int(_predict_cqi[0, 0] + 0.49995)
        old_print(f"pred = {_predict_cqi[0,0]} => {predict_cqi_int}")

        prediction.append(predict_cqi_int)
        last.append(one_data[-1])
        corrected_predict.append(predict_cqi_int)

        # online train logic
        if len(train_data) >= pred_len + delta:
            err_t = weighted_MSE(
                np.array(last[-(pred_len + delta):-delta]),
                np.array(target[-pred_len:])
            )
            err_p = weighted_MSE(
                np.array(prediction[-(pred_len + delta):-delta]),
                np.array(target[-pred_len:])
            )
            if err_p <= err_t * alpha:
                # good enough => 預測成功
                if err_t < 1e-6:
                    corrected_predict[-1] = last[-1]
                print(" ")
                print("OK %d %f %f" % (len(cqi_queue), err_t, err_p))
                right.append(1)
            else:
                # 預測失敗
                corrected_predict[-1] = last[-1]
                if err_t <= 1e-6:
                    msgInterface.PySendBegin()
                    msgInterface.GetPy2CppStruct().new_wbCqi = CQI_for_prediction
                    msgInterface.PySendEnd()
                    print("set: %d" % CQI_for_prediction)
                    continue
                else:
                    print("train %d" % (len(cqi_queue)))
                    right.append(0)
                    x_train = np.array(train_data[-delta - batch_size:-delta]).reshape(batch_size, input_len, 1) / 10
                    y_train = np.array(target[-batch_size:])
                    lstm_model_mse.fit(x_train, y_train,
                                       batch_size=batch_size,
                                       epochs=1,
                                       verbose=0)
        else:
            corrected_predict[-1] = last[-1]

        # send back
        msgInterface.PySendBegin()
        msgInterface.GetPy2CppStruct().new_wbCqi = corrected_predict[-1]
        msgInterface.PySendEnd()
        old_print("set: %d" % corrected_predict[-1])

except Exception as e:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    print(f"Exception occurred: {e}")
    print("Traceback:")
    traceback.print_tb(exc_traceback)
    exit(1)

else:
    with open("log_" + str(delta), "a+") as f:
        f.write("\n")
        # 顯示 LSTM 預測的 "準確度"
        if len(right):
            success_rate = sum(right) / len(right)
            f.write("LSTM accuracy = %f %%\n" % (success_rate * 100.0))

        # MSE_T
        if len(target) > delta:
            t1 = np.array(target[delta:])
            t2 = np.array(target[:-delta]) if delta else np.array(target)
            min_len = min(len(t1), len(t2))
            if min_len > 0:
                mse_t_val = simple_MSE(t1[:min_len], t2[:min_len])
                f.write("MSE_T = %f\n" % mse_t_val)

        # MSE_p
        if len(corrected_predict) > delta and len(target) > 0:
            p1 = np.array(corrected_predict[delta:])
            if delta:
                t2 = np.array(target[:-delta]) if len(target) > delta else np.array([])
            else:
                t2 = np.array(target)

            min_len = min(len(p1), len(t2))
            if min_len > 0:
                mse_p_val = simple_MSE(p1[:min_len], t2[:min_len])
                f.write("MSE_p = %f\n" % mse_p_val)
            else:
                f.write("MSE_p = N/A\n")

finally:
    print("Finally exiting...")
    del exp

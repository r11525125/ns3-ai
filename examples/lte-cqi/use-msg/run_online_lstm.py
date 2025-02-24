# Copyright (c) 2019-2023 Huazhong University of Science 
# and Technology, Dian Group
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 as
# published by the Free Software Foundation;
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
# Author: 
#   - Pengyu Liu <eic_lpy@hust.edu.cn>
#   - Xiaojun Guo <guoxj@hust.edu.cn>
#   - Hao Yin <haoyin@uw.edu>
#   - Muyuan Shen <muyuan_shen@hust.edu.cn>

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

#================================================
# 根據 sys.argv[1] 來判斷是否使用 LSTM
# 如果 delta == 0，就直接不做任何 LSTM 訓練 (not_train = True)
#================================================
delta = int(sys.argv[1])
if delta == 0:
    not_train = True
else:
    not_train = False

MAX_RBG_NUM = 32

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

# 相關參數
input_len = 200
pred_len = 40
batch_size = 20
alpha = 0.6

#======================
# 建立 LSTM 模型
#======================
lstm_input_vec = Input(shape=(input_len, 1), name="input_vec")
dense1 = Dense(30, activation='selu', kernel_regularizer='l1')(lstm_input_vec[:, :, 0])
old_print(dense1)  # 只是驗證一下 shape
lstm_l1_mse = tf.keras.ops.expand_dims(dense1, axis=-1)
lstm_mse = LSTM(20)(lstm_l1_mse)
predict_lstm_mse = Dense(1)(lstm_mse)
lstm_model_mse = keras.Model(inputs=lstm_input_vec, outputs=predict_lstm_mse)
lstm_model_mse.compile(optimizer="adam", loss="MSE")


#======================
# 評估函式
#======================
def simple_MSE(y_pred, y_true):
    return (((y_pred - y_true)**2)).mean()

def weighted_MSE(y_pred, y_true):
    return (((y_pred - y_true)**2) * (1 + np.arange(len(y_pred))) / len(y_pred)).mean()


#======================
# 變數初始化
#======================
cqi_queue = []
prediction = []
last = []
right = []
corrected_predict = []
target = []
train_data = []
delay_queue = []

#======================
# 啟動 ns3-ai 實驗
#======================
exp = Experiment("ns3ai_ltecqi_msg", "../../../../../", py_binding, handleFinish=True)
msgInterface = exp.run(show_output=True)

try:
    while True:
        msgInterface.PyRecvBegin()
        # 若 NS-3 結束了，就跳脫
        if msgInterface.PyGetFinished():
            break
        gc.collect()

        # 從 ns-3 收到當前 CQI
        CQI = msgInterface.GetCpp2PyStruct().wbCqi
        msgInterface.PyRecvEnd()

        # 基本檢查
        if CQI > 15:
            break

        old_print("get: %d" % CQI)

        #========================
        # 模擬回報延遲 delta
        #========================
        delay_queue.append(CQI)
        if len(delay_queue) < delta:
            # 若 queue 長度不足，就用最後一筆
            CQI_for_prediction = delay_queue[-1]
        else:
            # 否則就用 delta 前的那筆
            CQI_for_prediction = delay_queue[-delta]

        #========================
        # 不用 LSTM 的情況
        #========================
        if not_train:
            msgInterface.PySendBegin()
            msgInterface.GetPy2CppStruct().new_wbCqi = CQI_for_prediction
            msgInterface.PySendEnd()
            continue

        #========================
        # 以下是 LSTM 流程
        #========================
        cqi_queue.append(CQI_for_prediction)

        # 只有當 cqi_queue 足夠長，才將當前值加入 target
        if len(cqi_queue) >= input_len + delta:
            target.append(CQI_for_prediction)

        # 如果 cqi_queue >= input_len，截取最後 input_len 做為一筆訓練數據
        if len(cqi_queue) >= input_len:
            one_data = cqi_queue[-input_len:]
            train_data.append(one_data)
        else:
            # 還不夠長，無法做預測或訓練，先把原值送回 ns-3
            msgInterface.PySendBegin()
            msgInterface.GetPy2CppStruct().new_wbCqi = CQI_for_prediction
            msgInterface.PySendEnd()
            old_print("set: %d" % CQI_for_prediction)
            continue

        #========================
        # 進行一次 LSTM 預測
        #========================
        data_to_pred = np.array(one_data).reshape(-1, input_len, 1) / 10.0
        _predict_cqi = lstm_model_mse.predict(data_to_pred)
        old_print(_predict_cqi)
        del data_to_pred

        predict_cqi_int = int(_predict_cqi[0, 0] + 0.49995)
        prediction.append(predict_cqi_int)
        last.append(one_data[-1])
        corrected_predict.append(predict_cqi_int)
        del one_data

        #========================
        # 線上訓練
        #========================
        if len(train_data) >= pred_len + delta:
            err_t = weighted_MSE(
                np.array(last[(-pred_len - delta):-delta]),
                np.array(target[-pred_len:])
            )
            err_p = weighted_MSE(
                np.array(prediction[(-pred_len - delta):-delta]),
                np.array(target[-pred_len:])
            )
            if err_p <= err_t * alpha:
                # 若預測誤差小於門檻，視為成功，不額外訓練
                if err_t < 1e-6:
                    corrected_predict[-1] = last[-1]
                print(" ")
                print("OK %d %f %f" % ((len(cqi_queue)), err_t, err_p))
                right.append(1)
            else:
                # 若預測誤差較大，嘗試在線訓練
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

                    lstm_model_mse.fit(
                        x=np.array(train_data[-delta - batch_size:-delta]).reshape(batch_size, input_len, 1) / 10,
                        y=np.array(target[-batch_size:]),
                        batch_size=batch_size,
                        epochs=1,
                        verbose=0
                    )
        else:
            # 若 train_data 還不滿足條件，就暫時直接回傳
            corrected_predict[-1] = last[-1]

        #========================
        # 把 (修正後的) CQI 寫回 ns-3
        #========================
        msgInterface.PySendBegin()
        msgInterface.GetPy2CppStruct().new_wbCqi = corrected_predict[-1]
        msgInterface.PySendEnd()
        print("set: %d" % corrected_predict[-1])

except Exception as e:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    print("Exception occurred: {}".format(e))
    print("Traceback:")
    traceback.print_tb(exc_traceback)
    exit(1)

else:
    #========================
    # 計算並輸出 MSE
    #========================
    with open("log_" + str(delta), "a+") as f:
        f.write("\n")
        if len(right):
            success_rate = sum(right) / len(right)
            f.write("rate = %f %%\n" % (success_rate * 100.0))

        #--- MSE_T: target 自己跟自己做 shift 比對 ---
        # (target[delta:] vs. target[:-delta])
        # 先確保兩者長度一致
        if len(target) > delta:
            t1 = np.array(target[delta:])
            # 如果 delta=0，就相當於 target[:]
            t2 = np.array(target[:-delta]) if delta else np.array(target)
            min_len = min(len(t1), len(t2))
            mse_t_val = simple_MSE(t1[:min_len], t2[:min_len])
            f.write("MSE_T = %f\n" % mse_t_val)

        #--- MSE_p: corrected_predict 跟 target 比對 ---
        if len(corrected_predict) > delta and len(target) > 0:
            p1 = np.array(corrected_predict[delta:])
            # 如果 delta=0，就相當於 target[:]
            if delta:
                t2 = np.array(target[:-delta]) if len(target) > delta else np.array([])
            else:
                t2 = np.array(target)

            # 也要確保 p1, t2 長度一致
            min_len = min(len(p1), len(t2))
            if min_len > 0:
                mse_p_val = simple_MSE(p1[:min_len], t2[:min_len])
                f.write("MSE_p = %f\n" % mse_p_val)
            else:
                f.write("MSE_p = N/A (not enough data)\n")

finally:
    print("Finally exiting...")
    del exp

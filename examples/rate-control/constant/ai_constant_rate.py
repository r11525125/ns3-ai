#! /usr/bin/env python
# -*- Mode: python; py-indent-offset: 4; indent-tabs-mode: nil; coding: utf-8; -*-
# Copyright (c) 2021 Huazhong University of Science and Technology, Dian Group
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
# Author: Xun Deng <dorence@hust.edu.cn>
#         Hao Yin <haoyin@uw.edu>


import ns3ai_ratecontrol_constant_py as py_binding
from ns3ai_utils import Experiment


def get_action(env):
    nss = min(env.transmitStreams, env.supportedStreams)
    if env.mcs != 0xff:
        nss = 1 + env.mcs // 8
    # set next_mcs as previous mcs
    next_mcs = env.mcs
    # uncomment to specify arbitrary MCS
    # act.next_mcs = 5
    return nss, next_mcs


ns3Settings = {
    'raa': 'AiConstantRate',
    'nWifi': 3,
    'standard': '11ac',
    'duration': 5}

exp = Experiment("ns3ai_ratecontrol_constant", "../../../../../", py_binding, handleFinish=True)
msgInterface = exp.run(setting=ns3Settings, show_output=True)

try:
    while True:
        msgInterface.PyRecvBegin()
        msgInterface.PySendBegin()
        if msgInterface.PyGetFinished():
            break
        msgInterface.GetPy2CppStruct().nss, msgInterface.GetPy2CppStruct().next_mcs = (
            get_action(msgInterface.GetCpp2PyStruct()))
        msgInterface.PyRecvEnd()
        msgInterface.PySendEnd()

except Exception as e:
    print("Exception occurred in experiment:")
    print(e)

else:
    pass

finally:
    print("Finally exiting...")
    del exp

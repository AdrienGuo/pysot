# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.core.config import cfg
# from pysot.tracker.siamrpn_tracker import SiamRPNTracker
# 使用亭儀寫好的測試方法
from pysot.tracker.siamrpn_tracker_adrien import SiamRPNTracker

# from pysot.tracker.siammask_tracker import SiamMaskTracker
from pysot.tracker.siammask_tracker_amy import SiamMaskTracker

from pysot.tracker.siamrpnlt_tracker import SiamRPNLTTracker

TRACKS = {
          'SiamRPNTracker': SiamRPNTracker,
          'SiamMaskTracker': SiamMaskTracker,
          'SiamRPNLTTracker': SiamRPNLTTracker
         }


def build_tracker(model):
    return TRACKS[cfg.TRACK.TYPE](model)

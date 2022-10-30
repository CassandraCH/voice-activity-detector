import logging
#Retrieving notices: ...working... done

import os
#import sidiar
import torch

from collections import defaultdict
from pathlib import Path

from tqdm.auto import tqdm

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, check_and_rglob
from lhotse import LilcomFilesWriter
from lhotse import CutSet


class m2set(torch.utils.data.Dataset):
    """
    DataSet for sequence to sequence model training
    """

    def __init__(self, cutset_file):
        """
        """
        # load the cutset
        self.cutset = CutSet.from_file(cutset_file)
        self.len = len(self.cutset)


    def __getitem__(self, idx):
        """
        """
        feats = self.cutset[idx].features.load().T
        labels = self.cutset[idx].supervisions_feature_mask()
        return feats, labels

    def __len__(self):
        return self.len



if __name__ == '__main__':
    m2s = m2set(cutset_file="lists/allies_fbank_vad.jsonl.gz")

    # Display the size and characteristics of the m2set
    print(f"Length of m2s: {m2s.__len__()}")
    # for i, data in enumerate(m2s):
    #     print(i)
    #     feats = data[0]
    #     labels = data[1]
    #     print("feature:", feats, "labels:", labels)
    print("feats:",m2s[0][0])
    print("target:",m2s[0][1])

    # import ipdb
    # ipdb.set_trace()


import logging
import os
#import sidiar
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Union

from tqdm.auto import tqdm

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, check_and_rglob

from lhotse import CutSet
from lhotse import validate
from lhotse import Fbank, FbankConfig
from lhotse import LilcomFilesWriter


seg_duration = 30.
seg_shift = 30.


train_recording_set = RecordingSet.from_file('/lium/buster1/larcher/M2/deep_learning/lists/allies_train.jsonl')
train_uem = SupervisionSet.from_file('/lium/buster1/larcher/M2/deep_learning/lists/allies_uem_train.jsonl')
train_trs = SupervisionSet.from_file('/lium/buster1/larcher/M2/deep_learning/lists/allies_supervisions_train.jsonl')

# Remove the speaker IDs and set them to "speech"
vad_segs = []
for sup in train_trs:
    sup.speaker = "speech"
    vad_segs.append(sup)
train_vad = SupervisionSet.from_segments(vad_segs)

# Create the CutSet
cuts = CutSet.from_manifests(recordings=train_recording_set,
                             supervisions=train_vad,
                            )
cuts.describe()


# Split the cuts in 30 second chunks (can be changed to test different configurations of training)
new_cuts = []
for uem_cut in train_uem:
    rootname = uem_cut.id.split('-')[0]
    cut = cuts.filter(lambda c: c.id.startswith(rootname))[0]

    # get all cuts with duration seg_duration
    start = uem_cut.start
    stop = start + seg_duration
    while stop < uem_cut.start + uem_cut.duration:
        new_cuts.append(cut.truncate(offset=start, duration=seg_duration))
        start += seg_shift
        stop = start + seg_duration

chunkset = CutSet.from_cuts(new_cuts)

fbank = Fbank(config=FbankConfig(num_mel_bins=80))
storage = LilcomFilesWriter('fbanks')
cut_with_feats = chunkset.compute_and_store_features(extractor=fbank, storage_path="fbanks", num_jobs=8)
cut_with_feats.to_file("lists/allies_train_fbank_vad.jsonl.gz")






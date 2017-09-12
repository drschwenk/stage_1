import os
import copy
import json
import pandas as pd
from tqdm import tqdm


class FlintstonesDataset(object):
    def __init__(self, video_names):
        self.data = [VideoAnnotation(vid_fn) for vid_fn in tqdm(video_names)]

    def __repr__(self):
        return json.dumps(self.summarize_dataset(), default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def update_s1a(self, stage1a_annos):
        _ = [video.update_stage1a(stage1a_annos) for video in self.data]

    def update_s1b(self, stage1b_annos):
        _ = [video.update_stage1a(stage1b_annos) for video in self.data]

    def summarize_dataset(self):
        summary = {
            'video count': str(len(self.data)),
            'stage statuses': self.count_stages(),
            'go count': self.count_go(),
            'reasons for removal': self.count_reasons()
        }
        return summary

    def count_stages(self):
        vid_stages = pd.Series([vid.stage_status() for vid in self.data])
        stage_counts = vid_stages.value_counts().to_dict()
        return {k: str(v) for k, v in stage_counts.items()}

    def count_reasons(self):
        vid_reasons = pd.Series([vid.removal_reason() for vid in self.data])
        vid_reasons = vid_reasons[vid_reasons != ""]
        vid_reasons = vid_reasons.value_counts().to_dict()
        vid_reasons_str = {k: str(v) for k, v in vid_reasons.items() if k}
        vid_reasons_str['total removed'] = str(sum(vid_reasons.values()))
        return vid_reasons_str

    def count_go(self):
        vid_stages = pd.Series([vid.go_status() for vid in self.data])
        stage_counts = vid_stages.value_counts().to_dict()
        return [str(v) for k, v in stage_counts.items() if k][0]


class VideoAnnotation(object):

    def __init__(self, video_fn):

        self.properties = {
            'frame_width': 640
        }

        self.vid_data = {
            'globalID': '',
            'keyFrames': [],
            'setting': '',
            'status': {
                'stage': '',
                'go': None,
                'reason': ''
            },
            'characters': []
        }
        self.keyframe_postfixes = ['_' + str(x) + '.png' for x in [40]]

        self.vid_data['globalID'] = self.make_global_id(video_fn)
        self.vid_data['keyFrames'] = [self.vid_data['globalID'] + x for x in self.keyframe_postfixes]
        self.vid_data['status']['stage'] = 'stage_0'
        self.vid_data['status']['go'] = True

    def __repr__(self):
        return self.json()

    def json(self):
        return json.dumps(self.data(), default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def data(self):
        return self.vid_data

    def gid(self):
        return self.vid_data['globalID']

    def make_global_id(self, video_name):
        return os.path.splitext(video_name)[0]

    def stage_status(self):
        return self.vid_data['status']['stage']

    def go_status(self):
        return self.vid_data['status']['go']

    def removal_reason(self):
        return self.vid_data['status']['reason']

    def set_status(self, stage_completed, go=True, reason=""):
        self.vid_data['status']['stage'] = stage_completed
        self.vid_data['status']['go'] = go
        self.vid_data['status']['reason'] = reason

    def update_stage1a(self, s1_annos):
        if self.gid() not in s1_annos:
            self.set_status(self.stage_status(), False, "missing stage1a annotation")
            return

        character_annos = sorted(s1_annos[self.gid()], key=lambda x: x['area'], reverse=True)

        if not character_annos:
            self.set_status(self.stage_status(), False, "no characters annotated in stage1a")
            return

        self.vid_data['characters'] = [CharacterAnnotation(char, idx, self.gid()) for idx, char in enumerate(character_annos)]
        self.set_status('stage_1a')

    def update_stage1b(self, s1_annos):

        def recover_original_box(box):
            box_copy = copy.deepcopy(box)
            box_copy[0],  = box_copy[0] - self.properties['frame_width']
            box_copy[2] = box_copy[2] - self.properties['frame_width']
            return box_copy

        if self.gid() not in s1_annos:
            self.set_status(self.stage_status(), False, "missing stage1b annotation")
            return

        character_annos = s1_annos[self.gid()]

        if not character_annos:
            self.set_status(self.stage_status(), False, "no characters annotated in stage1b")
            return





    def update_stage3a(self, setting_anno):
        self.vid_data['setting'] = setting_anno

    def update_stage3b(self, raw_anno):
        pass


class CharacterAnnotation(object):

    def __init__(self, char_basics, char_idx, vid_gid):
        self.char_data = {
            'characterID': '',
            'characterName': '',
            'rectangles': [[], [], []],
            'nearbyObject': '',
            'interactsWith': {
                'positionID': '',
                'name': ''
            }
        }

        self.char_data['characterID'] = vid_gid + '_char_' + str(char_idx)
        self.char_data['characterName'] = char_basics['chosen_labels']
        self.char_data['rectangles'][1] = char_basics['box'].tolist()

    def __repr__(self):
        return self.json()

    def json(self):
        return json.dumps(self.data(), default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def data(self):
        return self.char_data


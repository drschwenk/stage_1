import os
import copy
import json
from collections import defaultdict
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
import PIL.Image as pilImage
import cv2
from IPython.display import Image


class FlintstonesDataset(object):
    def __init__(self, video_names):
        self.data = [VideoAnnotation(vid_fn) for vid_fn in video_names]

        self.main_characters = {
            "Fred",
            "Wilma",
            "Mr Slate",
            "Barney",
            "Betty",
            "Pebbles",
            "Dino",
            "Baby Puss",
            "Hoppy",
            "Bamm Bamm",
        }

    def __repr__(self):
        return json.dumps(self.summarize_dataset(), default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def update_s1a(self, stage1a_annos):
        _ = [video.update_stage1a(stage1a_annos) for video in self.data]

    def update_s1b(self, stage1b_annos):
        _ = [video.update_stage1b(stage1b_annos) for video in self.data]

    def update_s3a(self, stage3a_annos):
        _ = [video.update_stage3a(stage3a_annos) for video in self.data]

    def update_s3b(self, stage3b_annos):
        _ = [video.update_stage3b(stage3b_annos) for video in self.data]

    def update_s4a(self, stage4a_annos):
        _ = [video.update_stage4a(stage4a_annos) for video in self.data]

    def update_s4b(self, stage4b_annos):
        _ = [video.update_stage4b(stage4b_annos) for video in self.data]

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

    def count_stages_go_vids(self):
        vid_stages = pd.Series([vid.stage_status() for vid in self.data if vid.go_status()])
        stage_counts = vid_stages.value_counts().to_dict()
        return {k: str(v) for k, v in stage_counts.items()}

    def count_stages_cumulative(self):
        vid_stages = pd.Series([vid.stage_status() for vid in self.data])
        stage_counts = vid_stages.value_counts()
        shifted_counts = (20000 - stage_counts.sort_index().cumsum())
        shifted_labels = shifted_counts.index[1:]
        shifted_counts = shifted_counts[:-1]
        shifted_counts.index = shifted_labels
        return {k: str(v) for k, v in shifted_counts.to_dict().items()}

    def count_stages_go_vids_cumulative(self):
        vid_stages = pd.Series([vid.stage_status() for vid in self.data if vid.go_status()])
        stage_counts = vid_stages.value_counts()
        shifted_counts = (20000 - stage_counts.sort_index().cumsum())
        shifted_labels = shifted_counts.index[1:]
        shifted_counts = shifted_counts[:-1]
        shifted_counts.index = shifted_labels
        return {k: str(v) for k, v in shifted_counts.to_dict().items()}

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
        try:
            return [str(v) for k, v in stage_counts.items() if k][0]
        except IndexError:
            return str(0)

    def get_video(self, gid):
        return [vid for vid in self.data if vid.gid() == gid][0]

    def filter_videos(self, status_filters=None):
        if not status_filters:
            return self.data
        return [vid for vid in self.data if vid.check_status(status_filters)]

    def sorted_by_episode(self):
        return sorted(self.data, key=lambda x: x.gid())

    def write_to_json(self, version, out_dir='dataset'):
        to_json = copy.deepcopy(self.data)
        for vid in to_json:
            vid._data['characters'] = [char.data() for char in vid._data['characters']]

        ds_json = [vid._data for vid in to_json]
        out_file = os.path.join(out_dir, 'dataset_v{}.json'.format(version))
        with open(out_file, 'w') as f:
            json.dump(ds_json, f, sort_keys=True, indent=4)


class VideoAnnotation(object):

    def __init__(self, video_fn):

        self.properties = {
            'frame_width': 640,
            'has_objects': False,
            's3_gif_base': 'https://s3-us-west-2.amazonaws.com/ai2-vision-animation-gan/annotation_data/scene_gifs/',
            's3_still_base': 'https://s3-us-west-2.amazonaws.com/ai2-vision-animation-gan/annotation_data/still_frames/'
        }

        self._data = {
            'globalID': '',
            'keyFrames': [],
            'setting': '',
            'description': '',
            'parse': {},
            'status': {
                'stage': '',
                'go': None,
                'reason': ''
            },
            'characters': [],
            'objects': []
        }
        self.keyframe_postfixes = ['_' + str(x) + '.png' for x in [10, 40, 70]]

        self._data['globalID'] = self.make_global_id(video_fn)
        self._data['keyFrames'] = [self._data['globalID'] + x for x in self.keyframe_postfixes]

        self._data['status']['reason'] = ''
        self._data['status']['stage'] = 'stage_0'
        self._data['status']['go'] = True

    def __repr__(self):
        return self.json()

    def json(self):
        return json.dumps(self.data(), default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def data(self):
        return self._data

    def gid(self):
        return self._data['globalID']

    def make_global_id(self, video_name):
        return os.path.splitext(video_name)[0]

    def display_gif(self, url_only=False):
        gif_url = self.properties['s3_gif_base'] + self.gid() + '.gif'
        if url_only:
            return gif_url
        return Image(url=gif_url)

    def combine_frames(self, frame_images):
        widths, heights = zip(*(img.size for img in frame_images))
        total_width, max_height = sum(widths), max(heights)
        combined_img = pilImage.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in frame_images:
            combined_img.paste(im, (x_offset, 0))
            x_offset += im.size[0]
        return combined_img

    def get_key_frame_images(self):
        frame_urls = [''.join([self.properties['s3_still_base'], self.gid(), pfix]) for pfix in self.keyframe_postfixes]
        frame_images = [pilImage.open(requests.get(f_url, stream=True).raw) for f_url in frame_urls]
        return frame_images

    def display_keyframes(self):
        keyframes = self.get_key_frame_images()
        return self.combine_frames(keyframes)

    def display_bounding_boxes(self, frames=None):
        if not frames:
            frames = self.get_key_frame_images()
        frames_with_boxes = self.draw_char_boxes(frames)
        combined_image = np.hstack([np.asarray(img) for img in frames_with_boxes if img.any()])
        return pilImage.fromarray(combined_image)

    def draw_char_boxes(self, img_frames):
        drawn_frames = []
        for frame_idx, frame_img in enumerate(img_frames):
            open_cv_image = np.array(frame_img)
            open_cv_image = open_cv_image[:, :, ::].copy()
            for char in self._data['characters']:
                char_box = np.array(char.rect(frame_idx)).reshape(2, 2)
                char_idn = char.char_id().split('_')[-1]
                cv2.putText(open_cv_image,
                            char_idn, tuple(char_box[0] + np.array([0, 25])),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.rectangle(open_cv_image, tuple(char_box[0]), tuple(char_box[1]), color=(0, 255, 255), thickness=2)
            drawn_frames.append(open_cv_image)
        return drawn_frames

    def generate_md_review(self):
        pass

    def stage_status(self):
        return self._data['status']['stage']

    def go_status(self):
        return self._data['status']['go']

    def check_status(self, filters):
        status_checks = {
            'go': self.go_status,
            'stage': self.stage_status,
            'reason': self.removal_reason
        }

        checks_passed = [status_checks[fltr]() == status for fltr, status in filters.items()]
        return sum(checks_passed) == len(checks_passed)

    def setting(self):
        return self._data['setting']

    def description(self):
        return self._data['description']

    def characters_present(self):
        chars = self._data['characters']
        char_names_by_id = [char.name() for char in sorted(chars, key=lambda x: x.char_id())]
        char_ids = [char.char_id() for char in sorted(chars, key=lambda x: x.char_id())]
        return list(zip(char_ids, char_names_by_id))

    def removal_reason(self):
        return self._data['status']['reason']

    def set_status(self, stage_completed, go=True, reason=""):
        self._data['status']['stage'] = stage_completed
        self._data['status']['go'] = go
        self._data['status']['reason'] = reason

    def update_stage1a(self, s1_annos):
        if self.gid() not in s1_annos:
            self.set_status(self.stage_status(), False, "missing stage1a annotation")
            return

        character_annos = sorted(s1_annos[self.gid()], key=lambda x: x['area'], reverse=True)

        if not character_annos:
            self.set_status(self.stage_status(), False, "no consensus characters in stage1a")
            return

        self._data['characters'] = [CharacterAnnotation(char, idx, self.gid()) for idx, char in enumerate(character_annos)]
        self.set_status('stage_1a')

    def update_stage1b(self, s1b_annos):
        if self.removal_reason():
            return
        if self.gid() not in s1b_annos:
            self.set_status(self.stage_status(), True, "missing stage1b annotation")
            return

        character_annos = self._data['characters']
        stage_1b_boxes = sorted(s1b_annos[self.gid()], key=lambda x: x['label'])

        if not character_annos:
            self.set_status(self.stage_status(), False, "no characters annotated in stage1b")
            return

        try:
            _ = [char.update_1b(idx, stage_1b_boxes) for idx, char in enumerate(character_annos)]
            self.set_status('stage_1b')
        except IndexError:
            self.set_status(self.stage_status(), False, "characters not present in all frames")
            return

    def update_stage3a(self, s3a_annos):
        if self.removal_reason():
            return
        if self.gid() not in s3a_annos:
            self.set_status(self.stage_status(), True, "missing stage3a annotation")
            return
        self._data['setting'] = s3a_annos[self.gid()]
        self.set_status('stage_3a')

    def update_stage3b(self, s3b_annos):
        if self.removal_reason():
            return
        if self.gid() not in s3b_annos:
            self.set_status(self.stage_status(), True, "missing stage3b annotation")
            return
        self._data['description'] = s3b_annos[self.gid()]
        self.set_status('stage_3b')

    def update_stage4a(self, s4a_annos):
        if self.removal_reason():
            return
        if self.gid() not in s4a_annos:
            self.set_status(self.stage_status(), True, "missing stage4a annotation")
            return
        s4a_anno = s4a_annos[self.gid()]
        self._data['objects'] = [ObjectAnnotation(obj, self.gid()) for obj in
                                 zip(s4a_anno['descriptors'], s4a_anno['spans']) if obj[0] != self.setting()]
        if not self._data['objects']:
            self._data['objects'] = [ObjectAnnotation(('None', (0, 0)), self.gid())]
        if self._data['objects'][0].data()['localID'] == "None_0_0":
            self.set_status('stage_4b- no objects')
        else:
            self.set_status('stage_4a')
            self.properties['has_objects'] == True

    def update_stage4b(self, s4b_annos):
        if self.removal_reason() or self.stage_status() == 'stage_4b- no objects':
            return
        if self.gid() not in s4b_annos:
            if self.stage_status() != 'stage_4b- no objects':
                self.set_status(self.stage_status(), True, "missing object bounding boxes")
            return

        object_annos = self._data['objects']

        stage_4b_boxes = sorted(s4b_annos[self.gid()], key=lambda x: x['label'])

        if not stage_4b_boxes:
            self.set_status(self.stage_status(), False, "no consensus object annotations")
            return
        objects_by_label = defaultdict(list)
        for obj in stage_4b_boxes:
            objects_by_label[obj['label']].append(obj)

        for label, objects in objects_by_label.items():
            if len(objects) > 2:
                _ = [obj.update_4b(objects) for obj in object_annos if obj._data['localID'] == label]
        self.set_status('stage_4b- objects')


class BaseAnnotation(object):
    def __init__(self):
        self._data = {}
        self.properties = {
            'frame_width': 640,
        }

    def __repr__(self):
        return self.json()

    def json(self):
        return json.dumps(self.data(), default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def data(self):
        return self._data

    def rect(self, frame_idx='all'):
        three_frame_rectangles = self.data()['rectangles']
        if frame_idx == 'all':
            return three_frame_rectangles
        return three_frame_rectangles[frame_idx]

    def gid(self, id_type='globalID'):
        return self.data()[id_type]

    def name(self):
        return self._data['entityLabel']


class CharacterAnnotation(BaseAnnotation):

    def __init__(self, char_basics, char_idx, vid_gid):
        super().__init__()
        self._data = {
            'globalID': '',
            'entityLabel': '',
            'rectangles': [[], [], []],
            'nearbyObject': '',
            # remove for now
            # 'interactsWith': {
            #     'positionID': '',
            #     'name': ''
            # }
        }

        self.properties['char_base'] = 'https://s3-us-west-2.amazonaws.com/ai2-vision-animation-gan/' \
                                       'annotation_data/subtask_frames/'

        self._data['globalID'] = vid_gid + '_char_' + str(char_idx)
        self._data['entityLabel'] = char_basics['chosen_labels']
        self._data['rectangles'][1] = char_basics['box'].tolist()

    def update_1b(self, idx, stage_1b_boxes):

        def recover_original_box(box):
            box_copy = copy.deepcopy(box)
            box_copy[0] = box_copy[0] - self.properties['frame_width']
            box_copy[2] = box_copy[2] - self.properties['frame_width']
            return box_copy

        stage_1b_boxes = sorted(stage_1b_boxes[idx * 2: (idx + 1) * 2], key=lambda x: x['box'][0])
        self._data['rectangles'][0] = stage_1b_boxes[0]['box'].tolist()
        self._data['rectangles'][2] = recover_original_box(stage_1b_boxes[1]['box']).tolist()


class ObjectAnnotation(BaseAnnotation):
    def __init__(self, object_anno, vid_gid):
        super().__init__()
        self._data = {
            'rectangles': [[], [], []],
            'entityLabel': object_anno[0],
            'labelSpan': object_anno[1],
            'localID': '_'.join([object_anno[0], '_'.join([str(sp) for sp in object_anno[1]])]),
            'globalID': '_'.join([vid_gid, object_anno[0], '_'.join([str(sp) for sp in object_anno[1]])])
        }

    def update_4b(self, object_boxes):
        def recover_original_box(box, frame_n=1):
            box_copy = copy.deepcopy(box)
            box_copy[0] = box_copy[0] - self.properties['frame_width'] * frame_n
            box_copy[2] = box_copy[2] - self.properties['frame_width'] * frame_n
            return box_copy

        ordered_boxes = sorted(object_boxes, key=lambda x: x['box'][0])
        try:
            self._data['rectangles'][0] = ordered_boxes[0]['box'].tolist()
            self._data['rectangles'][1] = recover_original_box(ordered_boxes[1]['box']).tolist()
            self._data['rectangles'][2] = recover_original_box(ordered_boxes[2]['box'], 2).tolist()
        except IndexError:
            pass
           # print(self.global_id(), stage_4b_boxes)

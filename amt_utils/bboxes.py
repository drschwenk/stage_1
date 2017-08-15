import numpy as np
import scipy as st
import cv2
import os
import PIL.Image as Image
from sklearn.cluster import KMeans
import json
from copy import deepcopy
from collections import defaultdict


def box_area(box):
    height = box[1][1] - box[0][1]
    width = box[1][0] - box[0][0]
    return height * width


def box_area_ratio(box1, box2):
    return box_area(box2.reshape(2, 2)) / box_area(box1.reshape(2, 2))


def box_aspect_ratio(box):
    box = box.reshape(2, 2)
    return (box[1][1] - box[0][1]) / (box[1][0] - box[0][0])


def compute_intersection(b1, b2):
    dx = min(b1[1][0], b2[1][0]) - max(b1[0][0], b2[0][0])
    dy = min(b1[1][1], b2[1][1]) - max(b1[0][1], b2[0][1])
    if (dx >= 0) and (dy >= 0):
        intersection_area = dx * dy
        return intersection_area
    else:
        return 0


def comp_boxes_iou(b1, b2):
    b1 = b1.reshape(2, 2)
    b2 = b2.reshape(2, 2)
    b1_area = box_area(b1)
    b2_area = box_area(b2)
    intersection = compute_intersection(b1, b2)
    iou = intersection / (b1_area + b2_area - intersection)
    return iou


def comp_box_center(raw_box):
    box = raw_box.reshape(2, 2)
    return [(box[1][0] + box[0][0]) / 2, (box[1][1] + box[0][1]) / 2]


def characterbox_to_box(charBox):
    x1 = charBox['left']
    y1 = charBox['top']
    x2 = x1 + charBox['width']
    y2 = y1 + charBox['height']
    return np.array([x1, y1, x2, y2])


def is_duplicate(k, boxes, thresh):
    b1 = boxes[k]
    for i, b2 in enumerate(boxes):
        if i <= k:
            continue
        iou = comp_boxes_iou(b1['box'], b2['box'])
        if iou > thresh:
            return True
    return False


def assign_boxes(selected_boxes, duplicate_boxes):
    for b1 in duplicate_boxes:
        assign_idx = -1
        assign_iou = -1
        for i, b2 in enumerate(selected_boxes):
            iou = comp_boxes_iou(b1['box'], b2['box'])
            if iou > assign_iou:
                assign_iou = iou
                assign_idx = b2['idx']

        b1['duplicate_of'] = assign_idx


def print_boxes(selected_boxes, duplicate_boxes):
    print('-' * 10)
    print('Selected Boxes')
    print('-' * 10)
    [print(b) for b in selected_boxes]

    print('-' * 10)
    print('Duplicate Boxes')
    print('-' * 10)
    [print(b) for b in duplicate_boxes]


def filter_keep_by_area_fraction(boxes, keeps, thresh):
    for i, b1 in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            b2 = boxes[j]
            if comp_boxes_iou(b1['box'], b2['box']) > thresh:
                # print(b1, b2, box_area_ratio(b2['box'], b1['box']), 1.1 * box_aspect_ratio(b1['box']))
                # print(box_aspect_ratio(b2['box']), box_aspect_ratio(b1['box']))
                # print(b1, b2, box_aspect_ratio(b2['box']))
                # print(box_area_ratio(b2['box'], b1['box']),
                #       1.8 * (box_aspect_ratio(b2['box']) / box_aspect_ratio(b1['box']))**2 )
                # print((box_aspect_ratio(b2['box']) / box_aspect_ratio(b1['box']))**2)
                # print(b1, b2, box_area_ratio(b2['box'], b1['box']))
                if box_area_ratio(b2['box'], b1['box']) > 2.0 * (box_aspect_ratio(b2['box']) / box_aspect_ratio(b1['box']))**2 and keeps[i] != keeps[j]:

                    keeps[i] = not keeps[i]
                    keeps[j] = not keeps[j]
                break


def nms(charBoxes, thresh):
    boxes = [None] * len(charBoxes)
    for i, charBox in enumerate(charBoxes):
        box = characterbox_to_box(charBox)
        area = charBox['width'] * charBox['height']
        label = charBox['label']
        boxes[i] = {
            'box': box,
            'area': area,
            'label': label,
            'idx': i,
        }

    boxes = sorted(boxes, key=lambda x: x['area'], reverse=True)  # largest area first
    keep = [None] * len(boxes)
    for i, box in enumerate(boxes):
        keep[i] = not is_duplicate(i, boxes, thresh)
    # print(keep)
    filter_keep_by_area_fraction(boxes, keep, thresh)
    # print(keep)
    selected_boxes = [boxes[i] for i in range(len(boxes)) if keep[i]]
    duplicate_boxes = [boxes[i] for i in range(len(boxes)) if not keep[i]]
    assign_boxes(selected_boxes, duplicate_boxes)
    for b1 in selected_boxes:
        idx = b1['idx']
        votes = 1
        for b2 in duplicate_boxes:
            if b2['duplicate_of'] == idx:
                votes += 1

        b1['votes'] = votes
    # print_boxes(selected_boxes, duplicate_boxes)
    # print()
    if not duplicate_boxes:
        duplicate_boxes = []

    return selected_boxes, duplicate_boxes, boxes


def pick_consensus(clustered_boxes):
    return [sorted(cluster, key=lambda x: box_area(x))[:1] for cluster in clustered_boxes]


def limit_rect(rect, max_x, max_y, border_pad=2):
    if rect[0][0] < 0:
        rect[0][0] = border_pad
    if rect[0][1] < 0:
        rect[0][1] = border_pad
    if rect[1][0] > max_x:
        rect[1][0] = max_x - border_pad
    if rect[1][1] > max_y:
        rect[1][1] = max_y - border_pad


def draw_clusters(img_path, clustered_boxes, direction='rows', image=np.array([]), color=(0, 255, 255)):
    def random_color():
        import random
        return random.randint(0, 255), random.randint(0, 255), random.randint(0, 1)
    if not image.any():
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    max_height, max_width, channels = image.shape

    for idx, cluster in enumerate(clustered_boxes):
        if len(cluster) > 1:
            color = random_color()
            for box in cluster:
                limit_rect(box, max_width, max_height)
                cv2.rectangle(image, tuple(box[0]), tuple(box[1]), color=color, thickness=2)
        else:
            box = cluster[0]
            limit_rect(box, max_width, max_height)
            cv2.rectangle(image, tuple(box[0]), tuple(box[1]), color=color, thickness=3)
    return image


def get_frame_annos(frame_annos):
    rects = []
    labels = []
    for rect in json.loads(frame_annos['characterBoxes']):
        rects.append(rect_from_anno(rect))
        labels.append(rect['label'])
    sorted_idxs = [i[0] for i in sorted(enumerate(rects), key=lambda x: x[1])]
    return [rects[i] for i in sorted_idxs], [labels[i] for i in sorted_idxs]


def rect_from_anno(anno):
    upper_left = [anno['left'], anno['top']]
    upper_right = [upper_left[0] + anno['width'], upper_left[1] + anno['height']]
    return [upper_left, upper_right]


def print_labels(all_labels, frame_number):
    print()
    'Frame: ', frame_number + 1
    n_chars = [len(chars) for chars in all_labels]
    if len(set(n_chars)) == 1:
        char_array = np.array(all_labels)
        for char_idx in range(char_array.shape[1]):
            if len(set(char_array[:, char_idx])) == 1:
                print(char_array[0, char_idx], '     __all agree__')
            else:
                print('disagreement:')
                print('  | '.join(char_array[:, char_idx]))
            print()
    else:
        print('DISAGREE on NUMBER')

        print(all_labels)

    print()


def cluster_from_annos(annos, frame_number, n_turkers=3):
    rects_per_anno = [get_frame_annos(anno) for anno in annos]
    flattened_rects = [item for sublist in rects_per_anno for item in sublist[0]]
    labels = [rect[1] for rect in rects_per_anno]
    print_labels(labels, frame_number)
    box_clusters = cluster_diagram_text_centers(flattened_rects, n_turkers)
    return box_clusters


def cluster_from_nms(annos, _, __):
    # rects_per_anno = [get_frame_annos(anno) for anno in annos]
    # flattened_rects = [item for sublist in rects_per_anno for item in sublist[0]]
    boxes = [json.loads(anno['characterBoxes']) for anno in annos]
    flattened_boxes = [item for sublist in boxes for item in sublist]
    chars_present = [box['label'] for box in flattened_boxes]
    most_common = st.stats.mode(chars_present)
    if most_common[0][0] == 'empty_frame':
        flattened_boxes = [box for box in flattened_boxes if box['label'] == 'empty frame']
    else:
        flattened_boxes = [box for box in flattened_boxes if box['label'] != 'empty frame']
    selected_boxes, dupe_boxes, all_boxes = nms(flattened_boxes, 0.5)
    # if not dupe_boxes:
    #     filtered_boxes = selected_boxes
    # else:
    filtered_boxes = [box for box in selected_boxes if box['votes'] > 1]
    return filtered_boxes, dupe_boxes, all_boxes


def convert_nms_boxes(boxes):
    return [box['box'].tolist() for box in boxes]


def format_clusters(selected_boxes, dupe_boxes):
    clusters = {box['idx']: [box['box'].reshape(2, 2)] for box in selected_boxes}
    for box in dupe_boxes:
        if 'duplicate_of' in box.keys():
            clusters[box['duplicate_of']].append(box['box'].reshape(2, 2))
    return clusters


def draw_image_and_labels(still_annos, clusterer, frame_number=1, n_turkers=3, image_base_dir=None):
    if not image_base_dir:
        image_base_dir = '/Users/schwenk/wrk/animation_gan/build_dataset/Flintstone_Shots_Selected_Frames/'
    still_id = still_annos[0]['stillID']
    if clusterer.__name__ == 'cluster_from_annos':
        box_clusters = clusterer(still_annos, frame_number, n_turkers)
        consensus_boxes = pick_consensus(box_clusters)
        consensus_formatted = consensus_boxes
    else:
        consensus_boxes, box_clusters, all_boxes = clusterer(still_annos, frame_number, n_turkers)
        formatted_all = format_clusters(all_boxes, []).values()
        base_image = draw_clusters(os.path.join(image_base_dir, still_id), formatted_all, color=(128, 128, 128))
        if not box_clusters:
            formatted_boxes = format_clusters(consensus_boxes, []).values()
        boxes_formatted = format_clusters(consensus_boxes, box_clusters)
        consensus_formatted = format_clusters(consensus_boxes, consensus_boxes)
        consensus_formatted, box_clusters = consensus_formatted.values(), boxes_formatted.values()

    if set([box['label'] for box in consensus_boxes]) == set(['empty frame']):
        img_a = draw_clusters(os.path.join(image_base_dir, still_id), [], image=base_image)
        consensus_boxes = []
    else:
        img_a = draw_clusters(os.path.join(image_base_dir, still_id), box_clusters, image=base_image)
        img_a = draw_clusters(os.path.join(image_base_dir, still_id), consensus_formatted, image=img_a)
    return Image.fromarray(img_a), consensus_boxes, all_boxes


def find_matches(pairs):
    pairs_copy = deepcopy(pairs)
    for pair in pairs:
        for other_pair in pairs_copy:
            if bool(set(pair) & set(other_pair)):
                pair += other_pair

    matches = []
    for m in pairs:
        if set(m) not in matches:
            matches.append(set(m))
    return matches


def select_labels_three_frame(consensus_boxes, all_boxes, iou_thresh):
    all_char_boxes = []
    for frame_n, frame_boxes in enumerate(all_boxes):
        boxes_per_char_this_frame = defaultdict(list)
        for con_box in consensus_boxes[frame_n]:
            boxes_per_char_this_frame[con_box['idx']].append(con_box)
            for box in frame_boxes:
                if 'duplicate_of' in box and box['duplicate_of'] == con_box['idx']:
                    boxes_per_char_this_frame[box['duplicate_of']].append(box)
        all_char_boxes.append(boxes_per_char_this_frame)
    char_counts = [len(b) for b in all_char_boxes]
    first_frame_chars = list(all_char_boxes[1].values())
    all_frame_chars = deepcopy(first_frame_chars)
    # if len(set(char_counts)) != 1:
    #     print('number disagreement')
    if True:
        for frame in all_char_boxes[:1] + all_char_boxes[1:]:
            for chars in frame.values():
                cons_char = chars[0]
                con_center = comp_box_center(cons_char['box'])
                min_dist = 1000
                closest_char = None
                for idx, char in enumerate(first_frame_chars):
                    other_frame_center = comp_box_center(char[0]['box'])
                    dist = np.linalg.norm(np.array(other_frame_center) - np.array(con_center))
                    if dist < min_dist:
                        min_dist = dist
                        closest_char = idx
                all_frame_chars[closest_char].extend(chars)
    all_labels = [[cb['label'] for cb in char if cb['label'] != 'empty frame'] for char in all_frame_chars]
    chars_with_labels = []
    for idx, chars in enumerate(all_frame_chars):
        chars[0]['possible_labels'] = set(all_labels[idx])
        likely_char = st.stats.mode(all_labels[idx])
        if likely_char[1] > 5:
            chars[0]['chosen_labels'] = likely_char[0][0]
        else:
            chars[0]['chosen_labels'] = sorted(list(chars[0]['possible_labels']), key=lambda x: len(x), reverse=True)[0]
        chars_with_labels.append(chars[0])
    return chars_with_labels


def select_labels(consensus_boxes, all_boxes):
    all_char_boxes = defaultdict(list)
    for con_box in consensus_boxes:
        all_char_boxes[con_box['idx']].append(con_box)
        for box in all_boxes:
            if 'duplicate_of' in box and box['duplicate_of'] == con_box['idx']:
                all_char_boxes[box['duplicate_of']].append(box)
    all_labels = [[cb['label'] for cb in char if cb['label'] != 'empty frame'] for char in all_char_boxes.values()]
    chars_with_labels = []
    for idx, chars in enumerate(consensus_boxes):
        chars['possible_labels'] = set(all_labels[idx])
        likely_char = st.stats.mode(all_labels[idx])
        if likely_char[1] > 1:
            chars['chosen_labels'] = likely_char[0][0]
        else:
            chars['chosen_labels'] = sorted(list(chars['possible_labels']), key=lambda x: len(x), reverse=True)[0]
        chars_with_labels.append(chars)
    return chars_with_labels


def draw_animation_seq(anim_seq, clusterer):
    single_still_annos = anim_seq[:3], anim_seq[3:6], anim_seq[6:]
    images_and_boxes = [draw_image_and_labels(single_still_annos[frame_n], clusterer, frame_n) for frame_n in range(3)]
    three_frames, consensus_boxes, all_boxes = zip(*images_and_boxes)
    labels = select_labels(consensus_boxes, all_boxes, 0.6)
    imgs_comb = np.hstack([np.asarray(i) for i in three_frames if i])
    return Image.fromarray(imgs_comb), consensus_boxes, labels


# def cluster_from_annos_combined(annos, frame_number, n_turkers=3):
#     rects_per_anno = [get_frame_annos(anno) for anno in annos]
#     flattened_rects = [item for sublist in rects_per_anno for item in sublist[0]]
#     labels = [rect[1] for rect in rects_per_anno]
#     box_clusters = cluster_diagram_text_centers(flattened_rects, n_turkers)
#     return box_clusters


def draw_animation_combined_clusters(anim_seq, n_turkers):
    img = draw_image_and_labels(anim_seq, cluster_from_annos_combined, n_turkers=n_turkers)
    return img


def crop_character_box(img, char):
    crop = img.crop(char['box'])
    return crop


def create_subtask_data_three_frames(anim_seq, clusterer):
    image_base_dir = '/Users/schwenk/wrk/animation_gan/build_dataset/Flintstone_Shots_Selected_Frames/'
    # three_frames, consensus_boxes, all_boxes = draw_image_and_labels(anim_seq[3:6], clusterer, 1)
    consensus_boxes, box_clusters, all_boxes = clusterer(anim_seq[3:6], 1, 3)
    still_ids = [still_annos['stillID'] for still_annos in [anim_seq[3], anim_seq[0], anim_seq[-1]]]
    img_paths = [os.path.join(image_base_dir, still_id) for still_id in still_ids]
    imgs = [Image.open(img_path) for img_path in img_paths]
    mid_image = imgs[0]
    char_crops = [crop_character_box(mid_image, char) for char in consensus_boxes]
    imgs_comb = np.hstack(imgs[1:])
    return Image.fromarray(imgs_comb), char_crops


def create_subtask_data(anim_seq, clusterer):
    image_base_dir = '/Users/schwenk/wrk/animation_gan/build_dataset/Flintstone_Shots_Selected_Frames/'
    # three_frames, consensus_boxes, all_boxes = draw_image_and_labels(anim_seq[3:6], clusterer, 1)
    consensus_boxes, box_clusters, all_boxes = clusterer(anim_seq, 1, 3)
    still_mid_id = anim_seq[0]['stillID']
    still_ids = [still_mid_id, still_mid_id.replace('_40','_10'), still_mid_id.replace('_40','_70')]
    img_paths = [os.path.join(image_base_dir, still_id) for still_id in still_ids]
    imgs = [Image.open(img_path) for img_path in img_paths]
    mid_image = imgs[0]
    char_crops = [crop_character_box(mid_image, char) for char in consensus_boxes]
    imgs_comb = np.hstack(imgs[1:])
    return Image.fromarray(imgs_comb), char_crops

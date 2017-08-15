
import os
import multiprocessing
from multiprocessing import freeze_support
from ai2.vision.utils.io import init_logging
from amt_utils.mturk import pickle_this, unpickle_this
from amt_utils.bboxes import create_subtask_data
from amt_utils.bboxes import cluster_from_nms
from tqdm import tqdm


procs = 8

# def initialize_worker():
#     init_logging()


def multimap(method, iterable, *args):
    # Must use spawn instead of fork or native packages such as cv2 and igraph will cause
    # children to die sporadically causing the Pool to hang
    multiprocessing.set_start_method('spawn', force=True)


    # this forces all children to use the CPU device instead of a GPU (if configured)
    # this eliminates the slow startup and warnings we receive when many children compete
    # to compile Cuda code for the GPU:
    # example: INFO (theano.gof.compilelock): Waiting for existing lock by process '7163' (I am process '7204')
    # GPUs never make sense to use in a multiprocessing setting with Theano, so this should be safe
    old_theano_flags = os.environ.get('THEANO_FLAGS')
    os.environ['THEANO_FLAGS'] = 'device=cpu'

    pool = multiprocessing.Pool(procs)


    results = pool.starmap(method, iterable)
    pool.close()
    pool.join()

    if old_theano_flags:
        os.environ['THEANO_FLAGS'] = old_theano_flags

    return results


def test_gen_subtask(aid, animation_annos):
    return aid


def gen_subtask(aid, animation_annos):
    print(aid)
    try:
        two_frame_img, char_crops = create_subtask_data(animation_annos, cluster_from_nms)
        if char_crops:
            two_frame_img.save('./subtask_data/frames/' + aid + '_taskb.png')
            n_chars = len([char_image.save('./subtask_data/char_crops/' + aid + '_char_' + str(charn) +'_taskb.png') for charn, char_image in enumerate(char_crops)])
            return {aid + '_taskb.png': n_chars}
        else:
            return {aid + '_taskb.png': 0}
    except IndexError:
        return {aid + '_taskb.png': 0}


if __name__ == '__main__':
    freeze_support()
    subtask_stills = {}    
    annotations_by_frame = unpickle_this('batch_1_4_annotations_by_frame')
    st_stills = multimap(gen_subtask, list(annotations_by_frame.items()))
    subtask_stills = {}
    for st_n in st_stills:
        subtask_stills.update(st_n)
    pickle_this(subtask_stills, 'subtask_still.pkl')

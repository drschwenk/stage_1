import os
import multiprocessing
from multiprocessing import freeze_support
from ai2.vision.utils.io import init_logging
from amt_utils.mturk import pickle_this, unpickle_this
from keysTkingdom import mturk_ai2
from keysTkingdom import aws_tokes
from keysTkingdom import mturk_aristo
from amt_utils.mturk import MTurk
from amt_utils.flintstones import get_assignments


procs = 8

# def initialize_worker():
#     init_logging()

turk_account = mturk_ai2
rw_host='mechanicalturk.amazonaws.com'
amt_con = MTurk(turk_account.access_key, turk_account.access_secret_key, host=rw_host)

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

    results = pool.map(method, iterable)
    pool.close()
    pool.join()

    if old_theano_flags:
        os.environ['THEANO_FLAGS'] = old_theano_flags

    return results


def get_hit_chunk(hit):
    return amt_con.connection.get_assignments(hit.HITId)

if __name__ == '__main__':

    latest_hits = unpickle_this('stage_3_basic_hits_8_31.pkl')
    all_assignments = multimap(get_hit_chunk, latest_hits)
    # flattened_hits = [hit for sublist in all_hits for hit in sublist]
    pickle_this(all_assignments, 'latest_result_all_8_31.pkl')

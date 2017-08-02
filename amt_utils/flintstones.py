import pickle
from jinja2 import Environment, FileSystemLoader
import os
import json

from boto.mturk.qualification import PercentAssignmentsApprovedRequirement, Qualifications, Requirement


def create_result(assmt):
    result = json.loads(assmt.answers[0][0].fields[0])
    result['h_id'] = assmt.HITId
    return result

characters_present = [{'h_id': anno['h_id'], 'still_id': anno['stillID'], 'characters': set([ch['label'] for ch in json.loads(anno['characterBoxes'])])} for anno in assignment_results]


def pickle_this(results_df, file_name):
    with open(file_name, 'w') as f:
        pickle.dump(results_df, f)


def un_pickle_this(file_name):
    with open(file_name, 'r') as f:
        results_df = pickle.load(f)
    return results_df


def generate_task_page(s3_base_path, img_id, template_file='character_bbox.html'):

    env = Environment(loader=FileSystemLoader('hit_templates'))
    template = env.get_template(template_file)
    html_dir = './html_renders'
    if not os.path.exists(html_dir):
        os.makedirs(html_dir)
    page_html = template.render(s3_uri_base=s3_base_path, image_id=img_id)

    return page_html


def get_completed_hits(mturk_connection):
    """
    Queries amt for all active user HITs.
    :param mturk_connection: active mturk connection established by user in the nb.
    :return: list of boto HIT result objects
    """
    reviewable_hits = []
    page_n = 1
    hits_left = True
    while hits_left:
        hit_range = mturk_connection.get_reviewable_hits(page_size=100, page_number=page_n)
        if not hit_range:
            hits_left = False
            break
        reviewable_hits.extend(hit_range)
        page_n += 1
    return reviewable_hits


def get_assignments(mturk_connection, reviewable_hits, status=None):
    """
    Retrieves individual assignments associated with the specified HITs.
    :param mturk_connection: active mturk connection established by user in the nb.
    :param reviewable_hits: HITs to review
    :param status: HIT status to filter by.
    :return: hit_id:assignment dict
    """
    assignments = defaultdict(list)
    for hit in reviewable_hits:
        assignment = mturk_connection.get_assignments(hit.HITId, status=status)
        assignments[hit.HITId].extend(assignment)
    return assignments


def build_hit_params(qhtml, static_params):
    """
    Dynamically builds some HIT params that will change based on the book/url
    :param url: formatted url of page image on s3
    :param static_params: Universal HIT params (set by user in notebook).
    :return: complete HIT parameters.
    """
    import copy
    import boto
    def build_qualifications():
        """
        Creates a single qualification that workers have a > 95% acceptance rate.
        :return: boto qualification obj.
        """
        qualifications = Qualifications()
        req1 = PercentAssignmentsApprovedRequirement(comparator="GreaterThan", integer_value="95")
        qualifications.add(req1)
        return qualifications

    hit_params = copy.deepcopy(static_params)
    hit_params['qualifications'] = build_qualifications()
    hit_params['reward'] = boto.mturk.price.Price(hit_params['amount'])
    hit_params['html'] = qhtml
    return hit_params


def prepare_hit(s3_base_path, img_uri, static_parameters):
    question_html = generate_task_page(s3_base_path, img_uri)
    return build_hit_params(question_html, static_params)

static_params = {
    'title': "Annotate characters from an animation frame",
    'description': "Draw bounding boxes and label characters appearing in a image",
    'keywords': ['animation', 'image', 'bounding box', 'image annotation'],
    'frame_height': 1000,
    'amount': 0.05,
    'duration': 3600 * 1,
    'lifetime': 3600 * 24 * 2,
    'max_assignments': 3,
}

s3_base_path = 'https://s3-us-west-2.amazonaws.com/ai2-vision-animation-gan/annotation_data/still_frames/'
build_hit_group = [prepare_hit(s3_base_path, still, static_params) for still in stills_to_annotate]
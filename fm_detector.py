import os
import streamlit as st
import numpy as np
import tensorflow as tf
from object_detection.utils import config_util, label_map_util, visualization_utils as viz_utils
from object_detection.builders import model_builder
from matplotlib import pyplot as plt
from PIL import Image
import random
paths = {
    'tf2_ob': 'models',
    'checkpoint': 'checkpoint',
    'files': 'files'
}
files = {
    'labelmap': 'files/labelmap.pbtxt',
    'config' : 'files/pipeline.config'
}
# labels = [{'name': 'without_mask', 'id':1},
#           {'name': 'with_mask', 'id':2},
#           {'name': 'mask_weared_incorrect', 'id':3}]

@st.cache
def load_n_restore(config_dir, ckpt_dir, ckpt_ver = 0):
    # Load
    config = config_util.get_configs_from_pipeline_file(config_dir)
    detection_model = model_builder.build(model_config=config['model'], is_training=False)
    # Restore weights
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(ckpt_dir, f'ckpt-{ckpt_ver}')).expect_partial()
    return detection_model

@tf.function
def detect_fn(image, detector):
    image, shapes = detector.preprocess(image)
    prediction_dict = detector.predict(image, shapes)
    detections = detector.postprocess(prediction_dict, shapes)
    return detections

def load_img():
    image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

    if image_file is not None:

        # To See details
        file_details = {"filename":image_file.name, "filetype":image_file.type,
                        "filesize":image_file.size}
        st.write(file_details)
        # Convert to ndarray
        img = Image.open(image_file)
        return np.array(img)
    else:
        return None

def process_img(img):
    img_tensor = tf.constant(img)
    img_tensor = tf.expand_dims(tf.cast(img_tensor, tf.float32), axis=0)
    return img_tensor

def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    max_boxes_to_draw=10,
                    min_score_thresh=0.5):
    """Wrapper function to visualize detections.

    Args:
    image_np: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
        and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
        this function assumes that the boxes to be plotted are groundtruth
        boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
        category index `id` and category name `name`) keyed by category indices.
    figsize: size for the figure.
    image_name: a name for the image file.
    """
    image_np_with_annotations = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_annotations,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=max_boxes_to_draw,
        min_score_thresh=min_score_thresh)
    st.image(image_np_with_annotations)

def main(flags):
    category_index = label_map_util.create_category_index_from_labelmap(files['labelmap'])
    label_id_offset = 1

    try:
        detector = load_n_restore(files['config'], paths['checkpoint'], 8)

        img_np = load_img()
        if img_np:
            img_tensor = process_img(img_np)

            detections = detect_fn(img_tensor, detector=detector)

            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            label_id_offset = 1

            plot_detections(
                img_np,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                max_boxes_to_draw=10,
                min_score_thresh=.5
            )
    except:
        st.write('ERROR!')

main(True)
import argparse
import os
import sys
from glob import glob
import hashlib
import io
import warnings
# External packages
from lxml import etree
import PIL.Image
import tensorflow as tf
from object_detection.utils import dataset_util
from object_detection.protos import string_int_label_map_pb2
from google.protobuf import text_format

warnings.simplefilter(action='ignore', category=FutureWarning)

IMAGE_FORMATS = ('jpg',
                 'jpeg',
				 'png')


def compile_label_id_map(anno_file_paths):
    """Compile label map"""

    # Initialize label map
    label_map = {}

    # Gather all labels from annotation files
    for anno_file_path in anno_file_paths:
        # Load annotation file
        with open(anno_file_path, 'r') as f:
            xml_str = f.read()
        # Annotation data (list of objects as per PascalVOC)
        annos = dataset_util.recursive_parse_xml_to_dict(etree.fromstring(xml_str))['annotation']
        for anno in annos['object']:
            label_map[anno['name']] = None

    # Sort labels
    labels = sorted(list(label_map.keys()))

    # Assign label ids
    for label_ind, label in enumerate(labels):
        label_map[label] = label_ind + 1

    # Return label map
    return label_map


def image_tf_example(image_path, anno_path, label_id_map):
    """Image example"""

    # Load image
    with open(image_path, 'rb') as f:
        img_encoded = f.read()
    # Image hash key
    hash_key = hashlib.sha256(img_encoded).hexdigest()
    # Image data
    image = PIL.Image.open(io.BytesIO(img_encoded))
    # Image data in feature
    feature = {'image/height': dataset_util.int64_feature(image.height),
               'image/width': dataset_util.int64_feature(image.width),
               'image/filename': dataset_util.bytes_feature(image_path.encode('utf8')),
               'image/source_id': dataset_util.bytes_feature(image_path.encode('utf8')),
               'image/key/sha256': dataset_util.bytes_feature(hash_key.encode('utf8')),
               'image/encoded': dataset_util.bytes_feature(img_encoded),
               'image/format': dataset_util.bytes_feature('png'.encode('utf8'))
               }

    # Load annotation file
    with open(anno_path, 'r') as f:
        xml_str = f.read()
    # Annotation data (list of objects as per PascalVOC)
    annos = dataset_util.recursive_parse_xml_to_dict(etree.fromstring(xml_str))['annotation']
    # Initialize annotation data (list of objects as per PascalVOC)
    bb_xmins = []
    bb_ymins = []
    bb_xmaxs = []
    bb_ymaxs = []
    anno_class_ids = []
    anno_class_labels = []
    anno_truncated = []
    anno_difficult = []
    anno_views = []
    # Compile annotation data
    for anno in annos['object']:
        # Bounding box
        bb_xmins.append(float(anno['bndbox']['xmin']) / image.width)
        bb_ymins.append(float(anno['bndbox']['ymin']) / image.height)
        bb_xmaxs.append(float(anno['bndbox']['xmax']) / image.width)
        bb_ymaxs.append(float(anno['bndbox']['ymax']) / image.height)
        # Class
        anno_class_labels.append(anno['name'].encode('utf8'))
        anno_class_ids.append(label_id_map[anno['name']])
        # Attributes
        anno_difficult.append(int(anno['difficult']))
        anno_truncated.append(int(anno['truncated']))
        anno_views.append(anno['pose'].encode('utf8'))
    # Annotation data in feature
    feature.update({'image/object/bbox/xmin': dataset_util.float_list_feature(bb_xmins),
                    'image/object/bbox/ymin': dataset_util.float_list_feature(bb_ymins),
                    'image/object/bbox/xmax': dataset_util.float_list_feature(bb_xmaxs),
                    'image/object/bbox/ymax': dataset_util.float_list_feature(bb_ymaxs),
                    'image/object/class/text': dataset_util.bytes_list_feature(anno_class_labels),
                    'image/object/class/label': dataset_util.int64_list_feature(anno_class_ids),
                    'image/object/difficult': dataset_util.int64_list_feature(anno_difficult),
                    'image/object/truncated': dataset_util.int64_list_feature(anno_truncated),
                    'image/object/view': dataset_util.bytes_list_feature(anno_views),
                    })
    # TF example
    return tf.train.Example(features=tf.train.Features(feature=feature))


def create_tf_record(input_dir, record_file_path, label_file_path):
    """Create TensorFlow record and label map files from images with annotations"""

    # === Find images ===
    print('Looking for images...')
    sys.stdout.flush()

    # Images in annotation directory
    image_list = []
    #for image_format in IMAGE_FORMATS:
    #    image_list.extend(glob(os.path.join(input_dir, '**', '*.' + image_format)))
    for filename in os.listdir(input_dir):
        if filename.split(".")[-1] in IMAGE_FORMATS:
            image_list.append(os.path.join(input_dir, filename))

    # Ignore empty image list
    if not image_list:
        print('ERROR! No images found!')
        return
    print('OK!')

    # === Create label map ===
    print('Creating label map...')
    sys.stdout.flush()

    # Filter out images without annotations
    image_paths = []
    anno_paths = []
    for image_file_path in image_list:
        # Image directory
        image_dir = os.path.dirname(image_file_path)
        # Image filename
        image_file_name = os.path.basename(image_file_path)
        # Annotation filename
        anno_file_name = '.'.join(image_file_name.split('.')[:-1]) + '.xml'
        # Annotation file path
        anno_file_path = os.path.join(image_dir, anno_file_name)
        # Handle images with annotations
        if os.path.exists(anno_file_path):
            image_paths.append(image_file_path)
            anno_paths.append(anno_file_path)

    # Compile label map
    label_id_map = compile_label_id_map(anno_paths)

    # Compile label map file
    pb_label_map = string_int_label_map_pb2.StringIntLabelMap()
    for label, label_id in label_id_map.items():
        # Add item to add to label map
        item = pb_label_map.item.add()
        # Compile label map item
        item.name = label
        item.display_name = label
        item.id = label_id

    # Write label map file
    with open(label_file_path, 'w') as f:
        f.write(text_format.MessageToString(pb_label_map))
    print('OK!')

    # === Create record file ===
    print('Creating record file:')
    sys.stdout.flush()

    # TF writer
    writer = tf.python_io.TFRecordWriter(record_file_path)
    # Progress status
    progress = 0
    # Compile record file
    for image_no, (image_path, anno_path) in enumerate(zip(image_paths, anno_paths)):
        # Image example
        try:
            image_example = image_tf_example(image_path, anno_path, label_id_map)
        except (IOError, etree.LxmlError) as e:
            print('\nSkipping image: {}'.format(str(e)))
            sys.stdout.flush()
            continue
        # Add image example to TF record file
        writer.write(image_example.SerializeToString())
        # Show progress
        if (image_no + 1) / float(len(image_paths)) * 100 >= progress:
            print('..{}%'.format(progress))
            sys.stdout.flush()
            progress += 10
    # New line
    print
    # Close TF writer
    writer.close()


def run():
    """Main function"""

    # CLI argument parser
    parser = argparse.ArgumentParser(description='Tool for generating TensorFlow record file '
                                                 'and label map from annotations',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Add arguments
    parser.add_argument('input_directory',
                        help='Path to directory with images and annotations (annotation files '
                             'must be next to image files)',
                        type=str,
                        )
    parser.add_argument('-r', '--record_file',
                        help='Path to TF record file',
                        type=str,
                        default='datatf.record',
                        )
    parser.add_argument('-l', '--label_file',
                        help='Path to TF label map file',
                        type=str,
                        default='label_map.pbtxt',
                        )
    # Parse arguments
    args = parser.parse_args()

    # Create TF records
    create_tf_record(args.input_directory,
                     args.record_file,
                     args.label_file)


if __name__ == '__main__':
    run()

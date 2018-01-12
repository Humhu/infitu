"""Learns an embedding model from a dataset
"""

import argparse
import yaml

import infi_learn as rr
import adel
import tensorflow as tf
import time

import matplotlib.pyplot as plt

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str,
                        help='Path to the dataset pickle')
    parser.add_argument('config_path', type=str,
                        help='Path to configuration YAML')
    parser.add_argument('output_path', type=str,
                        help='Path to save checkpoints to')
    args = parser.parse_args()

    # Load dataset
    print 'Loading dataset %s...' % args.dataset_path
    all_data = adel.BasicDataset()
    all_data.load(args.dataset_path)
    all_labeled = adel.LabeledDatasetTranslator(all_data)

    # Parse data properties
    s, r = all_labeled.all_data[0]
    bel, img = s
    bel_size = len(bel)
    img_size = img.shape
    print 'Found belief size: %d img_size: %s' % (bel_size, str(img_size))

    # Load config
    with open(args.config_path) as f:
        config = yaml.load(f)
    network_config = config['network']
    problem_config = config['embedding']
    learner_config = config['learner']

    network = rr.EmbeddingNetwork(img_size=img_size,
                                  vec_size=bel_size,
                                  scope='embedding',
                                  **network_config)
    problem = rr.EmbeddingProblem(model=network, **problem_config)
    learner = rr.EmbeddingLearner(problem=problem,
                                  **learner_config)

    # Hand over data
    print 'Splitting data...'
    for s, r in all_labeled.all_data:
        learner.report_state_value(state=s, value=r)
    print learner.get_status()

    # Initialize
    sess = tf.Session()
    print 'Initializing models...'
    problem.initialize(sess)

    plot_group = rr.PlottingGroup()
    for p in learner.get_plottables():
        plot_group.add_plottable(p)

    # Run iterations
    # NOTE Need to spin before learner to initialize plots
    plot_group.spin()
    print 'Training...'
    num_iters = 100
    start_time = time.time()
    for i in range(num_iters):
        learner.spin(sess)
        plot_group.spin()
        
        if i % 10 == 0:
            now = time.time()
            elapsed = now - start_time
            rate = (i+1) / elapsed
            print 'Elapsed %f (s) at %f spins/sec' % (elapsed, rate)
        
    raw_input('Press any key to exit...')
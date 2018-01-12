"""Processes a bag into a SARS dataset
"""

import rosbag
import argparse

import adel
import infi_learn as rr

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('bag_path', type=str, help='Path to bag to process')
    parser.add_argument('out_path', type=str, help='Path to save dataset to')
    parser.add_argument('--reward_topic', default='reward', type=str,
                        help='The reward signal topic (infi_msgs.RewardStamped)')
    parser.add_argument('--action_topic', default='actions', type=str,
                        help='The executed action topic (broadcast.FloatVectorStamped)')
    parser.add_argument('--belief_topic', default='belief_state', type=str,
                        help='The belief state feature topic (broadcast.FloatVectorStamped)')
    parser.add_argument('--image_topic', default='image', type=str,
                        help='The state image topic (sensor_msgs.Image)')

    parser.add_argument('--sync_tol', default=0.1, type=float,
                        help='Tolerance in seconds for belief/image synchronization')
    parser.add_argument('--sync_lag', default=1.0, type=float,
                        help='Processing lag for buffering')
    parser.add_argument('--dt', default=1.0, type=float,
                        help='Discrete time step in seconds')
    parser.add_argument('--gamma', default=0, type=float,
                        help='Discount rate to use for reward integration')
    args = parser.parse_args()


    dataset = adel.BasicDataset()
    data_reporter = adel.LabeledDatasetTranslator(base=dataset)

    def queue_data(is_active, payload):
        if is_active:
            s, a, r, sn = payload
        else:
            s, a = payload
            r = 0

        data_reporter.report_label(x=s, y=r)
    
    belief_source = rr.VectorSource()
    image_source = rr.ImageSource()
    state_source = rr.MultiDataSource([belief_source, image_source],
                                      tol=args.sync_tol)
    frontend = rr.SARSFrontend(source=state_source, dt=args.dt, lag=args.sync_lag,
                               sync_time_tolerance=args.sync_tol, gamma=args.gamma)
    frontend.register_callback(queue_data)

    bag = rosbag.Bag(args.bag_path)
    all_topics = [args.reward_topic, args.action_topic, args.belief_topic, args.image_topic]
    for topic, msg, t in bag.read_messages(topics=all_topics):

        if topic == args.reward_topic:
            frontend.reward_callback(msg)
        elif topic == args.action_topic:
            frontend.action_callback(msg)
        elif topic == args.belief_topic:
            belief_source.vec_callback(msg)
        elif topic == args.image_topic:
            image_source.image_callback(msg)
        else:
            pass

        frontend.spin(msg.header.stamp.to_sec())

    print 'Saving %d datapoints...' % data_reporter.num_data
    dataset.save(args.out_path)
import pyglet
from connect_four import Window, gym, Game, RandomAgent
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--mode',
        required=False,
        default='train',
        type=str,
        help='''
        play \t play against current agent\n 
        train \t train the agent\n 
        benchmark \t compare win rate against random player''')
    parser.add_argument(
        '--epochs',
        required=False,
        default=100,
        type=int,
        help="number of games to play in train mode, ignored in play mode")
    args, _ = parser.parse_known_args()

    if args.mode == 'play':
        window = Window()
        pyglet.app.run()
    elif args.mode == 'train':
        gym.train(args.epochs)
    elif args.mode == 'benchmark':
        gym.benchmark(args.epochs)

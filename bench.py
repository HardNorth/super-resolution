import os
import glob
import json
import logging
import argparse

from data import fullsize_generator
from model import load_model
from util import tensorflow_session

from keras import backend as K


def model_paths(input_dir):
    path_pattern = os.path.join(input_dir, '**', 'epoch-*.h5')
    paths = glob.glob(path_pattern, recursive=True)
    paths.sort()
    return paths


def select_best_psnr(psnr_dict):
    best_psnr = 0.0
    best_model = None

    for model, psnr in psnr_dict.items():
        if psnr > best_psnr:
            best_psnr = psnr
            best_model = model

    return best_psnr, best_model


def evaluate_model(model_path, generator):
    """Evaluate model against DIV2K benchmark and return PSNR"""
    logging.info('Load model %s', model_path)
    model = load_model(model_path)

    logging.info('Evaluate model ...')
    return model.evaluate_generator(generator, steps=100, verbose=1)[1]


def main(args):
    """
    Evaluate all models in a user-defined directory against the DIV2K benchmark.

    The results are written to a user-defined JSON file. All models in the input
    directory must have been trained for the same downgrade operator (bicubic or
    unknown) and the same scale (2, 3 or 4).
    """
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    mps = model_paths(args.indir)

    if mps:
        generator = fullsize_generator(args.dataset, subset='valid', downgrade=args.downgrade, scale=args.scale)
        psnr_dict = {}
        for mp in mps:
            reset(args.gpu_memory_fraction)
            psnr = evaluate_model(mp, generator)
            psnr_dict[mp] = psnr

        logging.info('Write results to %s', args.outfile)
        with open(args.outfile, 'w') as f:
            json.dump(psnr_dict, f)

        best_psnr, best_model = select_best_psnr(psnr_dict)
        logging.info('Best PSNR = %.4f of model %s', best_psnr, best_model)
    else:
        logging.warning('No models found in %s', args.indir)


def reset(gpu_memory_fraction):
    K.clear_session()
    K.tensorflow_backend.set_session(tensorflow_session(gpu_memory_fraction=gpu_memory_fraction))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DIV2K benchmark')

    parser.add_argument('-d', '--dataset', type=str, default='./dataset',
                        help='path to DIV2K dataset')
    parser.add_argument('-i', '--indir', type=str,
                        help='path to models directory')
    parser.add_argument('-o', '--outfile', type=str, default='./bench.json',
                        help='output JSON file')
    parser.add_argument('-s', '--scale', type=int, default=2, choices=[2, 3, 4],
                        help='super-resolution scale')
    parser.add_argument('--downgrade', type=str, default='bicubic', choices=['bicubic', 'unknown'],
                        help='downgrade operation')
    parser.add_argument('--gpu-memory-fraction', type=float, default=0.8,
                        help='fraction of GPU memory to allocate')

    main(parser.parse_args())
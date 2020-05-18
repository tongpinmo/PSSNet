import summary
import test
import train
import experiments
import applyOnImage
import argparse
import os
import matplotlib
matplotlib.use('Agg')

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp_name', default="acacia")
    parser.add_argument('-m', '--mode', default=None)
    parser.add_argument('-image_path', '--image_path', default=None)
    parser.add_argument('-model_path', '--model_path', default=None)
    parser.add_argument('-model_name', '--model_name', default=None)
    parser.add_argument(
        '-r',
        '--reset',
        action="store_const",
        const=True,
        default=False,
        help="If set, a new model will be created, overwriting any previous version.")

    args = parser.parse_args()

    dataset_name, model_name, metric_name = experiments.get_experiment(
        args.exp_name)

    # Paths
    name = "{}_{}".format(dataset_name, model_name)

    # Train checkpoint file
    path_model = "checkpoints/best_model_{}.pth".format(name)
    path_opt = "checkpoints/opt_{}.pth".format(name)
    path_best_model = "checkpoints/best_model_save_{}.pth".format(name)
    path_history = "checkpoints/history_{}.json".format(name)


    if args.image_path is not None:
        applyOnImage.apply(args.image_path, args.model_name, args.model_path)

    elif args.mode == "train":
        train.train(
            dataset_name,
            model_name,
            metric_name,
            path_history,
            path_model,
            path_opt,
            path_best_model,
            args.reset)

    elif args.mode == "test":
        test.test(
            dataset_name,
            model_name,
            metric_name,
            path_history,
            path_model)

    elif args.mode == "summary":
        summary.summary(dataset_name, model_name, path_history)


if __name__ == "__main__":
    main()

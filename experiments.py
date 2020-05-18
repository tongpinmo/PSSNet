def get_experiment(exp_name):

    if exp_name == "acacia":
        dataset_name="acacia"
        model_name="ResFCN"
        metric_name = "MAE"

    if exp_name == "acacia":
        dataset_name = "acacia"
        model_name = "ResUnet"
        metric_name = "MAE"

    if exp_name == "oilpalm":
        dataset_name = "oilpalm"
        model_name = "ResUnet"
        metric_name = "MAE"

    print("Model: {} - Dataset: {} - Metric: {}".format(model_name, dataset_name,metric_name))
    return dataset_name, model_name, metric_name

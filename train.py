import os
import torch
import losses
import numpy as np
import utils as ut

from torchvision import transforms
from datasets import dataset_dict
from models import model_dict


def train(
        dataset_name,
        model_name,
        metric_name,
        path_history,
        path_model,
        path_opt,
        path_best_model,
        reset=False):
  # SET SEED
    np.random.seed(1)
    torch.manual_seed(1)  # set seed for CPU
    torch.cuda.manual_seed_all(1)  # set seed for all GPU

  # Train datasets
    transformer = ut.ComposeJoint(
        [ut.RandomHorizontalFlipJoint(),
         [transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.4)],
         [transforms.ToTensor(), None],
         [transforms.Normalize(*ut.mean_std), None],
         [None, ut.ToLong()],
         ])
    # load train dataset
    train_set = dataset_dict[dataset_name](split="train",
                                           transform_function=transformer)

    trainloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=1,
        num_workers=0,
        drop_last=False,
        sampler=ut.RandomSampler(train_set))
    # Val datasets
    transformer = ut.ComposeJoint(
        [
            [transforms.ToTensor(), None],
            [transforms.Normalize(*ut.mean_std), None],
            [None, ut.ToLong()]
        ])

    val_set = dataset_dict[dataset_name](split="val",
                                       transform_function=transformer)

    test_set = dataset_dict[dataset_name](split="test",
                                       transform_function=transformer)

  # Model
    model = model_dict[model_name](train_set.n_classes).cuda()
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()),
        lr=1e-4,
        weight_decay=0.0005)

  # Train
    if os.path.exists(path_history) and not reset:
        history = ut.load_json(path_history)
        model.load_state_dict(torch.load(path_model))
        opt.load_state_dict(torch.load(path_opt))
        s_epoch = history["train"][-1]["epoch"]
        print("Resuming epoch...{}".format(s_epoch))

    else:
        history = {"train": [], "val": [], "test": [],
                   "model_name": model_name,
                   "dataset_name": dataset_name,
                   "path_model": path_model,
                   "path_best_model": path_best_model,
                   "best_val_epoch": -1,
                   "best_val_mae": np.inf}
        s_epoch = 0
        print("Starting from scratch...")

    for epoch in range(s_epoch + 1, 500):
        train_dict = ut.fit(model, trainloader, opt,
                            loss_function=losses.lc_loss,
                            epoch=epoch)

        # Update history
        history["trained_images"] = list(model.trained_images)
        history["train"] += [train_dict]

        # Save model, opt and history
        torch.save(model.state_dict(), path_model)
        torch.save(opt.state_dict(), path_opt)
        ut.save_json(path_history, history)

    # %%%%%%%%%%% 2. VALIDATION PHASE %%%%%%%%%%%%"
    with torch.no_grad():
        val_dict = ut.val(model=model, dataset=val_set, epoch=epoch,
                        metric_name=metric_name)

        # Update history
        history["val"] += [val_dict]

        print('keys of history: ', history.keys())
        print('val_dict[metric_name]: ',val_dict[metric_name])
        # print('history["metric_name"]: ',history["metric_name"])
        history["best_val_mae"] = np.inf
      # Lower is better
        if val_dict[metric_name] <= history["best_val_mae"]:
            history["best_val_epoch"] = epoch
            history["best_val_mae"] = val_dict[metric_name]

            torch.save(model.state_dict(), path_best_model)

        # Test Model
        if not (dataset_name == "penguins" and epoch < 50):
            testDict = ut.val(model=model,
                                dataset=test_set,
                                epoch=epoch, metric_name=metric_name)
            history["test"] += [testDict]

        ut.save_json(path_history, history)

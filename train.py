import click

from torch.utils import data

from datasets import RawDataset
from datautils import HSIDataset
from models import get_model
from models import train
from shallow_models import SKLEARN_MODELS
from shallow_models import fit_sklearn_model
from shallow_models import infer_from_sklearn_model
from utils import find_cuda_device


@click.command()
@click.argument("model")
@click.argument("dataset")
@click.option("--patch_size")
@click.option("--device", default="cpu")
@click.option("--n_jobs", default=0)
@click.option("--batch_size", "--bs")
@click.option("--epochs")
@click.option("--learning_rate", "--lr")
@click.option("--class_balancing", is_flag=True)
def train_model(
    model,
    dataset,
    patch_size,
    device,
    n_jobs,
    batch_size,
    epochs,
    learning_rate,
    class_balancing,
):
    """
    Train a model on the specified dataset.
    """

    device = find_cuda_device(device)
    dataset = RawDataset(dataset)
    # TODO: split in train and test

    if model in SKLEARN_MODELS:
        X_train, y_train = dataset.to_sklearn_datasets()
        clf = fit_sklearn_model(
            model,
            X_train,
            y_train,
            exp_name=dataset.folder,
            class_balancing=class_balancing,
            n_jobs=n_jobs,
        )
        prediction = infer_from_sklearn_model(clf, img)

    else:
        hyperparams = {
            "n_classes": len(dataset.labels),  # TODO: FIXME
            "n_bands": dataset.bands,
            "ignored_labels": dataset.ignored_labels,
            "batch_size": batch_size,
        }
        # Filter out None hyperparameters so they can updated with correct values later
        hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)
        
        deepnet, optimizer, loss, hyperparams = get_model(model, **hyperparams)
        print(hyperparams)
        train_ds = HSIDataset(dataset.data, dataset.masks, window_size=8, overlap=0.5)
        train_loader = data.DataLoader(
            train_ds,
            batch_size=hyperparams["batch_size"],
            shuffle=True,
            num_workers=n_jobs,
        )
        print(f"Learning on patches {train_ds[0][0].shape}")
        train(
            deepnet,
            optimizer,
            loss,
            train_loader,
            hyperparams["epoch"],
            exp_name=dataset.folder,
            scheduler=hyperparams["scheduler"],
            device=device,
            #val_loader=val_dataloader,
            #writer=writer,
        ) # TODO: check that the last model is saved in the train loop
         
        probabilities = test(
            deepnet,
            img,
            window_size=hyperparams["patch_size"],
            n_classes=len(LABEL_VALUES),
            overlap=TEST_OVERLAP,
        )
        prediction = np.argmax(probabilities, axis=-1)


if __name__ == "__main__":
    train_model()
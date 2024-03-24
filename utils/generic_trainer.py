import numpy as np
import torch
from utils.gpu_monitoring import GPUMonitor


def train_detection_only(train_set, val_set, model, loss, correct_count, batch_size=128, epochs=10, learning_rate=0.01, temp_dir=""):
    """
    Train a model and return accuracy and errors over the number of epochs
    
    Args:
        train_set: train dataset

        val_set: validation dataset

        model: torch model to train

        loss: the loss to be minimized

        correct_count: the number of correct detection

        batch_size: sample size by batch

        epoch: Number of epoch for training

        learning_rate: initial learning rate for optimizer


    returns errors: {"train": [], "val": []}, accuracy: {"train": [], "val": []}, best_model_path: str, monitor: GPUMonitor
    """

    # create device based on GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Using {device} for the training")

    # send model to device
    model = model.to(device)

    # as before: create optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # store train/val error & accuracy
    
    errors = { "train": [], "val": [] }
    accuracies = {"train": [], "val": []}

    # initialize loaders
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # custom object in order to monitor how much we use the GPU
    # you can have a look at it in 0. Imports and Utils
    monitor = GPUMonitor()
    monitor.start()

    #
    # Start of the training procedure
    #

    best_val_accuracy = 0
    best_model_path = f"{temp_dir}/model.pt"

    for e in range(epochs):

        train_errors = []
        train_correct = 0

        # Dropout and BatchNorm behave differently if you are training or evaluating
        # the model. Here we need them to be in training mode
        # (for instance, no dropout in evaluation mode)
        model.train()

        # ignore: clear print line
        print(" " * 100, end="")

        for i, batch in enumerate(train_loader):
            print(
                f"\rEpoch {e+1}/{epochs} | Train Batch {i+1}/{len(train_loader)}",
                end="",
            )

            # clear gradients
            optimizer.zero_grad()

            # send tensors to the proper device
            data = batch[0].to(device)
            actual_event = batch[1].to(device)

            # forward through model
            pred_event = model(data)
            
            # compute prediction error
            error = loss(pred_event, actual_event)

            # perform backward pass
            error.backward()

            # update model parameters
            optimizer.step()

            # store batch error
            train_errors.append(error.cpu().item())
            # store batch correct count (see 0. ) to compute accuracy
            # on the full training set for this epoch
            train_correct += correct_count(pred_event, actual_event)

        # clear print line
        print(" " * 100, end="")

        # evaluation: no gradients needed
        with torch.no_grad():
            val_errors = []
            val_correct = 0

            # Put the model in evaluation mode (vs .train() mode)
            model.eval()

            for i, batch in enumerate(val_loader):
                print(
                    f"\rEpoch {e+1}/{epochs} | Validation Batch {i+1}/{len(val_loader)}",
                    end="",
                )
                prediction = model(batch[0].to(device))
                error = loss(prediction, batch[1].to(device))
                val_errors.append(error.cpu().item())
                val_correct += correct_count(prediction, batch[1])

        # compute average errors
        train_error = np.mean(train_errors)
        val_error = np.mean(val_errors)

        # compute epoch-wise accuracies
        train_acc = train_correct / len(train_set) * 100
        val_acc = val_correct / len(val_set) * 100

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), best_model_path)

        # store metrics
        accuracies["train"].append(train_acc)
        accuracies["val"].append(val_acc)
        errors["train"].append(train_error)
        errors["val"].append(val_error)

        print(
            f"\rEpoch {e+1}/{epochs} - Train error: {train_error:.4f} Train acc: {train_acc:.1f}% - Val error: {val_error:.4f} Val acc: {val_acc:.1f}%"
        )

        # --------------------
        # --  End of epoch  --
        # --------------------

    # -------------------------------------
    # --  End of the training procedure  --
    # -------------------------------------

    # stop the GPU monitor
    monitor.stop()

    # return metric
    return errors, accuracies, best_model_path, best_val_accuracy, monitor



def train_detection_and_phase(train_set, val_set, model, detection_loss, phase_loss, total_loss, detection_correct_count, phase_correct_count, total_correct_count, batch_size=64, epochs=20, learning_rate=0.001):
    """
    Train a model and return accuracy and errors over the number of epochs for event, p phase and s phase detection
    
    Args:
        train_set: train dataset

        val_set: validation dataset

        model: torch model to train

        detection_loss: the loss for earthquake signal detection to be minimized

        phase_loss: the loss for phase detection to be minimized

        total_loss: The final loss combining detection and phases

        detection_correct_count: the count of correct event detection

        phase_correct_count: the count of correct phases

        total_correct_count: the count of total correct

        batch_size: sample size by batch

        epoch: Number of epoch for training

        learning_rate: initial learning rate for optimizer


    returns errors: {"train": [], "val": []}, accuracy: {"train": [], "val": []}, monitor: GPUMonitor
    """

    # create device based on GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # send model to device
    model = model.to(device)

    # as before: create optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # store train/val error & accuracy
    errors = { 
        "detection": {"train": [], "val": []},
        "p_phase": {"train": [], "val": []},
        "s_phase": {"train": [], "val": []},
        "total": {"train": [], "val": []}
    }

    accuracies = { 
        "detection": {"train": [], "val": []},
        "p_phase": {"train": [], "val": []},
        "s_phase": {"train": [], "val": []},
        "total": {"train": [], "val": []}
    }

    # initialize loaders
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=2
    )

    print(model)

    # custom object in order to monitor how much we use the GPU
    # you can have a look at it in 0. Imports and Utils
    monitor = GPUMonitor()
    monitor.start()

    #
    # Start of the training procedure
    #

    for e in range(epochs):

        train_detection_errors = []
        train_p_errors = []
        train_s_errors = []
        train_total_errors = []
        train_detection_correct = 0
        train_p_correct = 0
        train_s_correct = 0
        train_total_correct = 0

        # Dropout and BatchNorm behave differently if you are training or evaluating
        # the model. Here we need them to be in training mode
        # (for instance, no dropout in evaluation mode)
        model.train()

        # ignore: clear print line
        print(" " * 100, end="")

        for i, batch in enumerate(train_loader):
            print(
                f"\rEpoch {e+1}/{epochs} | Train Batch {i+1}/{len(train_loader)}",
                end="",
            )

            # clear gradients
            optimizer.zero_grad()

            # send tensors to the proper device
            data = batch[0].to(device)
            actual_event = batch[1].to(device)
            actual_p_phase = batch[2].to(device)
            actual_s_phase = batch[3].to(device)

            # forward through model
            pred_event, pred_p_phase, pred_s_phase = model(data)
            event_loss = detection_loss(actual_event, pred_event)
            p_phase_loss = phase_loss(actual_p_phase, pred_p_phase)
            s_phase_loss = phase_loss(actual_s_phase, pred_s_phase)
            error = total_loss(event_loss, p_phase_loss, s_phase_loss)

            # perform backward pass
            error.backward()

            # update model parameters
            optimizer.step()

            # store batch error
            train_detection_errors.append(event_loss.cpu().item())
            train_p_errors.append(p_phase_loss.cpu().item())
            train_s_errors.append(s_phase_loss.cpu().item())
            train_total_errors.append(error.cpu().item())

            # store batch correct count (see 0. ) to compute accuracy
            # on the full training set for this epoch
            train_detection_correct += detection_correct_count(actual_event, pred_event)
            train_p_correct += phase_correct_count(actual_p_phase, pred_p_phase)
            train_s_correct += phase_correct_count(actual_s_phase, pred_s_phase)
            train_total_correct += total_correct_count(train_detection_correct, train_p_correct, train_s_correct)

        # clear print line
        print(" " * 100, end="")

        # evaluation: no gradients needed
        with torch.no_grad():
            val_detection_errors = []
            val_p_errors = []
            val_s_errors = []
            val_total_errors = []
            val_detection_correct = 0
            val_p_correct = 0
            val_s_correct = 0
            val_total_correct = 0

            # Put the model in evaluation mode (vs .train() mode)
            model.eval()

            for i, batch in enumerate(val_loader):
                print(
                    f"\rEpoch {e+1}/{epochs} | Validation Batch {i+1}/{len(val_loader)}",
                    end="",
                )

                # send tensors to the proper device
                data = batch[0].to(device)
                actual_event = batch[1].to(device)
                actual_p_phase = batch[2].to(device)
                actual_s_phase = batch[3].to(device)

                # forward through model
                pred_event, pred_p_phase, pred_s_phase = model(data)
                event_loss = detection_loss(actual_event, pred_event)
                p_phase_loss = phase_loss(actual_p_phase, pred_p_phase)
                s_phase_loss = phase_loss(actual_s_phase, pred_s_phase)
                error = total_loss(event_loss, p_phase_loss, s_phase_loss)

                val_detection_errors.append(event_loss.cpu().item())
                val_p_errors.append(p_phase_loss.cpu().item())
                val_s_errors.append(s_phase_loss.cpu().item())
                val_total_errors.append(error.cpu().item())

                val_detection_correct += detection_correct_count(actual_event, pred_event)
                val_p_correct += phase_correct_count(actual_p_phase, pred_p_phase)
                val_s_correct += phase_correct_count(actual_s_phase, pred_s_phase)
                val_total_correct += total_correct_count(train_detection_correct, train_p_correct, train_s_correct)

        # compute average errors
        train_detection_error = np.mean(train_detection_errors)
        train_p_error = np.mean(train_p_errors)
        train_s_error = np.mean(train_s_errors)
        train_total_error = np.mean(train_total_errors)

        val_detection_error = np.mean(val_detection_errors)
        val_p_error = np.mean(val_p_errors)
        val_s_error = np.mean(val_s_errors)
        val_total_error = np.mean(val_total_errors)

        # compute epoch-wise accuracies
        train_detection_acc = train_detection_correct / len(train_set) * 100
        train_p_acc = train_p_correct / len(train_set) * 100
        train_s_acc = train_s_correct / len(train_set) * 100
        train_total_acc = train_total_correct / len(train_set) * 100

        val_detection_acc = val_detection_correct / len(val_set) * 100
        val_p_acc = val_p_correct / len(val_set) * 100
        val_s_acc = val_s_correct / len(val_set) * 100
        val_total_acc = val_total_correct / len(val_set) * 100

        # store metrics
        accuracies["detection"]["train"].append(train_detection_acc)
        accuracies["p_phase"]["train"].append(train_p_acc)
        accuracies["s_phase"]["train"].append(train_s_acc)
        accuracies["total"]["train"].append(train_total_acc)

        accuracies["detection"]["val"].append(val_detection_acc)
        accuracies["p_phase"]["val"].append(val_p_acc)
        accuracies["s_phase"]["val"].append(val_s_acc)
        accuracies["total"]["val"].append(val_total_acc)

        errors["detection"]["train"].append(train_detection_error)
        errors["p_phase"]["train"].append(train_p_error)
        errors["s_phase"]["train"].append(train_s_error)
        errors["total"]["train"].append(train_total_error)

        errors["detection"]["val"].append(val_detection_error)
        errors["p_phase"]["val"].append(val_p_error)
        errors["s_phase"]["val"].append(val_s_error)
        errors["total"]["val"].append(val_total_error)


        print(
            f"\rEpoch {e+1}/{epochs} - Detection - Train error: {train_detection_error:.4f} Train acc: {train_detection_acc:.1f}% - Val error: {val_detection_error:.4f} Val acc: {val_detection_acc:.1f}%"
        )
        print(
            f"\rEpoch {e+1}/{epochs} - P phase - Train error: {train_p_error:.4f} Train acc: {train_p_acc:.1f}% - Val error: {val_p_error:.4f} Val acc: {val_p_acc:.1f}%"
        )
        print(
            f"\rEpoch {e+1}/{epochs} - S phase - Train error: {train_s_error:.4f} Train acc: {train_s_acc:.1f}% - Val error: {val_s_error:.4f} Val acc: {val_s_acc:.1f}%"
        )
        print(
            f"\rEpoch {e+1}/{epochs} - Total - Train error: {train_total_error:.4f} Train acc: {train_total_acc:.1f}% - Val error: {val_total_error:.4f} Val acc: {val_total_acc:.1f}%"
        )

        # --------------------
        # --  End of epoch  --
        # --------------------

    # -------------------------------------
    # --  End of the training procedure  --
    # -------------------------------------

    # stop the GPU monitor
    monitor.stop()

    # return metric
    return errors, accuracies, monitor
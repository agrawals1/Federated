import torch
from torch import nn
import random
from ...core.alg_frame.client_trainer import ClientTrainer
from ...utils.model_utils import check_device
import logging


class ScaffoldModelTrainer(ClientTrainer):
    def get_model_params(self):
        return self.model.state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args, c_model_global_params, c_model_local_params):
        model = self.model
        model.to(device)
        model.train()
        num_epochs = args.epochs if args.var_epoch == 0 else random.randint(1,5)
        criterion = nn.CrossEntropyLoss().to(device)
        
        # Setting up the optimizer based on the args configuration
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay
            )
        elif args.client_optimizer == "CosAnnealing":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()), 
                lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=.0001)
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        epoch_loss = []
        iteration_cnt = 0
        for epoch in range(num_epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                optimizer.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()

                # Apply SCAFFOLD correction before optimizer step
                current_lr = optimizer.param_groups[0]['lr']  # Get the current learning rate from the optimizer
                for name, param in model.named_parameters():
                    param.grad += current_lr * (c_model_global_params[name] - c_model_local_params[name])

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Apply gradient clipping
                optimizer.step()
                if args.client_optimizer == "CosAnnealing":
                    scheduler.step()  # Update the learning rate according to the scheduler

                batch_loss.append(loss.item())
                iteration_cnt += 1

            epoch_loss.append(sum(batch_loss) / len(batch_loss) if batch_loss else 0.0)
            # Example of logging, uncomment and modify as needed
            # print(f"Client Index = {self.id}\tEpoch: {epoch}\tLoss: {epoch_loss[-1]:.6f}")
        
        return iteration_cnt



    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)  # pylint: disable=E1102

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)
        return metrics

if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import math
    import numpy as np
    import os
    import random
    from scipy.io import savemat

    from load_args import load_args
    from data_loader import data_loader
    from cifar10_resnet import resnet20
    from cifar100_densenet import densenet
    from train import train
    from evaluate import evaluate

    def main():
    
    
        args = load_args()

        # Check the availability of GPU.
        use_cuda = args.use_cuda and torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        # Set the ramdom seed for reproducibility.
        if args.reproducible:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)
            if device != torch.device("cpu"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        # Load data, note we will also call the validation set as the test set.
        print('Loading data...')
        dataset = data_loader(dataset_name=args.dataset,
                              dataroot=args.dataroot,
                              batch_size=args.batchsize,
                              val_ratio=(args.val_ratio if args.validation else 0))
        train_loader = dataset[0]
        if args.validation:
            test_loader = dataset[1]
        else:
            test_loader = dataset[2]

        # Define the model and the loss function.
        if args.dataset == 'CIFAR10':
            net = resnet20()
        elif args.dataset == 'CIFAR100':
            net = densenet(depth=100, growthRate=12, num_classes=100)
        else:
            raise ValueError("Unsupported dataset {0}.".format(args.dataset))    
        net.to(device)
        criterion = nn.CrossEntropyLoss()
        
        
        ####### compute the milestones
        if args.optim_method == 'SGD_Step_Decay' and args.milestones == []:
           print('Compute the milestone')
           decay_rate = 1 / args.alpha
           n_train = len(train_loader)
           max_iter = args.train_epochs * n_train
           n_outer = int(math.log(max_iter, decay_rate)//2)
           n_inner = max_iter // n_outer
           print('n_train, max_iter, n_outer, n_inner', n_train, max_iter,  n_outer, n_inner)
           for i in range(n_outer-1):
               args.milestones.append(n_inner*(i+1))
           print('milestones', args.milestones)




        # Train and evaluate the model.
        print("Training...")
        running_stats = train(args, train_loader, test_loader, net,
                              criterion, device)
        all_train_losses, all_train_accuracies = running_stats[:2]
        all_test_losses, all_test_accuracies = running_stats[2:]

        print("Evaluating...")
        final_train_loss, final_train_accuracy = evaluate(train_loader, net,
                                                          criterion, device)
        final_test_loss, final_test_accuracy = evaluate(test_loader, net,
                                                        criterion, device)

        # Logging results.
        print('Writing the results.')
        output = {'train_loss': all_train_losses, 'train_accuracy': all_train_accuracies, 'test_loss': all_test_losses, 'test_accuracy': all_test_accuracies}
        savemat('sgd_results.mat', output, appendmat=True)
 


        print('Finished.')

    main()

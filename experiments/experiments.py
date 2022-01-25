from ViT import *

if __name__ == "__main__":


    import glob
    import os
    import torch
    from sklearn.model_selection import train_test_split
    #from torch.utils.tensorboard import SummaryWriter
    #import pytorch_warmup as warmup
    import matplotlib.pyplot as plt
    import gc
    import wandb
    import sys

    if len(sys.argv) > 1:
        checkpoint = torch.load('fire_state_dict.pt')
    else:
        checkpoint = None

    print(checkpoint)

    torch.cuda.empty_cache()

    torch.manual_seed(6652)

    def plot_preds_on_labels(labels, outputs, ax = None):
        fig, ax = plt.subplots()

        ax.set(xlabel='preds', ylabel='vals',
            title='Line of best fit')
        #fig = plt.figure(figsize =(12,12))
        m, b = np.polyfit(outputs, labels, 1)
        if ax is None:
            ax = plt.gca()
        ax.plot(outputs, labels)
        ax.plot(outputs, outputs*m + b)
        return(fig)

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']


    #tb = SummaryWriter('/home/connor/runs')

    train_transform_fire = albumentations.Compose(
    [
        albumentations.CoarseDropout(min_holes = 1, max_holes =8, max_height=3, max_width = 3, p=.1),
        albumentations.ShiftScaleRotate(shift_limit=1, scale_limit=0.1, rotate_limit=8, p=0.15),
        #albumentations.VerticalFlip(p=.1),
        albumentations.CropAndPad (px = (-5, 5), p = .1) #augmentations.crops.transforms
    ]
    )


    train_transform_hs = albumentations.Compose(
    [
        #albumentations.CoarseDropout(min_holes = 1, max_holes =8, max_height=3, max_width = 3, p=.1),
        #albumentations.VerticalFlip(p=.1),
        albumentations.HorizontalFlip(p = .025)])
        #albumentations.ShiftScaleRotate(shift_limit=1, scale_limit=0.1, rotate_limit=8, p=0.15)])

    #Hyperparameters
    loss_func = 'mse'
    weight_decay = 0.0000025
    lrate = 0.00003
    batch_size = 5
    num_epochs = 155
    min_valid_loss = .5
    num_workers = 1  # not used in script. This is here as a reminder that cuda
                     # and cpu do not do well paired when observations are huge
    num_heads = 3
    dim_head = 490
    dim = 2870
    dropout = .1
    emb_dropout = 0.01
    depth = 5
    warm_periods = 1100
    cosine_steps = 2000
    split_frac = .1

    wandb.init(project = 'fire-net-v1', config=dict(loss_func = loss_func,
    weight_decay = weight_decay,
    lrate = lrate,
    batch_size = batch_size,
    num_epochs = num_epochs,
    min_valid_loss = min_valid_loss,
    num_workers = num_workers,
    num_heads = num_heads,
    dim_head = dim_head,
    dim = dim,
    dropout = dropout,
    emb_dropout = emb_dropout,
    depth = depth,
    warm_periods = warm_periods,
    cosine_steps = cosine_steps,
    split_frac = split_frac))



    train_list = glob.glob(os.path.join('/home/connor/Desktop/MassCrop/OutputFolder/raster','*.tif'))
    #print(train_list)
    # train_list, valid_list = train_test_split(train_list,
    #                                        test_size=0.1,
    #                                        random_state=42)


    device = torch.device('cuda')

    if(loss_func == 'quant'):
        model = ViT(num_classes = 3, learned_pool = False,
            num_heads =num_heads, dim_head = dim_head, dim = dim,
            dropout = dropout, emb_dropout = emb_dropout, depth = depth).to(device)
    else:
        model = ViT(num_classes = 1, learned_pool = False,
            num_heads = num_heads, dim_head = dim_head, dim = dim,
            dropout = dropout, emb_dropout = emb_dropout, depth = depth).to(device)



    optimizer = torch.optim.AdamW(model.parameters(), weight_decay = weight_decay, lr = lrate)
    #warmup_scheduler = warmup.ExponentialWarmup(optimizer, warmup_period=warm_periods)
    #warmup_scheduler.last_step = 1000
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = cosine_steps)


    if loss_func == 'mse':
        criterion = torch.nn.MSELoss(reduction = 'sum')
    if loss_func == 'r-squared':
        def rsq_loss(output, target):
            target_mean = torch.mean(target)
            ss_tot = torch.sum((target - target_mean) ** 2)
            ss_res = torch.sum((target - output) ** 2)
            r2 = ss_res / ss_tot
            return r2

    # fire_retriever = SingleItemRetriever(pickle_file = '/home/connor/Desktop/MassCrop/OutputFolder/tabular/tabular_inputs.pckl',
    #     raster_dir = '/home/connor/Desktop/MassCrop/OutputFolder/raster', image_loader = tifffile_loader, outcome = 'CostPerAcre')
    # hp_retriever = SingleItemRetriever(pickle_file = '/home/connor/Desktop/MassCrop/OutputFolder/tabular/tabular_inputs.pckl',
    #     raster_dir = '/home/connor/Desktop/MassCrop/OutputFolder/raster', image_loader = tifffile_loader, outcome = 'housingval_m_building_20')
    idlist = ['947fbfb1-ebea-452f-9ad5-e4fc38c05068','d0d17250-64e5-4961-a1dc-42986547c3ff']

    item1 = SingleItemRetriever(pickle_file = '/home/connor/Desktop/MassCrop/OutputFolder/tabular/tabular_inputs_res.pckl',
        raster_dir = '/home/connor/Desktop/MassCrop/OutputFolder/raster', image_loader = tifffile_loader, outcome = 'CostPerAcre', label = idlist[0])
    item2 = SingleItemRetriever(pickle_file = '/home/connor/Desktop/MassCrop/OutputFolder/tabular/tabular_inputs_res.pckl',
        raster_dir = '/home/connor/Desktop/MassCrop/OutputFolder/raster', image_loader = tifffile_loader, outcome = 'CostPerAcre', label = idlist[1])
    item1_hp = SingleItemRetriever(pickle_file = '/home/connor/Desktop/MassCrop/OutputFolder/tabular/tabular_inputs_res.pckl',
        raster_dir = '/home/connor/Desktop/MassCrop/OutputFolder/raster', image_loader = tifffile_loader, outcome = 'housingval_m_building_20', label = idlist[0])
    item2_hp = SingleItemRetriever(pickle_file = '/home/connor/Desktop/MassCrop/OutputFolder/tabular/tabular_inputs_res.pckl',
        raster_dir = '/home/connor/Desktop/MassCrop/OutputFolder/raster', image_loader = tifffile_loader, outcome = 'housingval_m_building_20', label = idlist[1])
    saliency_data_list = [item1.ret(), item2.ret(), item1_hp.ret(), item2_hp.ret()]

    for fold in range(5,6,1):

        fire_data_train = FireManagement_Dataset(pickle_file = '/home/connor/Desktop/MassCrop/OutputFolder/tabular/tabular_inputs_res.pckl',
                raster_dir= '/home/connor/Desktop/MassCrop/OutputFolder/raster',
                outcome = 'CostPerAcre', transform = train_transform_fire, image_loader = tifffile_loader, train = True, log_label = False, fold = fold)

        fire_data_valid = FireManagement_Dataset(pickle_file = '/home/connor/Desktop/MassCrop/OutputFolder/tabular/tabular_inputs_res.pckl',
                raster_dir= '/home/connor/Desktop/MassCrop/OutputFolder/raster',
                outcome = 'CostPerAcre', transform = None, image_loader = tifffile_loader, train = False, log_label = False,fold = fold)

        # fire_data_train_housing = FireManagement_Dataset(pickle_file = '/home/connor/Desktop/MassCrop/OutputFolder/tabular/tabular_inputs_res.pckl',
        #     raster_dir= '/home/connor/Desktop/MassCrop/OutputFolder/raster',
        #     outcome = 'housingval_m_building_20', transform = train_transform_hs, image_loader = tifffile_loader, train = True, log_label = False, fold = fold)

        # fire_data_valid_housing = FireManagement_Dataset(pickle_file = '/home/connor/Desktop/MassCrop/OutputFolder/tabular/tabular_inputs_res.pckl',
        #     raster_dir= '/home/connor/Desktop/MassCrop/OutputFolder/raster',
        #     outcome = 'housingval_m_building_20', transform = None, image_loader = tifffile_loader, train = False, log_label = False,fold = fold)

        # train_loader_hs = torch.utils.data.DataLoader(fire_data_train_housing, batch_size = batch_size,
        #     shuffle = True)#, pin_memory = True)

        # valid_loader_hs = torch.utils.data.DataLoader(fire_data_valid_housing, batch_size = batch_size,
        #     shuffle = False)#, pin_memory = True)

        train_loader = torch.utils.data.DataLoader(fire_data_train, batch_size = batch_size,
            shuffle = True)#, pin_memory = True)

        valid_loader = torch.utils.data.DataLoader(fire_data_valid, batch_size = batch_size,
            shuffle = False)#, pin_memory = True)

        #data, tabular, label= next(iter(train_loader))
        #print(tabular)
        #data = data[:,[0,17,9,10,22,5,16],:,:]


        #print(data[data.isnan()])
        #tb.add_graph(model, [data.to(device), tabular.to(device)])

        smallest_valid_loss = 100000

        tablist = ['Hours', 'EACC', 'GBCC', 'NRCC',
                'NWCC', 'ONCC', 'OSCC', 'RMCC', 'SACC',
                'SWCC', 'Other',
                'Suppress', 'Month', 'Resources']

        if checkpoint is not None:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            loss = checkpoint['loss']
            del(checkpoint)
        else:
            start_epoch = 0

        for epoch in range(start_epoch, num_epochs):
            gc.collect()
            # if fold == 1:
            #     continue

            optimizer.zero_grad(set_to_none = True)
            gc.collect()
            model.train() # set to train mode with trainable weights

            epoch_loss = 0.0
            train_loss = 0.0


            dat_len = len(train_loader)

            for i,dat in enumerate(train_loader):
                data, tabular, label, _ = dat




                if(epoch < 5):
                    tabular = torch.randn(tabular.size())
                    #warmup_scheduler.dampen()


                data, label, tabular = mixup_data(Images = data, y = label, tab = tabular, p = .1, alpha = .1)
                #print(f'Irwin ID is {id_num}')
                data = data.to(device)
                tabular = tabular.to(device)
                label = label.to(device)


                for param in model.parameters():
                    param.grad = None

                output = model(data, tabular = tabular)
                if loss_func == 'quant':
                    loss = QuantileLoss(output, label)
                elif loss_func == 'mse':
                    loss = criterion(output.squeeze(), label.squeeze())
                elif loss_func == 'r-squared':
                    loss = rsq_loss(output, label)

                gc.collect()
                loss.backward()
                epoch_loss += loss.detach().item()
                train_loss += loss.detach().item()
                optimizer.step()
                lr_scheduler.step()
                lr = lr_scheduler.get_last_lr()
                wandb.log({"learning rate":lr})

                if((i+1) % 4 == 0):
                    optimizer.zero_grad(set_to_none = True)


                # if i % 100 == 99:

                    # tb.add_scalar('training loss',
                    #             train_loss / 100,
                    #             epoch * len(train_loader) + i)

                    # ...log a Matplotlib Figure showing the model's predictions on a
                    # random mini-batch


                    # tb.add_figure('predictions vs. actuals',
                    #             plot_preds_on_labels(output.cpu().detach().numpy(), label.cpu().detach().numpy()),
                    #             global_step=epoch * len(train_loader) + i)
                    #train_loss = 0.0

                #print(f"Prediction for items: {output}")
                #print(f"Items values: {label}")

                wandb.log({f"intra-epoch loss: Epoch {epoch + 1}, fold {fold} fire": train_loss/(i+1)})



            #tb.add_scalar("Loss/train", epoch_loss, epoch)
            gc.collect()
            print(
                f"Epoch : {epoch+1} - loss : {epoch_loss:.4f}" #"- acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
            )
            #wandb.log({"training loss": epoch_loss, 'Epoch': epoch+1}, commit = False)


            with torch.no_grad():
                model.eval()

                valid_loss = 0
                for i, dat_v in enumerate(valid_loader):
                      # switch to evaluation mode

                    data, tabular, label, _ = dat_v

                    data = data.to(device)
                    tabular = tabular.to(device)
                    label = label.to(device)

                    guess = model(data, tabular)
                    loss = criterion(guess.squeeze(),label.squeeze())
                    valid_loss += loss.detach().item()  # accumulate SSE

            # tb.add_scalar("Loss/valid", valid_loss, epoch)
            if smallest_valid_loss > valid_loss / len(valid_loader):
                smallest_valid_loss = valid_loss / len(valid_loader)
                mod_path = 'model_best_epoch_firecost_fold_' + str(fold) + '.pth'
                torch.save(model, mod_path)
            PATH = 'fire_state_dict.pt'
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_dict':lr_scheduler.state_dict(),
            'loss': train_loss,
            'smallest_valid_loss': smallest_valid_loss
            }, PATH)


            print(f'Epoch {epoch+1} \t\t Training Loss: {epoch_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(valid_loader)}')
            wandb.log({f"valid loss, fire, fold {fold}": valid_loss / len(valid_loader), f"train loss, fire, fold {fold}": epoch_loss / len(train_loader)})


            tablist = ['Hours', 'EACC', 'GBCC', 'NRCC',
                'NWCC', 'ONCC', 'OSCC', 'RMCC', 'SACC',
                'SWCC', 'Other',
                'Suppress', 'Month', 'Resources']

            device = torch.device('cuda')
            if fold == 1:
                data, tabular, label = saliency_data_list[0]
                # Retrieve output from the image

                #tabular = tabular[None,:]

                data = data.unsqueeze(0)



                data = data.to(device)
                data.requires_grad_()
                data.retain_grad()
                tabular = tabular.to(device)
                tabular.requires_grad_()
                tabular.retain_grad()



                output = model(data, tabular = tabular)

                # Catch the output
                output_idx = output.argmax()
                output_max = output[0,output_idx]
                #print(output_max)
                output_max.backward()

                #print(data.grad)
                saliency, _ = torch.max(data.grad.data.abs(), dim=1)
                saliency = saliency.reshape(1001, 1001)
                sal_data = tabular.grad.data.abs()

                #print(sal_data)
                sal_data = sal_data.reshape(len(tablist))

                cmap = plt.get_cmap("hot")
                rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
                fig, (ax, ax2) = plt.subplots(1, 2, figsize = (15,15))
                ax.bar(tablist,sal_data.cpu().numpy(), color =cmap(rescale(sal_data.cpu().numpy())))
                fig.canvas.draw()
                ax.set_xticklabels(tablist, rotation = 45)
                ax2.imshow(saliency.cpu(), cmap='hot')
                ax2.axis('off')
                #plt.tight_layout()
                fig.suptitle('Saliency Map for Image and Tabular Data')
                wandb.log({f"firecost saliency log fold {fold}, epoch {epoch + 1}": plt, "epoch":epoch+1})
                gc.collect()

            if fold == 2:
                data, tabular, label = saliency_data_list[1]
               # Retrieve output from the image

                #tabular = tabular[None,:]
                data = data.unsqueeze(0)



                data = data.to(device)
                data.requires_grad_()
                data.retain_grad()
                tabular = tabular.to(device)
                tabular.requires_grad_()
                tabular.retain_grad()



                output = model(data, tabular = tabular)

                # Catch the output
                output_idx = output.argmax()
                output_max = output[0,output_idx]
                #print(output_max)
                output_max.backward()

                #print(data.grad)
                saliency, _ = torch.max(data.grad.data.abs(), dim=1)
                saliency = saliency.reshape(1001, 1001)
                sal_data = tabular.grad.data.abs()

                #print(sal_data)
                sal_data = sal_data.reshape(len(tablist))

                cmap = plt.get_cmap("hot")
                rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
                fig, (ax, ax2) = plt.subplots(1, 2, figsize = (15,15))
                ax.bar(tablist,sal_data.cpu().numpy(), color =cmap(rescale(sal_data.cpu().numpy())))
                fig.canvas.draw()
                ax.set_xticklabels(tablist, rotation = 45)
                ax2.imshow(saliency.cpu(), cmap='hot')
                ax2.axis('off')
                fig.suptitle('Saliency Map for Tabular and Raster Data, Holiday Farm Fire')

                wandb.log({f"firecost saliency log fold {fold}, epoch {epoch + 1}": plt, "epoch":epoch+1})
                gc.collect()
            del data
            del tabular
            plt.clf()
            gc.collect()
        # if fold != 1:
            if min_valid_loss > valid_loss:
                print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
                min_valid_loss = valid_loss
                # Saving State Dict
                torch.save(model.state_dict(), 'saved_model.pth')
            #'model_finished_training_firecost_fold_' + str(fold) + '.pth'
        mod_path = 'model_finished_training_firecost_fold_' + str(fold) + '.pth'
        torch.save(model, mod_path)

        stdct_path = 'stdct_finished_training_firecost_fold_' + str(fold) + '.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            }, stdct_path)

        irwinidls = []
        guesses = []
        valid_out_fire = pd.DataFrame()
        with torch.no_grad():
            model.eval()
            for i, dat in enumerate(valid_loader):
              # switch to evaluation mode

                data, tabular, label, irwinid = dat

                data = data.to(device)
                tabular = tabular.to(device)
                label = label.to(device)

                guess = model(data, tabular)
                #valid_out_fire.append(pd.DataFrame(tabular.detach().cpu().tolist(), columns = tabular_pd))
                irwinidls.extend(irwinid)
                guesses.extend(guess.cpu().detach().tolist())
                loss = criterion(guess.squeeze(),label.squeeze())
                valid_loss += loss.detach().item()  # accumulate SSE

        valid_out_fire['guesses_fire'] = guesses
        valid_out_fire['IrwinID'] = irwinidls

        valid_out_fire.to_csv('train_tabular_data_with_preds_v2_fire_fold{}.csv'.format(fold))


        # #torch.cuda.empty_cache()
        # model = ViT(num_classes = 1, learned_pool = False,
        #     num_heads = num_heads, dim_head = dim_head, dim = dim,
        #     dropout = dropout, emb_dropout = emb_dropout, depth = depth).to(device)





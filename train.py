import os,json,random
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from model.EcgClassifier import EcgClassifer
from model.Generator import Ecgedit
from model.BaseModel import VQSeparator
from dataset.PTBXLdataset import PTBXLOridataset as ECGDataset
from dataset.PTBXLdataset import collect_fn_ori, PTBXLTestClsdataset
import utils
from torch.utils.tensorboard import SummaryWriter
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
torch.manual_seed(41)
torch.cuda.manual_seed(41)


def load_matching_parameters(model, pretrained_state_dict):
    model_dict = model.state_dict()
    new_state_dict = {}

    for name, param in pretrained_state_dict.items():
        if name in model_dict and param.shape == model_dict[name].shape:
            new_state_dict[name] = param

    model.load_state_dict(new_state_dict, strict=False)


def print_log(text, path):
    print(text)
    with open(os.path.join(path, 'log.txt'), 'a') as f:
        f.write(text+"\n")


def save_ckpt(g_state, d_state, f_state,is_best, model_save_dir):
    torch.save(g_state, os.path.join(model_save_dir, 'current.pth'))
    torch.save(d_state, os.path.join(model_save_dir, 'current_d.pth'))
    torch.save(f_state, os.path.join(model_save_dir, 'current_f.pth'))
    if is_best:
        torch.save(g_state, os.path.join(model_save_dir, 'best.pth'))
        torch.save(d_state, os.path.join(model_save_dir, 'best_d.pth'))
        torch.save(f_state, os.path.join(model_save_dir, 'best_f.pth'))


def reset_grad(d_optimizer, g_optimizer):
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()


def train_epoch(Editor, classifier, epoch, logger, criterion, train_dataloader, model_save_dir, show_interval=10):
    Editor.gen.train()
    Editor.dis.train()
    Editor.disease_model.train()
    f1_meter, acc_meter, loss_meter, it_count = 0, 0, 0, 0
    f1, acc = 0, 0
    for idx, batch in enumerate(train_dataloader):
        
        ab_ecg = batch[0].to(device).float()[:, :, 452:4548]
        ab_descrip = batch[1]
        target = batch[2].to(device)
        normal_ecg = batch[3].to(device).float()[:, :, 452:4548] # choose Normal
        norm_descrip = batch[4]
        batch_size = ab_ecg.shape[0]

        d_loss, d_ori_loss, d_cls_loss = Editor.optimize_discriminator(ab_ecg, ab_descrip, normal_ecg, norm_descrip, 10, 1)
        g_loss, g_f_loss, g_rec_loss, g_cls_loss, g_q_loss, indices = Editor.optimize_generator(ab_ecg, ab_descrip, normal_ecg, norm_descrip, 10, 1)

        logger.add_scalar('D_loss', d_loss, global_step=idx+(epoch-1)*len(train_dataloader))
        logger.add_scalar('D_cls_loss', d_cls_loss, global_step=idx+(epoch-1)*len(train_dataloader))
        logger.add_scalar('D_ori_loss', d_ori_loss, global_step=idx+(epoch-1)*len(train_dataloader))
        logger.add_scalar('G_loss', g_loss, global_step=idx+(epoch-1)*len(train_dataloader))
        logger.add_scalar('G_gan_loss', g_loss, global_step=idx+(epoch-1)*len(train_dataloader))
        logger.add_scalar('G_cls_loss', g_cls_loss, global_step=idx+(epoch-1)*len(train_dataloader))
        logger.add_scalar('G_rec_loss', g_rec_loss, global_step=idx+(epoch-1)*len(train_dataloader))
        logger.add_scalar('G_ori_loss', g_f_loss, global_step=idx+(epoch-1)*len(train_dataloader))
        logger.add_scalar('G_q_loss', g_q_loss, global_step=idx+(epoch-1)*len(train_dataloader))


        if (epoch>50 and idx % 30 == 0) or (epoch == 1 and idx % 200 == 0):
            disease, const_disease, q_loss_s, prefix_s, mask = Editor.disease_model(ab_ecg[:, :12], ab_ecg[:, 12:], ab_descrip)
            disease_r, const_r, q_loss_r, prefix_r,_  = Editor.disease_model(normal_ecg[:, :12], normal_ecg[:, 12:], norm_descrip)
            disease = disease.detach()
            const_r = const_r.detach()
            ecg_edit = Editor.gen(disease, 10, 1, const_input=const_r).detach()
            mask = mask.detach()
            output = classifier(ecg_edit)
            loss_classifier = criterion[1](output, target)
            loss = loss_classifier.item()
            logger.add_image('Mask', mask[0].unsqueeze(0), global_step=idx+(epoch-1)*len(train_dataloader), dataformats='CHW')
            logger.add_scalar('Cls_loss', loss_classifier.item(), global_step=idx+(epoch-1)*len(train_dataloader))
            it_count += 1
            f1, acc = utils.calc_f1_acc_one_hot(target, torch.sigmoid(output))
            f1_meter += f1
            acc_meter += acc
    
        # loss_meter += loss

        if idx != 0 and idx % show_interval == 0:
            print_log(f"Indices {indices}", model_save_dir) # <=1 ==>collaspe
            print_log("Iter %d,G_loss:%.3e, D_loss:%.3e f1:%.3f acc:%.3f" % (idx, g_loss, d_loss, f1, acc), model_save_dir)
            print_log(f"G_q_loss:{g_q_loss} G_cls_loss:{g_cls_loss}, G_ori_loss:{g_f_loss}, G_rec_loss:{g_rec_loss}", model_save_dir)
            print_log(f"D_cls_loss:{d_cls_loss}, D_ori_loss:{d_ori_loss}", model_save_dir)
    return loss_meter / len(train_dataloader), f1_meter / (it_count+0.000001), acc_meter / (it_count+0.000001)


def val_epoch(Editor, classifier, epoch, logger, criterion, val_dataloader, threshold=0.5, save_dir='./'):
    Editor.gen.eval()
    Editor.dis.eval()
    f1_meter, acc_meter, loss_meter, it_count = 0, 0, 0, 0
    with torch.no_grad():
        for idx, batch in enumerate(val_dataloader):
            ab_ecg = batch[0].to(device).float()[:, :, 452:4548]
            ab_descrip = batch[1]
            target = batch[2].to(device)
            normal_ecg = batch[3].to(device).float()[:, :, 452:4548]
            normal_type = batch[5].to(device)
            norm_descrip = batch[4]
            batch_size = ab_ecg.shape[0]

            # forward
            disease, const_disease, q_loss_s, prefix_s, mask = Editor.disease_model(ab_ecg[:, :12], ab_ecg[:, 12:], ab_descrip)
            disease_r, const_r, q_loss_r, prefix_r,_  = Editor.disease_model(normal_ecg[:, :12], normal_ecg[:, 12:], norm_descrip)
            disease = disease.detach()
            const_r = const_r.detach()
            ecg_edit = Editor.gen(disease, 10, 1, const_input=const_r)
            ecg_edit = ecg_edit.detach()
            mask = mask.detach()
            output = classifier(ecg_edit)
            loss = criterion[1](output, target)
            logger.add_image('Val_Mask', mask[0].unsqueeze(0), global_step=idx+(epoch-1)*len(val_dataloader), dataformats='CHW')
            logger.add_scalar('Val_Cls_loss', loss.item(), global_step=idx+(epoch-1)*len(val_dataloader))
            logger.add_scalar('Val_perplexit', prefix_s.detach(), global_step=idx+(epoch-1)*len(val_dataloader))
            loss_meter += loss.item()
            it_count += 1
            output = torch.sigmoid(output)
            f1, acc = utils.calc_f1_acc_one_hot(target, output, threshold)
            f1_meter += f1
            acc_meter += acc
        
        if epoch % 50 == 0 and epoch > 10:
            utils.save_ecg_image(ecg_edit[:5], f'{save_dir}/epoch_ecg_{epoch}/')
            utils.save_ecg_image(normal_ecg[:5, :12], f'{save_dir}/epoch_ecg_{epoch}_input/')
    return loss_meter / it_count, f1_meter / it_count, acc_meter / it_count


def GenerateECG(Editor, classifier, logger, test_loader, val_dataloader, threshold=0.5, save_dir='./'):
    Editor.gen.eval()
    Editor.disease_model.eval()
    f1_meter, acc_meter, loss_meter, it_count = 0, 0, 0, 0
    gen_ecg = []
    gen_cls = []
    gen_id = []
    gen_des = []
    classes = ['HYP', 'MI', 'CD', 'STTC', 'NORM']
    cls_map = {classes[id]: id for id in range(len(classes))}
    weight = {classes[id]: 0 for id in range(len(classes))}
    p_num = {}
    for ind, batch_norm in enumerate(test_loader):
        ecg_ori = batch_norm[0].to(device).float()
        ori_cls = batch_norm[1].to(device)
        ori_des = batch_norm[3]
        patient_id = batch_norm[2].to(device)
        ori_cls = ori_cls.repeat(128)
        
        patient_id = patient_id.repeat(128)
        
        with torch.no_grad():
            print(f"Paient_id {patient_id}")
            num_per = 0
            for idx, batch in enumerate(val_dataloader): # choose some item for generate ecg
                if num_per >= 200:
                    break
                ab_ecg = batch[0].to(device).float()
                ab_descrip = batch[1]
                target = batch[2].to(device)

                choose_item = target.cpu().numpy()!=4
                choose_item = np.argwhere(choose_item == True).squeeze(1)
                print(choose_item)
                if num_per+len(choose_item) > 200:
                    choose_item = random.sample(list(choose_item), 200-num_per)
                num_per+=len(choose_item)
                choose_item = torch.tensor(choose_item, device=ab_ecg.device).long()

                normal_ecg = ecg_ori[:, :, 452:4548].repeat(ab_ecg.shape[0], 1, 1)
                normal_des = list(ori_des)*ab_ecg.shape[0]
                # forward
                disease, const_disease, q_loss_s, prefix_s, mask = Editor.disease_model(ab_ecg[:, :12], ab_ecg[:, 12:], ab_descrip)
                disease_r, const_r, q_loss_r, prefix_r,_  = Editor.disease_model(normal_ecg[:, :12], normal_ecg[:, 12:], normal_des)
                disease = disease.detach().to(device)
                const_r = const_r.detach().to(device)
                ecg_edit = Editor.gen(disease, 10, 1, const_input=const_r)


                gen_ecg.append(ecg_edit[choose_item].cpu().detach().numpy())
                gen_cls.append(target[choose_item].cpu().detach().numpy())
                gen_id.append(patient_id[choose_item].cpu().numpy())
                gen_des.extend([ab_descrip[i] for i in choose_item])
                output = classifier(ecg_edit)

                it_count += 1
                output = torch.sigmoid(output)
                f1, acc = utils.calc_f1_acc_one_hot(target, output, threshold)
                f1_meter += f1
                acc_meter += acc
                for i in target.cpu().detach().numpy():
                    weight[classes[i]] += 1



    utils.save_ecg_image(ecg_edit[:10], f'{save_dir}/')
    utils.save_ecg_image(ab_ecg[:10], f'{save_dir}/ab/')
    ecg = np.concatenate(gen_ecg, axis=0)
    clss = np.concatenate(gen_cls, axis=0)
    id = np.concatenate(gen_id, axis=0)
    np.save(save_dir+'/ecg.npy', ecg)
    df = pd.DataFrame({'patient_id': id, 'detail_superclass': clss, 'report': gen_des})
    df.to_csv(save_dir+'/ecg.csv')
        

    return loss_meter / it_count, f1_meter / it_count, acc_meter / it_count


def train(args):
    model_save_dir = os.path.join(args.work_dir, args.model_name, time.strftime("%Y%m%d%H%M"))
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
    with open(model_save_dir+"/config.json", mode="w") as f:
        json.dump(args.__dict__, f, indent=4)

    print_log(f"{args.model_name}: {args.detail}", model_save_dir)
    logger = SummaryWriter(log_dir=model_save_dir, flush_secs=2)
    model = VQSeparator(embedding_dim=512, context_dim=1024, resolution=4096, language_model='model/Bio_ClinaBert')
    if args.clip_dir:
        model.load_state_dict(torch.load(args.clip_dir, map_location='cpu')['state_dict'])
        print(f'Model Load from {args.clip_dir}....')

    Editor = Ecgedit(disease_model=model, structure='linear', resolution=4096, num_channels=12, latent_size=1024, dlatent_size=2048, fmap_max=512,
                   loss="logistic", const_input_dim=0, device=device, n_classes=5, logger=logger, lr=args.lr)
    classifier = EcgClassifer(classifer=args.classifer, num_classes=5, load_pretrain='./ckpt2/resnet34_202311301714sclc-5/best_w.pth')
    best_f1 = -1
    best_acc = -1
    lr = args.lr
    start_epoch = 1
    stage = 1
    if args.resume:
        state = torch.load(f'{args.resume}/current.pth', map_location='cpu')
        d_state = torch.load(f'{args.resume}/current_d.pth', map_location=torch.device('cpu'))
        f_state = torch.load(f'{args.resume}/current_f.pth', map_location='cpu')
        Editor.gen.load_state_dict(state['state_dict'])
        Editor.dis.load_state_dict(d_state['state_dict'])
        load_matching_parameters(Editor.disease_model, f_state['state_dict'])
        # Editor.gen_optim.load_state_dict(state['optimizer'])
        # Editor.dis_optim.load_state_dict(d_state['optimizer'])
        start_epoch = state['epoch']
        # load_matching_parameters(Editor.gen, state['state_dict'])
        # load_matching_parameters(Editor.dis, d_state['state_dict'])
        print_log(f"train with resume weight val_f1 {state['f1']}", model_save_dir)

    print(f"Model init....")
    if args.pretrain:
        state = torch.load(args.resume, map_location='cpu')
        d_state = torch.load(args.resume.replace('.pth', '_d.pth'), map_location='cpu')
        Editor.gen.load_state_dict(state['state_dict'])
        Editor.dis.load_state_dict(d_state['state_dict'])
        print_log(f"train with pretrained weight val_f1 {state['f1']}", model_save_dir)
    
    model = model.to(device)
    classifier = classifier.to(device)
    # utils.freeze_model(model)
    utils.freeze_model(classifier)

    train_dataset = ECGDataset(args.data_root+'reprepared/', True, classifier=True, choose_norm=True)
    train_weight = train_dataset.sample_weight(path='weight_5_scls.json')
    train_sampler = WeightedRandomSampler(train_weight, len(train_weight))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                  sampler=train_sampler, num_workers=args.num_workers, collate_fn=collect_fn_ori, drop_last=True)
    val_dataset = ECGDataset(args.data_root+'reprepared/', False, classifier=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, 
                                num_workers=args.num_workers, collate_fn=collect_fn_ori, drop_last=True)
    print_log(f"train_datasize {len(train_dataset)} val_datasize {len(val_dataset)}", model_save_dir)
    
    # optimizer and loss

    print(f"Optimizer init")
    criterion = [nn.BCELoss(), nn.CrossEntropyLoss(), nn.MSELoss()]
    


    print(f"Start Training......")
    for epoch in range(start_epoch, args.max_epoch + 1):
        since = time.time()
        train_loss, train_f1, train_acc = train_epoch(Editor, classifier, 
                                                          epoch, logger, 
                                                          criterion, 
                                                          train_dataloader, model_save_dir, show_interval=10)
        val_loss, val_f1, val_acc = val_epoch(Editor, classifier, epoch, logger, criterion, val_dataloader, save_dir=model_save_dir)
        print_log('#epoch:%02d stage:%d train_loss:%.3e train_f1:%.3f  train_acc:%.3f val_loss:%0.3e val_f1:%.3f val_acc:%.3f time:%s\n'
              % (epoch, stage, train_loss, train_f1, train_acc, val_loss, val_f1, val_acc, utils.print_time_cost(since)), model_save_dir)
        logger.add_scalar('train_loss', train_loss, global_step=epoch)
        logger.add_scalar('train_f1', train_f1, global_step=epoch)
        logger.add_scalar('train_acc', train_acc, global_step=epoch)
        logger.add_scalar('val_loss', val_loss, global_step=epoch)
        logger.add_scalar('val_f1', val_f1, global_step=epoch)
        logger.add_scalar('val_acc', val_acc, global_step=epoch)
        state, d_state, f_state = Editor.save_ckpt(epoch, val_loss, val_f1, val_acc, stage)
        save_ckpt(state, d_state, f_state, best_acc < val_acc, model_save_dir)
        best_f1 = max(best_f1, val_f1)
        best_acc = max(best_acc, val_acc)
        if epoch in args.stage_epoch:
            stage += 1
            lr /= args.lr_decay
            best_w = os.path.join(model_save_dir, args.best_w)
            Editor.gen.load_state_dict(torch.load(best_w)['state_dict'])
            Editor.dis.load_state_dict(torch.load(best_w.replace('.pth', '_d.pth'))['state_dict'])
            print_log(f"*" * 10 + "step into stage%02d lr %.3ef" % (stage, lr), model_save_dir)
            # utils.adjust_learning_rate(g_optimizer, lr) 


def Gen_test(args):
    model_save_dir = os.path.join(args.work_dir, args.model_name, time.strftime("%Y%m%d%H%M"))
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
    with open(model_save_dir+"/config.json", mode="w") as f:
        json.dump(args.__dict__, f, indent=4)

    print_log(f"{args.model_name}: {args.detail}", model_save_dir)
    logger = Logger(logdir=model_save_dir, flush_secs=2)
    model = VQSeparator(embedding_dim=512, context_dim=1024, resolution=4096, language_model='model/Bio_ClinaBert')
    Editor = Ecgedit(disease_model=model, structure='linear', resolution=4096, num_channels=12, latent_size=1024, dlatent_size=2048, fmap_max=512,
                   loss="logistic", const_input_dim=0, device=device, n_classes=5, logger=logger, use_w3=False, lr=args.lr)
    
    if args.clip_dir:
        model.load_state_dict(torch.load(args.clip_dir, map_location='cpu')['state_dict'])
        print(f'Model Load from {args.clip_dir}....')
    classifier = EcgClassifer(classifer=args.classifer, num_classes=5, load_pretrain='./ckpt2/resnet34_202311301714sclc-5/best_w.pth')
    

    print(f"Model init....")
    if args.pretrain:
        state = torch.load(args.resume, map_location='cpu')
        print(state['state_dict'].keys())
        d_state = torch.load(args.resume.replace('.pth', '_d.pth'), map_location='cpu')
        f_state = torch.load(args.resume.replace('.pth', '_f.pth'), map_location='cpu')
        Editor.gen.load_state_dict(state['state_dict'])
        Editor.dis.load_state_dict(d_state['state_dict'])
        load_matching_parameters(Editor.disease_model, f_state['state_dict'])
        # Editor.disease_model.load_state_dict(f_state['state_dict'])
        print_log(f"train with pretrained weight val_f1 {state['f1']}", model_save_dir)
    else:
        print(f'Model did not pretrained!')
        return
    
    model = model.to(device)
    classifier = classifier.to(device)
    utils.freeze_model(Editor.disease_model)
    utils.freeze_model(classifier)
    utils.freeze_model(Editor.dis)
    utils.freeze_model(Editor.gen)
    # Editor = nn.DataParallel(Editor)
    # classifier = nn.DataParallel(classifier)
    # data
    train_dataset = ECGDataset(args.data_root+'reprepared/', True, classifier=True)
    train_weight = train_dataset.sample_weight(path='weight_5_scls.json')
    train_sampler = WeightedRandomSampler(train_weight, len(train_weight))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                  sampler=train_sampler, num_workers=args.num_workers, collate_fn=collect_fn_ori, drop_last=True)
    # val_dataset = ECGDataset(args.data_root+'reprepared/', False, classifier=True)
    # test_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, 
    #                             num_workers=args.num_workers, collate_fn=collect_fn_ori, drop_last=True)
    # generator
    test_dataset = PTBXLTestClsdataset(args.data_root+'reprepared/', True, choose=['NORM',], 
                                       train_lis=['./PublicDataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/reprepared/Patient_Select_145_sclc_X_half1'])  # choose normal
    test_dataloader = DataLoader(test_dataset, batch_size=1, 
                                num_workers=args.num_workers, drop_last=True)
    print_log(f"train_datasize {len(train_dataset)} test_datasize {len(test_dataset)}", model_save_dir)
    
    # optimizer and loss

    
    print(f"Start Generate......")
    val_loss, val_f1, val_acc = GenerateECG(Editor, classifier, logger, test_dataloader,
                                          train_dataloader, 
                                          save_dir=model_save_dir)



if __name__ == '__main__':


    from config.parse import get_parse
    
    parser, args = get_parse()

    import yaml
    with open(args.config, 'r') as f:
        configs = yaml.safe_load(f)
        for key,value in configs.items():
            parser.add_argument("--"+key, default=value, type=type(value))
        
    configs = parser.parse_args()
    print(configs)
    train(configs)
    # configs.pretrain = True
    # configs.resume = 'work_dir/best.pth'
    # Gen_test(configs)
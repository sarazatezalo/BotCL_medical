import torch
import torch.nn.functional as F
from utils.record import AverageMeter, ProgressMeter, show
from model.contrast.loss import batch_cpt_discriminate, att_consistence, quantization_loss, att_area_loss


def cal_acc(preds, labels):
    with torch.no_grad():
        pred = preds.argmax(dim=-1)
        acc = torch.eq(pred, labels).sum().float().item() / labels.size(0)
        return acc


def train(args, model, device, loader, rec_loss, optimizer, epoch):
    recon_losses = AverageMeter('Reconstruction Loss', ':.4')
    # att_losses = AverageMeter('Att Loss', ':.4')
    pred_losses = AverageMeter('Pred Loss', ':.4')
    batch_dis_losses = AverageMeter('Dis_loss_batch', ':.4')
    consistence_losses = AverageMeter('Consistence_loss', ':.4')
    q_losses = AverageMeter('Q_loss', ':.4')
    pred_acces = AverageMeter('Acc', ':.4')
    progress = ProgressMeter(len(loader),
                             [recon_losses, pred_losses, pred_acces, batch_dis_losses, consistence_losses, q_losses],
                             prefix="Epoch: [{}]".format(epoch))

    # Regular training of the model (CNN for obtaining features of an image)
    model.train()
    # Additional losses integrated in the pipeline
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        cpt, pred, out, att, update = model(data)

        # Softmax cross-entropy for target classification loss
        loss_pred = F.nll_loss(F.log_softmax(pred, dim=1), label)
        acc = cal_acc(pred, label)
        # Reconstruction loss of decoder (takes t as input and reconstructs the original image)
        reconstruction_loss = rec_loss(out.view(data.size(0), 1, 28, 28), data)
        # Quatization loss: to assure closeness of t values to 0 or 1
        quantity_loss = quantization_loss(cpt)
        # Mutual distinctiveness of concepts 
        batch_dis_loss = batch_cpt_discriminate(update, att)
        # Individual consisteny of a feature in different images
        consistence_loss = att_consistence(update, att)
        att_loss = att_area_loss(att)  # attention loss used to prevent overflow

        recon_losses.update(reconstruction_loss.item())
        # att_losses.update(att_loss.item())
        pred_losses.update(loss_pred.item())
        pred_acces.update(acc)
        q_losses.update(quantity_loss.item())
        batch_dis_losses.update(batch_dis_loss.item())
        consistence_losses.update(consistence_loss.item())

        loss_total = args.weak_supervision_bias * reconstruction_loss + args.att_bias * att_loss + loss_pred + args.quantity_bias * quantity_loss + \
                     args.distinctiveness_bias * batch_dis_loss + args.consistence_bias * consistence_loss

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            progress.display(batch_idx)


@torch.no_grad()
def evaluation(model, device, loader, rec_loss):
    model.eval()
    record_res = 0.0
    record_att = 0.0
    accs = 0
    L = len(loader)
    preds = []
    labels = []

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        cpt, pred, out, att, update = model(data)

        preds.append(pred)
        labels.append(label)

        acc = cal_acc(pred, label)
        reconstruction_loss = rec_loss(out.view(data.size(0), 28, 28), data)
        record_res += reconstruction_loss.item()
        att_loss = att_area_loss(att)
        record_att += att_loss.item()
        accs += acc
    return round(record_res/L, 4), round(record_att/L, 4), round(accs/L, 4), preds, labels


def vis_one(model, device, loader, epoch=None, select_index=0):
    data, label = iter(loader).next()
    img_orl = data[select_index]
    img = img_orl.unsqueeze(0).to(device)
    pred = model(img)[2].view(28, 28).cpu().detach().numpy()
    show(img_orl.numpy()[0], pred, epoch)

import torch.utils.data
from evaluate import Evaluation
from loss_function import loss_function
from loss_function import cs

def train_model(train_data, dev_data, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    lasttime = time.time()
    for epoch in range(1, 11):
        print("-------------\nEpoch {}:\n".format(epoch))

        loss = run_epoch(train_data, True, model, optimizer)
        
        #return loss

        print('Train loss: {:.6f}'.format(loss))
        torch.save(model, "model{}".format(epoch))
        
        (MAP, MRR, P1, P5) = run_epoch(dev_data, False, model, optimizer)
        print('Val MAP: {:.6f}, MRR: {:.6f}, P1: {:.6f}, P5: {:.6f}'.format(MAP, MRR, P1, P5))
        
        print('This epoch took: {:.6f}'.format(time.time() - lasttime))
        lasttime = time.time()


def run_epoch(data, is_training, model, optimizer):
    '''
    Train model for one pass of train data, and return loss, acccuracy
    '''
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=20,
        shuffle=True,
        num_workers=4,
        drop_last=False)

    losses = []

    if is_training:
        model.train()
    else:
        model.eval()

    for batch in data_loader:
        pid_title = torch.unsqueeze(Variable(batch['pid_title']), 1)
        pid_body = torch.unsqueeze(Variable(batch['pid_body']), 1)
        rest_title = Variable(batch['rest_title'])
        rest_body = Variable(batch['rest_body'])

        pid_title_pad = torch.unsqueeze(Variable(batch['pid_title_pad']), 1)
        pid_body_pad = torch.unsqueeze(Variable(batch['pid_body_pad']), 1)
        rest_title_pad = Variable(batch['rest_title_pad'])
        rest_body_pad = Variable(batch['rest_body_pad'])
        
        pid_title, pid_body = pid_title.cuda(), pid_body.cuda()
        rest_title, rest_body = rest_title.cuda(), rest_body.cuda()
        pid_title_pad, pid_body_pad = pid_title_pad.cuda(), pid_body_pad.cuda()
        rest_title_pad, rest_body_pad = rest_title_pad.cuda(), rest_body_pad.cuda()
        
        if is_training:
            optimizer.zero_grad()
        
        pt = model(pid_title)
        pb = model(pid_body)
        rt = model(rest_title)
        rb = model(rest_body)

        # we need to take the mean pooling taking into account the padding
        # tensors are of dim batch_size x samples x output_size x (len - kernel + 1)
        # pad tensors are of dim batch_size x samples x (len - kernel + 1)
        
        pid_title_pad_ex = torch.unsqueeze(pid_title_pad, 2).expand_as(pt)
        pid_body_pad_ex = torch.unsqueeze(pid_body_pad, 2).expand_as(pb)
        rest_title_pad_ex = torch.unsqueeze(rest_title_pad, 2).expand_as(rt)
        rest_body_pad_ex = torch.unsqueeze(rest_body_pad, 2).expand_as(rb)
        
        pt = torch.squeeze(torch.sum(pt * pid_title_pad_ex, dim = 3), dim = 3)
        pb = torch.squeeze(torch.sum(pb * pid_body_pad_ex, dim = 3), dim = 3)
        rt = torch.squeeze(torch.sum(rt * rest_title_pad_ex, dim = 3), dim = 3)
        rb = torch.squeeze(torch.sum(rb * rest_body_pad_ex, dim = 3), dim = 3)

        # tensors are not of dim batch_size x samples x output_size
        # need to scale down because not all uniformly padded

        ptp_norm = torch.sum(pid_title_pad, dim = 2).clamp(min = 1).expand_as(pt)
        pbp_norm = torch.sum(pid_body_pad, dim = 2).clamp(min = 1).expand_as(pb)
        rtp_norm = torch.sum(rest_title_pad, dim = 2).clamp(min = 1).expand_as(rt)
        rbp_norm = torch.sum(rest_body_pad, dim = 2).clamp(min = 1).expand_as(rb)
        
        pt = pt / ptp_norm
        pb = pb / pbp_norm
        rt = rt / rtp_norm
        rb = rb / rbp_norm
        
        pid_tensor = (pt + pb)/2
        rest_tensor = (rt + rb)/2
        
        if is_training:
            loss = loss_function(pid_tensor, rest_tensor)
            loss.backward()
            losses.append(loss.cpu().data[0])
            optimizer.step()
        else:
            expanded = pid_tensor.expand_as(rest_tensor)
            similarity = cs(expanded, rest_tensor, dim=2).squeeze(2)
            similarity = similarity.data.cpu().numpy()
            labels = batch['labels'].numpy()
            l = convert(similarity, labels)
            losses.extend(l)

    # Calculate epoch level scores
    if is_training:
        avg_loss = np.mean(losses)
        return avg_loss
    else:
        e = Evaluation(losses)
        MAP = e.MAP()*100
        MRR = e.MRR()*100
        P1 = e.Precision(1)*100
        P5 = e.Precision(5)*100
        return (MAP, MRR, P1, P5)

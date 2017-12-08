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
        
        pid_title, pid_body = pid_title.cuda(), pid_body.cuda()
        rest_title, rest_body = rest_title.cuda(), rest_body.cuda()
        
        if is_training:
            optimizer.zero_grad()
        
        pt = model(pid_title)
        pb = model(pid_body)
        rt = model(rest_title)
        rb = model(rest_body)
        
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

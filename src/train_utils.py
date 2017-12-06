import torch.utils.data

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
        
        val_loss = run_epoch(dev_data, False, model, optimizer)
        print('Val loss: {:.6f}'.format( val_loss))
        
        print('This epoch took: {:.6f}'.format(time.time() - lasttime))
        lasttime = time.time()

        
def run_epoch(data, is_training, model, optimizer):
    '''
    Train model for one pass of train data, and return loss, acccuracy
    '''
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=100,
        shuffle=True,
        num_workers=4,
        drop_last=True)

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
        
        if is_training:
            optimizer.zero_grad()
        
        pt = model(pid_title)
        pb = model(pid_body)
        rt = model(rest_title)
        rb = model(rest_body)
        
        pid_tensor = (pt + pb)/2
        rest_tensor = (rt + rb)/2
        
        loss = loss_function(pid_tensor, rest_tensor)
        
        if is_training:
            loss.backward()
            optimizer.step()

        losses.append(loss.cpu().data[0])

    # Calculate epoch level scores
    avg_loss = np.mean(losses)
    return avg_loss

import torch
import torch.nn as nn
import numpy as np
import random
import pickle

import MODEL
import matplotlib.pyplot as plt

def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr




def training(train_x,train_y,val_x,val_y,epochs,saved_best_loss=9999,pretrain_flag=False):
    ##### initial model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MODEL.NN(train_x.shape[1],64).to(device)
    # model = MODEL.ANN64(train_x.shape[1],64).to(device)
    if not pretrain_flag:
        model_param = torch.load("../0 save data/pretrain-model.pth")
        model.load_state_dict(model_param["net"])
    learning_rate = 0.001
    criterion = nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate)


    ##### model training
    loss,loss_val = [],[]
    for epoch in range(epochs):

        model.train()
        inputs = torch.tensor(train_x).float().to(device) 
        target = torch.tensor(train_y).float().to(device) 

        outputs = model(inputs)
        loss_value = criterion(outputs,target)

        # Backward and optimize
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        loss.append(loss_value.item())

        ##### for validation and test
        # for validation, get validation result and chose best model
        if not pretrain_flag:
            with torch.no_grad():
                model.eval()
                inputs = torch.tensor(val_x).float().to(device) 
                target = torch.tensor(val_y).float().to(device) 
            
                outputs = model(inputs)
                loss_val.append(criterion(outputs,target).item())
                predict_val = outputs.to("cpu").detach().numpy() 
        
    
    # save best result model
    if not pretrain_flag:
        if saved_best_loss>loss_val[-1]:
            is_better = True
            torch.save({'net':model.state_dict(),'optimizer':optimizer.state_dict()},"../0 save data/best-trained-model.pth")
        else:
            is_better = False
        return loss,loss_val,predict_val,is_better
    else:
        torch.save({'net':model.state_dict(),'optimizer':optimizer.state_dict()},"../0 save data/pretrain-model.pth")
        return loss, None, None, None





def training_main(px,py,x,y):

    # training, using cross-validation
    cv_num = 10
    batch_xs = np.array_split(x,cv_num)
    batch_ys = np.array_split(y,cv_num)

    # pre-training
    training_loss,training_val_loss,val_predict,better = training(px,
                                                                  py,
                                                                  None,
                                                                  None,
                                                                  epochs=300,
                                                                  saved_best_loss=None,
                                                                  pretrain_flag=True)

    # cross validate training
    min_loss,LOSS,predict = 9999, None, []
    for i,(batch_x,batch_y) in enumerate(zip(batch_xs,batch_ys)):

        # get train, val
        train_x,train_y = [],[]
        for j in range(cv_num):
            if i!=j:
                train_x.append(batch_xs[j])
                train_y.append(batch_ys[j])
            else:
                val_x,val_y, = batch_x,batch_y

        train_x = np.concatenate(train_x,axis=0)
        train_y = np.concatenate(train_y,axis=0)

        
        # training
        training_loss,training_val_loss,val_predict,better = training(train_x,
                                                                      train_y,
                                                                      val_x,
                                                                      val_y,
                                                                      epochs=200,
                                                                      saved_best_loss=min_loss,
                                                                      pretrain_flag=False)

        # save best model result
        if better:
            min_loss = training_val_loss[-1]
        
        if LOSS is None:
            LOSS = np.array(training_loss)
            LOSS_val = np.array(training_val_loss)
        else:
            LOSS +=training_loss
            LOSS_val +=training_val_loss

        predict.append(val_predict)
        print(".",end="")


    return LOSS/cv_num, \
           LOSS_val/cv_num, \
           np.concatenate(predict,axis=0)




#### only calculate error
def cal_error(output, target):
    output = np.squeeze(output)
    target = np.squeeze(target)
    return {"R": np.corrcoef([output,target])[1,0],
            "R2":1-((output-target)**2).sum()/((target-target.mean())**2).sum(),
            "MAE": np.abs(output-target).mean(),
            "RMSE":np.sqrt(((output-target)**2).mean()),
            "Bias": (target-output).mean()} 

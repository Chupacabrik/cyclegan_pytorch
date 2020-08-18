import pandas as pd
from models import *
from utils import *


def train(num_epochs,decay_epoch, lrD, lrG, lambdaA, lambdaB):
    step = 0
    for epoch in range(num_epochs):
        D_A_losses = []
        D_B_losses = []
        G_A_losses = []
        G_B_losses = []
        cycle_A_losses = []
        cycle_B_losses = []

        # Learing rate decay 
        if(epoch + 1) > decay_epoch:
            D_A_optimizer.param_groups[0]['lr'] -= lrD / (num_epochs - decay_epoch)
            D_B_optimizer.param_groups[0]['lr'] -= lrD / (num_epochs - decay_epoch)
            G_optimizer.param_groups[0]['lr'] -= lrG / (num_epochs - decay_epoch)
    # training 
        for i, (real_A, real_B) in enumerate(zip(train_data_loader_A, train_data_loader_B)):

            # input image data
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            # -------------------------- train generator G --------------------------
            # A --> B
            fake_B = G_A(real_A)
            D_B_fake_decision = D_B(fake_B)
            G_A_loss = MSE_Loss(D_B_fake_decision, torch.ones(D_B_fake_decision.size()).cuda())

            # forward cycle loss
            recon_A = G_B(fake_B)
            cycle_A_loss = L1_Loss(recon_A, real_A) * lambdaA

            # B --> A
            fake_A = G_B(real_B)
            D_A_fake_decision = D_A(fake_A)
            G_B_loss = MSE_Loss(D_A_fake_decision, torch.ones(D_A_fake_decision.size()).cuda())

            # backward cycle loss
            recon_B = G_A(fake_A)
            cycle_B_loss = L1_Loss(recon_B, real_B) * lambdaB

            # Back propagation
            G_loss = G_A_loss + G_B_loss + cycle_A_loss + cycle_B_loss
            G_optimizer.zero_grad()
            G_loss.backward()
            G_optimizer.step()


            # -------------------------- train discriminator D_A --------------------------
            D_A_real_decision = D_A(real_A)
            D_A_real_loss = MSE_Loss(D_A_real_decision, torch.ones(D_A_real_decision.size()).cuda())

            fake_A = fake_A_pool.query(fake_A)

            D_A_fake_decision = D_A(fake_A)
            D_A_fake_loss = MSE_Loss(D_A_fake_decision, torch.zeros(D_A_fake_decision.size()).cuda())

            # Back propagation
            D_A_loss = (D_A_real_loss + D_A_fake_loss) * 0.5
            D_A_optimizer.zero_grad()
            D_A_loss.backward()
            D_A_optimizer.step()

            # -------------------------- train discriminator D_B --------------------------
            D_B_real_decision = D_B(real_B)
            D_B_real_loss = MSE_Loss(D_B_real_decision, torch.ones(D_B_fake_decision.size()).cuda())

            fake_B = fake_B_pool.query(fake_B)

            D_B_fake_decision = D_B(fake_B)
            D_B_fake_loss = MSE_Loss(D_B_fake_decision, torch.zeros(D_B_fake_decision.size()).cuda())

            # Back propagation
            D_B_loss = (D_B_real_loss + D_B_fake_loss) * 0.5
            D_B_optimizer.zero_grad()
            D_B_loss.backward()
            D_B_optimizer.step()

            # ------------------------ Print -----------------------------
            # loss values
            D_A_losses.append(D_A_loss.item())
            D_B_losses.append(D_B_loss.item())
            G_A_losses.append(G_A_loss.item())
            G_B_losses.append(G_B_loss.item())
            cycle_A_losses.append(cycle_A_loss.item())
            cycle_B_losses.append(cycle_B_loss.item())

            if i%100 == 0:
                print('Epoch [%d/%d], Step [%d/%d], D_A_loss: %.4f, D_B_loss: %.4f, G_A_loss: %.4f, G_B_loss: %.4f'
                      % (epoch+1, num_epochs, i+1, len(train_data_loader_A), D_A_loss.item(), D_B_loss.item(), G_A_loss.item(), G_B_loss.item()))

            step += 1

        D_A_avg_loss = torch.mean(torch.FloatTensor(D_A_losses))
        D_B_avg_loss = torch.mean(torch.FloatTensor(D_B_losses))
        G_A_avg_loss = torch.mean(torch.FloatTensor(G_A_losses))
        G_B_avg_loss = torch.mean(torch.FloatTensor(G_B_losses))
        cycle_A_avg_loss = torch.mean(torch.FloatTensor(cycle_A_losses))
        cycle_B_avg_loss = torch.mean(torch.FloatTensor(cycle_B_losses))

        # avg loss values for plot
        D_A_avg_losses.append(D_A_avg_loss.item())
        D_B_avg_losses.append(D_B_avg_loss.item())
        G_A_avg_losses.append(G_A_avg_loss.item())
        G_B_avg_losses.append(G_B_avg_loss.item())
        cycle_A_avg_losses.append(cycle_A_avg_loss.item())
        cycle_B_avg_losses.append(cycle_B_avg_loss.item())

        # Show result for test image
        test_real_A = test_real_A_data.cuda()
        test_fake_B = G_A(test_real_A)
        test_recon_A = G_B(test_fake_B)

        test_real_B = test_real_B_data.cuda()
        test_fake_A = G_B(test_real_B)
        test_recon_B = G_A(test_fake_A)

        plot_train_result([test_real_A, test_real_B], [test_fake_B, test_fake_A], [test_recon_A, test_recon_B],
                                epoch, save=True)
        
        # Save model checkpoints
        torch.save(G_A.state_dict(), "output/saved_models/G_AB_%d.pth" % (epoch + 1))
        torch.save(G_B.state_dict(), "output/saved_models/G_BA_%d.pth" % (epoch + 1))
        torch.save(D_A.state_dict(), "output/saved_models/D_A_%d.pth" % (epoch + 1))
        torch.save(D_B.state_dict(), "output/saved_models/D_B_%d.pth" % (epoch + 1))

    all_losses = pd.DataFrame()
    all_losses['D_A_avg_losses'] = D_A_avg_losses
    all_losses['D_B_avg_losses'] = D_B_avg_losses
    all_losses['G_A_avg_losses'] = G_A_avg_losses
    all_losses['G_B_avg_losses'] = G_B_avg_losses
    all_losses['cycle_A_avg_losses'] = cycle_A_avg_losses
    all_losses['cycle_B_avg_losses'] = cycle_B_avg_losses
    all_losses.to_csv('avg_losses',index=False)
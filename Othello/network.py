import torch
import torch.nn as nn
import torch.nn.functional as F

import othello_env

"""Body Of The Network"""

#Initial Block
class Initial_Block(nn.Module):
    def __init__(self, C = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=C, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(C)

    def forward(self, input):
        out = F.relu(self.bn1(self.conv1(input)))
        return out

#Residual Block
class Residual_Block(nn.Module):
    def __init__(self, C = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(C, C, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(C)
        self.conv2 = nn.Conv2d(C, C, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(C)

    def forward(self, x):
        save = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + save
        return F.relu(out)
    
class Othello_Net(nn.Module):
    def __init__(self, n, C = 64, num_res_blocks = 4):
        super().__init__()
        self.n = n
        self.action_size = n*n + 1
        self.initial_block = Initial_Block()
        self.res_blocks = nn.ModuleList([Residual_Block(C) for _ in range(num_res_blocks)])
        #policy head
        self.policy_conv = nn.Conv2d(C, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(in_features=2*n*n, out_features=self.action_size)
        #value head
        self.value_conv = nn.Conv2d(C, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(in_features=n*n, out_features=64)
        self.value_fc2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        x = self.initial_block(x)
        for block in self.res_blocks:
            x = block(x)
        #policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        #value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v

def compute_loss(policy_logits, value, pi, z):
    value_loss = (z - value).pow(2).mean()
    policy_loss = -(pi * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
    return value_loss + policy_loss

#testing shapes and architecture
"""
net = Othello_Net(n = 6)
dummy = torch.randn((32, 2, 6, 6)) #(batch_size, planes, board_size, board_size)
policy_logits, value = net(dummy)
print(f'policy_logits shape: {policy_logits.shape}') #expecting [32, 37]
print(f'value shape: {value.shape}') #expecting [32, 1]
"""

if __name__ == "__main__":
    #testing if this overfits 1 state of othello - if yes means are arch is fine

    env = othello_env.OthelloEnv(6)
    net = Othello_Net(6)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    net.train()

    state = env.reset()
    state_tensor = torch.FloatTensor(state) #(2,6,6)
    state_tensor = state_tensor.unsqueeze(0) #now (1,2,6,6)

    n = 6
    action_size = n*n + 1

    pi = torch.zeros(1, action_size) #target policy
    legal = env.legal_actions()
    pi[0, legal[0]] = 1.0 # / len(legal) # set probs for only legal actions (divided by legal.length to make sure the probs add up to 1)
    """
    however rn we removed divide by len(legal) cause we wanted to make sure 
    the loss goes to 0 and overfits... but due to /len(legal), if say only 4 legal moves
    then the cross entropy loss will go to minimum of ln(4) which is abt 1.38... so it wont ever go to 0
    and we saw this when the first run gave this:
    step 0 loss 4.0909
    step 50 loss 1.4986
    step 100 loss 1.4177
    step 150 loss 1.4020
    step 200 loss 1.3959
    step 250 loss 1.3929

    the next run however gave this:
    step 0 loss 4.7532
    step 50 loss 0.1620
    step 100 loss 0.0470
    step 150 loss 0.0241
    step 200 loss 0.0151
    step 250 loss 0.0105
    """

    z = torch.tensor([[1.0]]) #fake value set

    state_tensor = state_tensor.repeat(8,1,1,1)
    pi = pi.repeat(8,1)
    z = z.repeat(8,1)

    for step in range(300):
        policy_logits, value = net(state_tensor)

        value_loss = (z - value).pow(2).mean()
        policy_loss = -(pi * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
        loss = value_loss + policy_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"step {step} loss {loss.item():.4f}")    

import torch
import numpy as np
import network
import self_play
import mcts

def train_network(net, examples, optimizer, batch_size, epochs):
    device = next(net.parameters()).device
    net.train()
    for epoch in range(epochs):
        loss_sum = 0.0
        num_batches = 0
        np.random.shuffle(examples)
        for i in range(0, len(examples), batch_size):
            batch = examples[i : i + batch_size]
            states = torch.FloatTensor(np.array([ex[0] for ex in batch])).to(device)
            pis = torch.FloatTensor(np.array([ex[1] for ex in batch])).to(device)
            zs = torch.FloatTensor(np.array([ex[2] for ex in batch])).unsqueeze(1).to(device)

            policy_logits, value = net(states)
            loss = network.compute_loss(policy_logits, value, pis, zs)
            loss_sum += loss.item()
            num_batches += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return loss_sum/num_batches

def train(num_iterations, games_per_iteration, epochs, n=6, sims=25, batch_size=32):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    net = network.Othello_Net(n).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    print(f"Training on {device}")

    for iteration in range(num_iterations):
        examples = self_play.generate_selfplay_data(net, games_per_iteration, n, sims)
        avg_loss = train_network(net, examples, optimizer, batch_size, epochs)
        win_rate = mcts.evaluate_vs_random(net, sims, games=20, n=n)
        print(f"Iteration: {iteration+1} Avg Loss: {avg_loss:.4f} Win Rate vs Random: {win_rate:.4f}")
        torch.save({"iteration": iteration, "net": net.state_dict(), "optimizer": optimizer.state_dict(),}, f"checkpoint_iter{iteration}.pt")

if __name__ == "__main__":
    train(num_iterations=10, games_per_iteration=20, sims=25, epochs=4, n=6, batch_size=32)
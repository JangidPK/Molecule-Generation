import torch
from molecule_generator.reinforcement_learning.reward import compute_reward

def reinforce_step(model, optimizer, tokenizer):

    model.train()

    start_token = torch.tensor(
    [[tokenizer.token_to_idx["<START>"]]],
    device=next(model.parameters()).device
        )

    generated = model.generate(
        start_token,
        max_new_tokens=100
    )

    tokens = tokenizer.decode(
             generated.squeeze().tolist())

    smiles = tokenizer.detokenize(tokens)

    reward = compute_reward(smiles)

    log_probs = model.compute_log_probs(generated)

    loss = -reward * log_probs.mean()

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    return loss.item(), reward

def rl_optimize(model, optimizer, tokenizer, steps):

    for step in range(steps):

        loss, reward = reinforce_step(
            model,
            optimizer,
            tokenizer
        )

        print(
            f"Step {step} | "
            f"Loss {loss:.4f} | "
            f"Reward {reward:.4f}"
        )
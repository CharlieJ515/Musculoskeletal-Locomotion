import torch
import torch.nn.functional as F

from rl.replay_buffer import ReplayBuffer


def pretrain_actor(
    rb: ReplayBuffer,
    actor,
    name: str = "actor",
    num_epoch: int = 10,
    batch_size: int = 256,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    lr: float = 0.001,
):
    actor.to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
    num_step = max(1, len(rb) // batch_size)

    print("Starting actor pretraining")
    for epoch in range(1, num_epoch + 1):
        loss_sum = 0.0
        for step in range(1, num_step + 1):
            batch = rb.sample(256, pin_memory=True).to(device)
            s, a, _, _, _ = batch.unpack()

            action, _ = actor(s, deterministic=True)
            loss = F.mse_loss(action, a)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.detach().cpu().item()

            t = num_step * epoch + step
            # writer.add_histogram(f"pretrain/{name}/output", action.detach().cpu(), t)
            # writer.add_histogram(f"pretrain/{name}/replay_buffer", a, t)

            # log_weight_hist(f"pretrain/actor/weights", actor, t)
            # log_grad_hist(f"pretrain/actor/grads", actor, t)

        epoch_loss = loss_sum / num_step
        # print(f"epoch {epoch:3}, loss: {epoch_loss:.4f}")

    # s, a, _, _, _ = rb.all(pin_memory=True).to(device).unpack()
    # action, _ = actor(s, deterministic=True)

    # writer.add_histogram(f"pretrain/{name}/total/replay_buffer", a)
    # writer.add_histogram(f"pretrain/{name}/total/output", action.detach().cpu())

    print("Actor pretraining complete")


def pretrain_critic(
    rb: ReplayBuffer,
    critic,
    name: str = "critic",
    num_epoch: int = 10,
    batch_size: int = 256,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    lr: float = 1e-3,
):
    critic.to(device)
    optimizer = torch.optim.Adam(critic.parameters(), lr=lr)
    num_step = max(1, len(rb) // batch_size)

    print("Starting critic pretraining")
    for epoch in range(1, num_epoch + 1):
        loss_sum = 0.0
        for step in range(1, num_step + 1):
            batch = rb.sample(batch_size, pin_memory=True).to(device)
            s, a, r, _, _ = batch.unpack()

            q_pred = critic(s, a)
            loss = F.mse_loss(q_pred, r)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.detach().cpu().item()

            t = num_step * epoch + step
            # writer.add_histogram(f"pretrain/{name}/output", q_pred.detach().cpu(), t)
            # writer.add_histogram(f"pretrain/{name}/replay_buffer", r, t)

            # log_weight_hist(f"pretrain/{name}/weights", critic, t)
            # log_grad_hist(f"pretrain/{name}/grads", critic, t)

        epoch_loss = loss_sum / num_step
        # print(f"epoch {epoch:3}, loss: {epoch_loss:.4f}")

    # s, a, r, _, _ = rb.all(pin_memory=True).to(device).unpack()
    # q_pred = critic(s, a).detach().cpu()

    # plt.figure(figsize=(8, 6))
    # colors = ["r", "g", "b", "orange"]  # one color per distribution
    # labels = [f"Q{i}" for i in range(4)]

    # for i in range(4):
    #     plt.hist(q_pred[:, i], bins=50, alpha=0.5, color=colors[i], label=labels[i])

    # plt.title("Q-value Distributions")
    # plt.xlabel("Q value")
    # plt.ylabel("Frequency")
    # plt.legend()
    # plt.grid(True, linestyle="--", alpha=0.7)
    # plt.show()

    # writer.add_histogram(f"pretrain/{name}/replay_buffer", a)
    # writer.add_histogram(f"pretrain/{name}/output", q_pred)

    print("Critic pretraining complete")

import wandb

wandb.init(
    project="Hemorrhage_SOTA_V2",
    name="wandb_test"
)

for epoch in range(5):
    wandb.log({
        "loss": 1.0 / (epoch + 1),
        "dice": epoch * 0.2
    })

wandb.finish()

print("W&B Test Successful!")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_v3 import FluidDatasetV3
from model_v3 import HybridUNet  

class PINNLoss(nn.Module):
    def __init__(self, lambda_physics=0.01):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_physics = lambda_physics

        # Fixed Sobel kernels to calculate spatial derivatives on the GPU/CPU grid
        # dx detects horizontal changes, dy detects vertical changes
        kernel_dx = torch.tensor([[[[-0.5, 0.0, 0.5]]]], dtype=torch.float32)
        kernel_dy = torch.tensor([[[[-0.5], [0.0], [0.5]]]], dtype=torch.float32)
        
        self.register_buffer('k_dx', kernel_dx)
        self.register_buffer('k_dy', kernel_dy)

    def compute_divergence(self, velocity_field):
        # velocity_field shape, where channel 0 is U, channel 1 is V
        u = velocity_field[:, 0:1, :, :]
        v = velocity_field[:, 1:2, :, :]

        # Compute spatial derivatives using standard 2D convolutions with fixed weights
        du_dx = nn.functional.conv2d(u, self.k_dx, padding=(0, 1))
        dv_dy = nn.functional.conv2d(v, self.k_dy, padding=(1, 0))

        # Divergence = du/dx + dv/dy
        divergence = du_dx + dv_dy
        return divergence

    def forward(self, pred, target):
        #  Standard Data Loss (literally the MSE)
        loss_mse = self.mse(pred, target)

        # Physical Law Loss (Conservation of Mass)
        div = self.compute_divergence(pred)
        loss_physics = torch.mean(div ** 2)

        # Total combined loss for this resolution scale
        return loss_mse + (self.lambda_physics * loss_physics)

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training execution target hardware: {device}")

    # Init dataset and loader
    dataset = FluidDatasetV3()
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    # Architecture yeahh
    model = HybridUNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    #Drop learning rate by 50% if the loss gets stuck for 4 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    
    # Deep Supervision requires grading at every scale
    criterion = PINNLoss(lambda_physics=0.01)

    # Core Training Loop
    epochs = 50
    print("Beginning execution of the training loop...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (sdf, t16, t32, t64) in enumerate(train_loader):
            sdf = sdf.to(device)
            t16, t32, t64 = t16.to(device), t32.to(device), t64.to(device)

            # Construct the input tensor
            zeros = torch.zeros_like(t64)
            x_in = torch.cat([sdf, zeros], dim=1) # Shape: [Batch, 3, 64, 64]

            # Forward pass returns all three scaled predictions
            p16, p32, p64 = model(x_in)

            # Calculate Deep Supervision Loss across all resolutions
            loss_16 = criterion(p16, t16)
            loss_32 = criterion(p32, t32)
            loss_64 = criterion(p64, t64)
            
            # Total Loss balances macro flow learning with micro edge sharpness
            batch_loss = loss_64 + (0.5 * loss_32) + (0.25 * loss_16)

            # Backpropagation
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1:02d}/{epochs}] -> Average Scaled Loss: {avg_loss:.6f}")
        
        # Tell the scheduler to check if we are stuck
        scheduler.step(avg_loss)

    # Save weights after verification
    torch.save(model.state_dict(), "hybrid_unet_v4.pth")
    print("Training complete. State dict successfully saved.")

if __name__ == "__main__":
    train_model()
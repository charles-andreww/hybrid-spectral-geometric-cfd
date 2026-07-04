import os
import random
import math
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

# The exact physics engine it was trained on
from phi.torch.flow import *
from phi.geom import union
from model_v3 import HybridUNet 


def generate_jagged_obstacle():
    """Identical to the training data generator."""
    primitives = []
    
    center_x = random.uniform(30, 45)
    center_y = random.uniform(25, 39)
    core_radius = random.uniform(6, 12)
    primitives.append(Sphere(center=vec(x=center_x, y=center_y), radius=core_radius))
    
    num_spikes = random.randint(5, 12)
    for _ in range(num_spikes):
        angle = random.uniform(0, 2 * math.pi)
        dist = random.uniform(core_radius * 0.3, core_radius * 1.5)
        
        cx = center_x + math.cos(angle) * dist
        cy = center_y + math.sin(angle) * dist
        
        if random.choice([True, False]):
            primitives.append(Sphere(center=vec(x=cx, y=cy), radius=random.uniform(2, 5)))
        else:
            w, h = random.uniform(4, 12), random.uniform(4, 12)
            primitives.append(Box(x=(cx-w/2, cx+w/2), y=(cy-h/2, cy+h/2)))
            
    return union(*primitives)

def live_evaluation():
    print("\nLIVE EVALUATION")
    
    
    device = torch.device("cpu") 
    model = HybridUNet().to(device)
    
    try:
        model.load_state_dict(torch.load("hybrid_unet_v4.pth", map_location=device))
        model.eval()
        print("V4 Hybrid Architecture Online.")
    except Exception as e:
        print(f"Critical Error loading weights: {e}")
        return

    # Physics, real physics
    print("Generating shape n running Phi-Flow...")
    start_trad = time.perf_counter()
    
    # Setup Wind Tunnel identical to training
    grid_size = [64, 64]
    velocity_grid = StaggeredGrid(vec(x=5.0, y=0.0), extrapolation.BOUNDARY, x=grid_size[0], y=grid_size[1], bounds=Box(x=100, y=100))
    pressure_grid = CenteredGrid(0, extrapolation.BOUNDARY, x=grid_size[0], y=grid_size[1], bounds=Box(x=100, y=100))
    
    obstacle_geometry = generate_jagged_obstacle()
    obstacle = Obstacle(obstacle_geometry, velocity=[0, 0], angular_velocity=0)
    
    # 30-Step Warm-up Loop
    for step in range(30):
        velocity_grid = advect.semi_lagrangian(velocity_grid, velocity_grid, dt=0.5)
        velocity_grid, pressure_grid = fluid.make_incompressible(velocity_grid, [obstacle])
        
    # Extract Ground Truth Data
    sdf_tensor = obstacle_geometry.approximate_signed_distance(pressure_grid.points)
    sdf_matrix = sdf_tensor.numpy('y,x')
    
    centered_velocity = velocity_grid.at_centers()
    vel_array = centered_velocity.values.numpy('y,x,vector')
    true_u = vel_array[..., 0]
    true_v = vel_array[..., 1]
    
    end_trad = time.perf_counter()
    trad_time_ms = (end_trad - start_trad) * 1000

    # AI
    print("V4 AI...")
    
    # Format the data for PyTorch [Batch, Channels, H, W]
    sdf_torch = torch.tensor(sdf_matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    zeros = torch.zeros((1, 2, 64, 64), dtype=torch.float32).to(device)
    x_in = torch.cat([sdf_torch, zeros], dim=1)
    
    # Warmup the CPU cache
    with torch.no_grad():
        _ = model(x_in)
        
    start_ai = time.perf_counter()
    with torch.no_grad():
        _, _, p64 = model(x_in)
    end_ai = time.perf_counter()
    
    ai_time_ms = (end_ai - start_ai) * 1000
    speedup = trad_time_ms / ai_time_ms
    
# Calculate Error n Accuracy 
    pred_u = p64[0, 0].cpu().numpy()
    pred_v = p64[0, 1].cpu().numpy()
    
    # Stack the U and V vectors back into full wind arrays for the math
    true_wind = np.stack([true_u, true_v], axis=0)
    pred_wind = np.stack([pred_u, pred_v], axis=0)
    
    # The Machine Learning Metric (MSE)
    mse_u = np.mean((true_u - pred_u) ** 2)
    mse_v = np.mean((true_v - pred_v) ** 2)
    total_mse = (mse_u + mse_v) / 2.0

    # Normalized Accuracy %
    # We compare the average absolute mistake to the absolute peak wind speed
    max_wind_speed = np.max(np.abs(true_wind)) + 1e-5 
    mean_absolute_error = np.mean(np.abs(true_wind - pred_wind))
    
    # Convert to a clean percentage 
    accuracy_percentage = max(0.0, (1.0 - (mean_absolute_error / max_wind_speed))) * 100.0

    # Metrics
    print("\nMETRICS")
    print(f"Phi-Flow Engine Time: {trad_time_ms / 1000:.4f} seconds")
    print(f"Hybrid AI Inference:  {ai_time_ms:.4f} milliseconds")
    print(f"Acceleration Factor:  {speedup:.1f}x Faster")
    print(f"Mean Squared Error:   {total_mse:.6f}")
    print(f"Physical Accuracy:    {accuracy_percentage:.2f}%")

    # Graph
    gt_mag = np.sqrt(true_u**2 + true_v**2)
    ai_mag = np.sqrt(pred_u**2 + pred_v**2)
    error_map = np.abs(gt_mag - ai_mag)
    
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 4, 1)
    plt.title("Generated Shape")
    plt.imshow(sdf_matrix, cmap='inferno')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.title(f"Ground Truth({trad_time_ms / 1000:.2f}s)")
    plt.imshow(gt_mag, cmap='viridis')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.title(f"AI ({ai_time_ms:.1f}ms)")
    plt.imshow(ai_mag, cmap='viridis')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.title(f"Absolute Error (MSE: {total_mse:.4f})")
    plt.imshow(error_map, cmap='magma')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    live_evaluation()
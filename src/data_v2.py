import os
import random
import math
import numpy as np
from phi.torch.flow import *
from phi.geom import union
import gc
from tqdm import tqdm


DATASET_DIR = "fluid_dataset_2"
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

def generate_jagged_obstacle():
    """
    Creates a highly irregular, organic, and jagged shape by clustering 
    multiple sharp boxes and smooth spheres together.
    """
    primitives = []
    
    # Establish the core of the asteroid/blob/thingy/meow
    center_x = random.uniform(30, 45)
    center_y = random.uniform(25, 39)
    core_radius = random.uniform(6, 12)
    primitives.append(Sphere(center=vec(x=center_x, y=center_y), radius=core_radius))
    
    # Sprout 5 to 12 jagged "spikes" and "bumps" around the core
    num_spikes = random.randint(5, 12)
    for _ in range(num_spikes):
        # Pick a random angle and distance from the core
        angle = random.uniform(0, 2 * math.pi)
        dist = random.uniform(core_radius * 0.3, core_radius * 1.5)
        
        cx = center_x + math.cos(angle) * dist
        cy = center_y + math.sin(angle) * dist
        
        # 50/50 chance to grow a smooth bump or a sharp jagged edge
        if random.choice([True, False]):
            primitives.append(Sphere(center=vec(x=cx, y=cy), radius=random.uniform(2, 5)))
        else:
            w, h = random.uniform(4, 12), random.uniform(4, 12)
            primitives.append(Box(x=(cx-w/2, cx+w/2), y=(cy-h/2, cy+h/2)))
            
    # Fuse them all together into one chaotic silhouette
    return union(*primitives)

def generate_and_save_data(target_samples=3000, start_idx=0):
    print(f"Igniting V2 Physics Engine: Generating pure geometric chaos.")
    print(f"Targeting {target_samples} samples in '{DATASET_DIR}'...")
    
    successful_saves = start_idx
    
    # Initialize the beautiful progress bar yayasyayayay
    pbar = tqdm(total=target_samples, initial=start_idx, desc="Generating Fluid Data", unit="sim")
    
    while successful_saves < target_samples:
        try:
            # Setup the Wind Tunnel
            grid_size = [64, 64]
            velocity_grid = StaggeredGrid(vec(x=5.0, y=0.0), extrapolation.BOUNDARY, x=grid_size[0], y=grid_size[1], bounds=Box(x=100, y=100))
            pressure_grid = CenteredGrid(0, extrapolation.BOUNDARY, x=grid_size[0], y=grid_size[1], bounds=Box(x=100, y=100))
            
            # Add the custom jagged obstacle
            obstacle_geometry = generate_jagged_obstacle()
            obstacle = Obstacle(obstacle_geometry, velocity=[0, 0], angular_velocity=0)
            
            # Warm-up Loop
            for step in range(30):
                velocity_grid = advect.semi_lagrangian(velocity_grid, velocity_grid, dt=0.5)
                velocity_grid, pressure_grid = fluid.make_incompressible(velocity_grid, [obstacle])
                
            #Extract Data Matrices
            sdf_tensor = obstacle_geometry.approximate_signed_distance(pressure_grid.points)
            sdf_matrix = sdf_tensor.numpy('y,x')
            
            centered_velocity = velocity_grid.at_centers()
            vel_array = centered_velocity.values.numpy('y,x,vector')
            u_matrix = vel_array[..., 0]
            v_matrix = vel_array[..., 1]
            
            # Save
            filename = os.path.join(DATASET_DIR, f"sim_v2_{successful_saves:04d}.npz")
            np.savez_compressed(filename, sdf=sdf_matrix, u=u_matrix, v=v_matrix)
            
            # SUCCESS! Increment the counter and update the progress bar
            successful_saves += 1
            pbar.update(1)
                
        except Exception as e:
            # Physics broke! Silently discard and retry TOOK WAYYY TOO LONG TO DEBUG
            pass
            
        finally:
            del velocity_grid, pressure_grid, obstacle, obstacle_geometry
            import gc
            gc.collect()
            
    # Close the progress bar when finished
    pbar.close()

if __name__ == "__main__":
    # Set to 3000 to get a massive, highly generalized dataset
    generate_and_save_data(target_samples=3000, start_idx=0)
    print(f"\n[!] Mission accomplished. All V2 files compiled in '{DATASET_DIR}'.")
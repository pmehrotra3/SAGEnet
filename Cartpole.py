import gymnasium as gym
import time

# ==============================================================================
# Important Note on Dependencies:
#
# Some of these environments require extra packages to be installed.
# Before running, you might need to install them via pip:
#
# pip install gymnasium[box2d] pygame  # For LunarLander, BipedalWalker, CarRacing
# pip install gymnasium[mujoco]         # For HalfCheetah, Hopper
# ==============================================================================

def run_environment(env_name, description):
    """
    A helper function to create, run, and display an environment.
    The agent will only take random actions.
    """
    print(f"\n########## Running: {env_name} ##########")
    print(description)
    
    try:
        # Create the environment with a render_mode to see the simulation
        env = gym.make(env_name, render_mode="human")

        # Reset the environment to get the initial state
        state, info = env.reset(seed=42) # Using a seed for reproducibility

        total_reward = 0
        max_steps = 1000 # Limit the number of steps per episode

        for t in range(max_steps):
            # Render the environment on the screen
            env.render()

            # The agent takes a random action from the action space
            action = env.action_space.sample()

            # The environment processes the action and returns the next state, reward, etc.
            next_state, reward, terminated, truncated, info = env.step(action)

            # Accumulate the reward
            total_reward += reward

            # Move to the next state
            state = next_state

            # If the episode is over (terminated) or timed out (truncated), we stop
            if terminated or truncated:
                print(f"Episode finished after {t+1} timesteps.")
                break
        
        # Close the environment window
        env.close()
        print(f"Total reward for {env_name}: {total_reward:.2f}")
        print("##########################################\n")

    except Exception as e:
        print(f"Could not run {env_name}. Error: {e}")
        print("Please make sure you have installed the necessary dependencies.")
        print("For example: pip install gymnasium[box2d] or pip install gymnasium[mujoco]")
        print("##########################################\n")
    
    # A small delay to make it easier to see the transition between environments
    time.sleep(2)


if __name__ == "__main__":
    # --- 1. CartPole-v1 ---
    # Goal: Balance a pole on a moving cart for as long as possible.
    # Action Space: Discrete (0: push left, 1: push right)
    run_environment(
        "CartPole-v1",
        "Goal: Balance the pole on the cart."
    )

    # --- 2. MountainCar-v0 ---
    # Goal: Drive an underpowered car up a steep hill to reach the flag.
    # Action Space: Discrete (0: push left, 1: no push, 2: push right)
    run_environment(
        "MountainCar-v0",
        "Goal: Drive the car to the flag on the right hill."
    )

    # --- 3. Acrobot-v1 ---
    # Goal: Swing a two-link robotic arm up to a target height.
    # Action Space: Discrete (0: no torque, 1: positive torque, 2: negative torque on the joint)
    run_environment(
        "Acrobot-v1",
        "Goal: Swing the lower link up above the target line."
    )

    # --- 4. Pendulum-v1 ---
    # Goal: Swing a pendulum up so it stays upright. This is a classic continuous control task.
    # Action Space: Continuous (A value representing the amount of torque to apply)
    run_environment(
        "Pendulum-v1",
        "Goal: Swing the pendulum up and keep it balanced."
    )

    # --- 5. LunarLander-v2 ---
    # Goal: Land a spacecraft safely between two flags on the moon's surface.
    # Action Space: Discrete (0: do nothing, 1: fire left engine, 2: fire main engine, 3: fire right engine)
    run_environment(
        "LunarLander-v2",
        "Goal: Land the spacecraft gently between the yellow flags."
    )

    # --- 6. BipedalWalker-v3 ---
    # Goal: Teach a two-legged robot to walk across challenging, randomly generated terrain.
    # Action Space: Continuous (Four values controlling the torque of the hip and knee joints)
    run_environment(
        "BipedalWalker-v3",
        "Goal: Walk across the terrain without falling."
    )

    # --- 7. HalfCheetah-v4 ---
    # Goal: Make a 2D "cheetah" robot run forward as fast as possible. A standard benchmark for continuous control.
    # Action Space: Continuous (Six values controlling the torque of the robot's joints)
    run_environment(
        "HalfCheetah-v4",
        "Goal: Run forward as fast as possible."
    )

    # --- 8. Blackjack-v1 ---
    # Goal: Get a card total closer to 21 than the dealer, without busting (going over 21).
    # Action Space: Discrete (0: stick/stand, 1: hit)
    run_environment(
        "Blackjack-v1",
        "Goal: Get as close to 21 as possible without going over."
    )

    # --- 9. CarRacing-v2 ---
    # Goal: Drive a car around a randomly generated track. State is pixels from a top-down view.
    # Action Space: Continuous (Steering: -1 to 1, Gas: 0 to 1, Brake: 0 to 1)
    run_environment(
        "CarRacing-v2",
        "Goal: Drive the car around the track quickly."
    )

    # --- 10. Hopper-v4 ---
    # Goal: Make a one-legged robot hop forward as fast as possible.
    # Action Space: Continuous (Three values controlling the torques of the robot's joints)
    run_environment(
        "Hopper-v4",
        "Goal: Hop forward as fast as possible."
    )



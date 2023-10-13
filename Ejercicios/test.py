import gym_examples

import gymnasium


env = gymnasium.make('gym_examples/GridWorld-v0')
recorder = GymRecorder(env)

for episode in range(10):
    state, info = env.reset()

    done = False
    
    while not done:
        action = env.action_space.sample()

        next_state, reward, terminal, truncated, info = env.step(action)

        state = next_state
        done = terminal or truncated
    
    recorder.record(env.render())

        
recorder.make_video('videos/CartPole-v1.gif')
ipython_display('videos/CartPole-v1.gif', autoplay=1, loop=1)

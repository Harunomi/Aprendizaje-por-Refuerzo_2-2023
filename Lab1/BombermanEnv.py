import gym
from gym import spaces
import numpy as np
import pygame


class Item(object):
    def __init__(self,pos):
        self.pos = pos

class Box(Item):
    def __init__(self,pos,breakable,broken,target):
        super(Box,self).__init__(pos)
        self.isBreakable = breakable
        self.isTarget = target
        self.isBroken = broken

    def get_state(self):
        return (self.pos, self.isBreakable, self.isTarget,self.isBroken)
    
class Enemy(Item):
    def __init__(self,pos,orientation,way,isAlive):
        super(Enemy,self).__init__(pos)
        self.orientation = orientation
        self.way = way
        self.isAlive = isAlive


    def get_state(self):
        return (self.pos, self.orientation, self.way, self.isAlive)
    


class Bomb(Item):
    def __init__(self,pos,timer):
        super(Bomb,self).__init__(pos)
        self.timer = timer


def mult(x,y):
        if(x > y):
            multi = x/y
        else:
            multi = y/x
        return multi

def exist(list,pos):
    for i in range(len(list)):
        if(pos[0] == list[i].pos[0] and pos[1] == list[i].pos[1]):
            return True
    return False


class BombermanEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 2}

    def __init__(self, width, height, boxes, enemies_x, enemies_y, rompible_file, render_mode = None):
        super(BombermanEnv, self).__init__()

        self.width = width
        self.height = height
        if boxes < 1:
            print("El numero de cajas debe ser mayor que 1.")
            return 
        self.active_explosion = 0
        self.active_bomb = 0
        self.list_boxes = [] # lista para las cajas rompibles e irrompibles
        self.boxes = boxes # total de cajas rompibles
        self.list_boxes_breakable = [] # lista para las cajas rompibles 
        self.list_enemies = [] # lista que contendra a los enemigos
        self.enemies_x = enemies_x # total de enemigos horizontales
        self.enemies_y = enemies_y # total de enemigos verticales
        self.rompible_file = rompible_file # archivo de cajas irrompibles
        m = mult(self.width, self.height)
        if(self.width > self.height):
            self.window_height = 670
            self.window_width = m * self.window_height
        else:
            self.window_width = 512
            self.window_height = m * self.window_width
            
        # Define la forma del espacio de observación (en este caso, una imagen binaria)
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, 1, shape=(2,), dtype=int),
            }
        )

        # Define el espacio de acción (puedes personalizar esto según tu entorno)
        self.action_space = spaces.Discrete(4)  # Ejemplo: acciones discretas 0, 1, 2, 3

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
            4: np.array([0, 0]),
            5: np.array([0, 0]),
        }
        self._action_to_names = {
            0: 'RIGHT',
            1: 'DOWN',
            2: 'LEFT',
            3: 'UP',
            4: 'BOMB',
            5: 'WAIT',
        }
        

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Define cualquier otro atributo específico de tu entorno
        self.window = None
        self.clock = None

        # Inicializa el estado inicial de tu entorno
        #self.state = np.zeros((width, height), dtype=np.float32)

    
    def _tile_is_free(self,direction):
        movement = self._agent_location + direction
        if (exist(self.list_boxes,movement)):
            print("ESTOY CHOCANDO LOL")
            return False
        return True

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}
    
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                np.array(self._agent_location) - np.array(self._target_location), ord=1
            )
        }
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.player_alive = True 
        # generamos las cajas irrompibles
        for i in range(self.width):
            for j in range(self.height):                        
                if ((i % 2 == 1) and (j % 2 == 1)):
                    self.list_boxes.append(Box(np.array([i,j]),False,False,False))
        
        # generamos las cajas rompibles
        if not self.rompible_file == '':
            rompible_info = []
            with open(self.rompible_file, "r") as file:
                for line in file:
                    rompible_info.append(line.strip())
            if not rompible_info == []:
                if len(rompible_info) == self.boxes:
                # Procesa la información y genera cajas irrompibles
                    rompible_coordinates = []
                    for line in rompible_info:
                        x, y = map(int, line.split(","))
                        rompible_coordinates.append((x, y))
                    for coordinates in rompible_coordinates:
                        self.list_boxes.append(Box(np.array(coordinates), True, False,False))
                        self.list_boxes_breakable.append(Box(np.array(coordinates), True, False,False))
        else:
            i=0
            while i < self.boxes:
                box_pos = np.array([self.np_random.integers(0,self.width-1,dtype=int),self.np_random.integers(0,self.height-1,dtype=int)])
                for j in range(len(self.list_boxes)):
                    if not (exist(self.list_boxes,box_pos)):
                        self.list_boxes.append(Box(box_pos,True,False,False))
                        self.list_boxes_breakable.append(Box(box_pos,True,False,False))
                        i+=1

        # Choose the agent's location uniformly at random
        aux = True
        while (aux):
            x = self.np_random.integers(0, self.width, dtype=int)
            y = self.np_random.integers(0, self.height, dtype=int)
            if not (exist(self.list_boxes,np.array([x,y]))):
                aux = False
                self._agent_location = np.array([x,y])


        # Choose target location between breakable boxes
        target_box = self.list_boxes_breakable[self.np_random.integers(0,len(self.list_boxes_breakable),dtype=int)]
        for i in range(len(self.list_boxes)):
            if (np.array_equal(target_box.pos,self.list_boxes[i].pos)):
                self.list_boxes[i].isTarget = True
                self._target_location = target_box.pos
                self._target_index = i


        # Choose random position for the enemies in horizontal axis
        i = 0
        while i < self.enemies_x:
            m = self.np_random.integers(0, self.width, dtype=int)
            n = self.np_random.integers(0, self.height, dtype=int)
            if not (exist(self.list_boxes,np.array([m,n])) or np.array_equal(self._agent_location,np.array([m,n])) or exist(self.list_enemies,np.array([m,n]))):
                w = self.np_random.integers(0, 2, dtype=int) # 0: Up, 1: Down
                self.list_enemies.append(Enemy(np.array([m,n]),0,w,True)) # 0 en orientation es horizontal
                i+=1

        i = 0
        # Choose random position for the enemies in vertical axis
        while i < self.enemies_y:
            o = self.np_random.integers(0, self.width, dtype=int)
            p = self.np_random.integers(0, self.height, dtype=int)
            if not (exist(self.list_boxes,np.array([o,p])) or np.array_equal(self._agent_location,np.array([o,p])) or exist(self.list_enemies,np.array([o,p]))):
                w = self.np_random.integers(0, 2, dtype=int) # 0: Left, 1: Right
                self.list_enemies.append(Enemy(np.array([o,p]),1,w,True)) # 1 en orientation es vertical
                i+=1
        observation = self._get_obs()
        info = self._get_info()
        for i in range(len(self.list_enemies)):
            print(self.list_enemies[i].get_state())

        if self.render_mode == "human":
            self._render_frame()

        return observation, info



    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        print(self._action_to_names[action])
        # We use `np.clip` to make sure we don't leave the grid
        x,y = self._agent_location + direction
        print(x,y)

        #-------------------------------------Agent movement-------------------------------------------#
        if (action == 2 or action == 0) and self._tile_is_free(direction):
            self._agent_location = np.clip(
                self._agent_location + direction, 0, self.width - 1
            )
        elif (action == 1 or action == 3) and self._tile_is_free(direction):
            self._agent_location = np.clip(
                self._agent_location + direction, 0, self.height - 1
            )
        if (exist(self.list_enemies,self._agent_location)):
            terminated = np.array_equal(self._agent_location, self._target_location)
            reward = 1 if terminated else 0  # Binary sparse rewards
            observation = self._get_obs()
            info = self._get_info()
            return observation, reward, terminated, True, info

        
        #-------------------------------------Enemy Movement-------------------------------------------#
        for i in range(len(self.list_enemies)):
            if (self.list_enemies[i].isAlive): # verificamos que este vivo
                # Separamos el movimiento entre vertical y horizontal
                if self.list_enemies[i].orientation == 0: #movimiento horizontal
                    if self.list_enemies[i].way == 0: #movimiento a la izquierda
                        # verificamos que no choque con el limite
                        if (self.list_enemies[i].pos[0] == 0) or (exist(self.list_boxes,self.list_enemies[i].pos + self._action_to_direction[2]) or (exist(self.list_enemies,self.list_enemies[i].pos + self._action_to_direction[2]))): # llego al limite de la izquierda, entonces lo cambiamos de sentido
                            self.list_enemies[i].way = 1
                            if exist(self.list_boxes,self.list_enemies[i].pos + self._action_to_direction[0]):
                                self.list_enemies[i].pos = self.list_enemies[i].pos + self._action_to_direction[4] # lo dejamos quieto
                            else:
                                self.list_enemies[i].pos = self.list_enemies[i].pos + self._action_to_direction[0] #lo movemos a la derecha
                        else:
                            self.list_enemies[i].pos = self.list_enemies[i].pos + self._action_to_direction[2] # lo movemos a la izquierda

                    else: #movimiento a la derecha
                        # verificamos que no choque con el limite
                        if (self.list_enemies[i].pos[0] == self.width-1) or (exist(self.list_boxes,self.list_enemies[i].pos + self._action_to_direction[0]) or (exist(self.list_enemies,self.list_enemies[i].pos + self._action_to_direction[0]))):
                            self.list_enemies[i].way = 0
                            if exist(self.list_boxes,self.list_enemies[i].pos + self._action_to_direction[0]):
                                self.list_enemies[i].pos = self.list_enemies[i].pos + self._action_to_direction[4] # lo dejamos quieto
                            else:   
                                self.list_enemies[i].pos = self.list_enemies[i].pos + self._action_to_direction[2] # lo movemos a la izquierda
                        else:
                            self.list_enemies[i].pos = self.list_enemies[i].pos + self._action_to_direction[0] # lo movemos a la derecha
                
                
                else:
                    # movimiento vertical
                    if self.list_enemies[i].way == 0: # movimiento hacia arriba
                        # se verifica que no choque con nada
                        if ((self.list_enemies[i].pos[1] == 0) or (exist(self.list_boxes,self.list_enemies[i].pos + self._action_to_direction[3])) or (exist(self.list_enemies,self.list_enemies[i].pos + self._action_to_direction[3]))):
                            self.list_enemies[i].way = 1
                            if exist(self.list_boxes,self.list_enemies[i].pos + self._action_to_direction[1]):
                                self.list_enemies[i].pos = self.list_enemies[i].pos + self._action_to_direction[4]
                            else:
                                self.list_enemies[i].pos = self.list_enemies[i].pos + self._action_to_direction[1] # lo movemos hacia abajo
                        else:
                            self.list_enemies[i].pos = self.list_enemies[i].pos + self._action_to_direction[3] # lo movemos hacia arriba
                    else: # movimiento hacia abajo
                        # se verifica que no choque con nada
                        if ((self.list_enemies[i].pos[1] == self.height-1) or (exist(self.list_boxes,self.list_enemies[i].pos + self._action_to_direction[1])) or (exist(self.list_enemies,self.list_enemies[i].pos + self._action_to_direction[1]))):
                            self.list_enemies[i].way = 0
                            if exist(self.list_boxes,self.list_enemies[i].pos + self._action_to_direction[3]):
                                self.list_enemies[i].pos = self.list_enemies[i].pos + self._action_to_direction[4]
                            else:
                                self.list_enemies[i].pos = self.list_enemies[i].pos + self._action_to_direction[3] # lo movemos hacia arriba
                        else:
                            self.list_enemies[i].pos = self.list_enemies[i].pos + self._action_to_direction[1] # lo movemos hacia abajo


        #-------------------------------------Bomb Placement-------------------------------------------#
        #WIP

        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        

        return observation, reward, terminated, False, info




    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("Mi Juego Genial")
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((255, 255, 255))
        if(self.width > self.height):
            pix_square_size = (
                self.window_width / self.width
        )  # The size of a single grid square in pixels
        else:
            pix_square_size = (
                self.window_height / self.height
            )

        # ---------------------------Now we draw the agent -------------------------
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # ------------------------------------Now we draw the boxes------------------------------------
        for i in range(len(self.list_boxes)):
            if (self.list_boxes[i].isBreakable):
                if(self.list_boxes[i].isTarget and self.list_boxes[i].isBroken): # target visible
                    pygame.draw.rect(
                        canvas,
                        (52, 118, 32),
                        pygame.Rect(
                            (self.list_boxes[i].pos * pix_square_size),
                            (pix_square_size, pix_square_size),
                        ),
                    )
                elif not (self.list_boxes[i].isBroken): #cajas todavia visibles
                    pygame.draw.rect(
                    canvas,
                    (157, 124, 63),
                    pygame.Rect(
                        (self.list_boxes[i].pos * pix_square_size),
                        (pix_square_size, pix_square_size),
                    ),
                )
            else: # undestructible boxes
                pygame.draw.rect(
                    canvas,
                    (88, 82, 72),
                    pygame.Rect(
                        (self.list_boxes[i].pos * pix_square_size),
                        (pix_square_size, pix_square_size),
                    ),
                )

        # ------------------------------------Now we draw the enemies------------------------------------
        for i in range(len(self.list_enemies)):
            pygame.draw.circle(
                canvas,
                (184, 22, 37),
                (self.list_enemies[i].pos + 0.5) * pix_square_size,
                pix_square_size / 3,
            )

        # ------------------------------------Now we draw the bombs------------------------------------
        #WIP


        # Finally, add some gridlines
        for x in range(self.height + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_width, pix_square_size * x),
                width=3,
            )
        
        for y in range(self.width + 1):
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * y, 0),
                (pix_square_size * y, self.window_height),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )



file_str = "cajas.txt"

# Ejemplo de uso
env = BombermanEnv(15,11,20,3,3,'','human')

for episode in range(100):

    print("Episode ", episode)

    state, info = env.reset()


    return_episode = 0.0
    done = False
    while not done:
         # Seleccionar una accion aleatoria
        action = env.action_space.sample()

        # Genere una transicion
        next_state, reward, terminal, truncated, info = env.step(action)

        # Vaya al siguiente estado
        state = next_state

        # Ajuste el retorno
        return_episode = reward + return_episode

        # Render the current state
        print(env.render())

        # End of the episode
        done = terminal or truncated



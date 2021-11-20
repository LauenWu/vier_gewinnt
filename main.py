import pyglet
from game import Game
import agent
import game
import numpy as np

class Window(pyglet.window.Window):

    def __init__(self):
        super().__init__()

        self.cell_size = 80
        self.cell_margin = 5
        self.lower_margin = 100
        self.set_size(game.m*self.cell_size, game.n*self.cell_size + self.lower_margin)
        self.game = Game(agent.Agent())

        self.colors = {
            1: (255, 0, 0),
            -1: (255, 255, 0),
            0: (50, 50, 50)}

        self.hovered_col = -1
        self.circles = np.ndarray((game.m, game.n), dtype=pyglet.shapes.Circle)
        self.circles_batch = pyglet.graphics.Batch()

        self.game_over_batch = pyglet.graphics.Batch()

        pyglet.text.Label(
            x=10, y=10, text='GAME OVER', 
            batch = self.game_over_batch, font_size=40
            )

        for i in range(game.m):
            for j in range(game.n):
                x = i * self.cell_size + self.cell_size / 2
                y = j * self.cell_size + self.cell_size / 2 + self.lower_margin
                self.circles[i,j] = pyglet.shapes.Circle(
                    x,y,self.cell_size/2-self.cell_margin, 
                    color=self.colors[0], batch=self.circles_batch
                    )
        

    def on_draw(self):
        self.clear()
        for i, row in enumerate(self.game.playfield):
            for j, val in enumerate(row):
                self.circles[j,i].color = self.colors[val]
                self.circles[j,i].opacity = 255

        if not self.game.game_over:
            row = self.game.col_height[self.hovered_col]
            self.circles[self.hovered_col, row].color = self.colors[self.game.marker]
            self.circles[self.hovered_col, row].opacity = 50
        
        self.circles_batch.draw()

        if self.game.game_over:
            self.game_over_batch.draw()

    def on_mouse_motion(self, x, y, dx, dy):
        self.hovered_col = int(x/self.cell_size)

    def on_mouse_press(self, x, y, button, modifiers):
        if self.game.game_over:
            return
        if button == 1:
            self.game.play_col(self.hovered_col)
        

        





if __name__ == '__main__':
    window = Window()
    pyglet.app.run()
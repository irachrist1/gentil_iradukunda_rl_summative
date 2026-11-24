"""SDL simulation display driven by the exact state used in replay."""

from pathlib import Path

import moderngl
import pygame

from environment.custom_env import DISTRICTS, HORIZON


class DispatchDisplay:
    WIDTH, HEIGHT = 1080, 672
    BG = "#edf3f1"
    PANEL = "#ffffff"
    INK = "#172d2a"
    MUTED = "#4c625d"
    GREEN = "#23785e"
    AMBER = "#a25124"
    LINE = "#d2e0dc"

    def __init__(self, *, headless=False):
        pygame.display.init()
        pygame.font.init()
        self.context = None
        if not headless:
            try:
                self._open_opengl()
            except (pygame.error, moderngl.Error, ValueError) as error:
                print(f"OpenGL unavailable ({error}). Using the SDL display.", flush=True)
                pygame.display.quit()
                pygame.display.init()
        if self.context is None:
            self.surface = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        else:
            self.surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("District Food Dispatch | Simulation")
        self.fonts = {size: pygame.font.Font(None, size) for size in (19, 21, 24, 26, 30, 34, 44)}

    def _open_opengl(self):
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.set_mode((self.WIDTH, self.HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
        context = moderngl.create_context()
        program = context.program(
            vertex_shader='''#version 330
                out vec2 uv;
                void main() {
                    vec2 corner = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
                    uv = corner;
                    gl_Position = vec4(2.0 * corner - 1.0, 0.0, 1.0);
                }''',
            fragment_shader='''#version 330
                uniform sampler2D board;
                in vec2 uv;
                out vec4 color;
                void main() { color = texture(board, uv); }''',
        )
        self.texture = context.texture((self.WIDTH, self.HEIGHT), components=3)
        self.texture.use(0)
        self.triangle = context.vertex_array(program, [])
        self.context = context
        print(f"Display: OpenGL {context.info['GL_VERSION']}", flush=True)

    def text(self, value, x, y, *, size=24, color=None):
        self.surface.blit(self.fonts[size].render(str(value), True, color or self.INK), (x, y))

    def panel(self, x, y, width, height, color=None):
        pygame.draw.rect(self.surface, color or self.PANEL, (x, y, width, height), border_radius=14)

    def bar(self, value, total, x, y, width):
        pygame.draw.rect(self.surface, self.LINE, (x, y, width, 7), border_radius=3)
        if value:
            pygame.draw.rect(self.surface, self.GREEN, (x, y, round(width * min(value / total, 1)), 7),
                             border_radius=3)

    def draw(self, env, policy, seed, totals, history):
        self.surface.fill(self.BG)
        self.text("District Food Dispatch", 30, 27, size=44)
        self.text("Simulated emergency food distribution in Rwanda", 32, 76, size=24, color=self.MUTED)
        self.text(f"{policy.upper()}  /  SEED {seed}", 840, 45, size=21, color=self.GREEN)

        self.panel(30, 115, 691, 483)
        self.text("Depot and district network", 51, 134, size=26)
        self.text("Schematic routes", 51, 163, size=19, color=self.MUTED)
        positions = [(56, 202), (490, 202), (56, 436), (490, 436)]
        day = min(env.day, HORIZON - 1)
        previous = history[-1] if history else None
        hub_center = (377, 376)
        for i, (x, y) in enumerate(positions):
            center = (x + 103, y + 65)
            active = env.inbound[i] or (previous is not None and previous["delivered"][i])
            color = self.GREEN if active else self.LINE
            pygame.draw.line(self.surface, color, hub_center, center, 5 if active else 2)

        for i, (x, y) in enumerate(positions):
            self.panel(x, y, 207, 127, "#f0f5f2")
            self.text(DISTRICTS[i], x + 13, y + 12, size=30)
            self.text(f"{env.stock[i]} on hand", x + 13, y + 44, size=21, color=self.MUTED)
            self.text(f"+{env.inbound[i]} due", x + 124, y + 44, size=19, color=self.GREEN)
            self.bar(env.stock[i], 18, x + 13, y + 71, 181)
            demand_alert = env.day < HORIZON and env.alerts[day, i]
            road_alert = env.day < HORIZON and env.roads[day, i]
            signal = "Demand alert" if demand_alert else "Normal demand"
            if road_alert:
                signal += " / slow road"
            self.text(signal if env.day < HORIZON else "Episode ended", x + 13, y + 92,
                      size=19, color=self.AMBER if demand_alert or road_alert else self.MUTED)

        self.panel(295, 322, 166, 110, "#dcf0e7")
        self.text("Central hub", 310, 335, size=30)
        self.text(f"{env.hub} / 36 crates", 310, 368, size=21, color=self.GREEN)
        self.text(f"Supplier due  +{env.supplier}", 310, 399, size=19, color=self.MUTED)

        self.panel(742, 115, 308, 202)
        self.text(f"Day {env.day:02d} of {HORIZON}", 760, 133, size=34)
        self.bar(env.day, HORIZON, 762, 178, 269)
        for idx, (label, key, color) in enumerate((
            ("Cumulative reward", "return", self.GREEN), ("Unmet demand", "unmet", self.AMBER),
            ("Requested crates", "demand", self.INK),
        )):
            y = 201 + idx * 32
            self.text(label, 762, y, size=21, color=self.MUTED)
            self.text(f"{totals[key]:.2f}" if key == "return" else totals[key],
                      951, y, size=24, color=color)

        self.panel(742, 333, 308, 265)
        self.text("Recent decisions", 760, 351, size=26)
        if not history:
            self.text("Ready for the first decision", 762, 398, size=21, color=self.MUTED)
        for idx, item in enumerate(reversed(history[-4:])):
            y = 388 + idx * 46
            self.text(f"{item['day']:02d}  {item['action']}", 761, y, size=21)
            status = "rejected" if item["rejected"] else f"{item['unmet_total']} unmet"
            self.text(f"{status}  /  reward {item['reward']:.2f}", 780, y + 20, size=19, color=self.MUTED)

        self.text("Green route  active shipment or delivery   •   Amber text  demand or road alert",
                  32, 622, size=21, color=self.MUTED)
        self.text("Esc to exit", 962, 622, size=21, color=self.MUTED)
        if self.context is not None:
            self.texture.write(pygame.image.tobytes(self.surface, "RGB", flipped=True))
            self.triangle.render(vertices=3)
        pygame.display.flip()

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(self.surface, path)

    @staticmethod
    def wait(seconds):
        deadline = pygame.time.get_ticks() + round(seconds * 1000)
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return False
            if pygame.time.get_ticks() >= deadline:
                return True
            pygame.time.wait(20)

    @staticmethod
    def close():
        pygame.quit()

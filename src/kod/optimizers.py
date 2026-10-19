import math
import src.kod.containers


class TriangleContainerOptimizer():
    """optimizator za pronalazenje najbolje rotacije i pomaka trokuta radi maksimizacije broja celija."""

    def __init__(self, container, grid):
        # container: spremnik koji sadrzi vrhove i metode za pozicioniranje
        # grid: mreza koja sadrzi celije i njihove dimenzije
        self.container = container
        self.grid = grid
        self.d = self.grid.d

    def solve(self, angle_precision=5, shift_precision=1):
        """glavna funkcija koja pronalazi rotaciju i smjestaj trokuta koji maksimizira broj celija.

        argumenti:
        angle_precision (int): korak kuta za pretragu rotacija
        shift_precision (int): korak pomaka po x i y osima

        povratna vrijednost:
        tuple: (best_c_ids, best_v, self.grid) gdje su best_c_ids id-ovi obuhvacenih celija, best_v vrhovi trokuta i grid objekt
        """
        all_c = self.grid.get_all_cell_coordinates()
        original_vertices = self.container.vertices
        max_count = 0
        best_v = []
        best_c_ids = []
        ids = []
        # best_angle=0
        # best_dx=0
        # best_dy=0
        base_angles = self.get_alignment_angles(original_vertices)
        search_angles = sorted(list(set(list(range(0, 60, angle_precision)) + base_angles)))
        for angle in search_angles:
            rotated_v = containers.rotate_and_shift_triangle(original_vertices, angle)
            for dx in range(0, self.d + 1, shift_precision):
                for dy in range(0, self.d + 1, shift_precision):
                    self.container.update_position(angle, dx, dy)
                    ids = self.container.get_cells_in_shape(all_c, self.d)
                    # if(dx==6 and dy==12 and angle in base_angles):
                    #     self.grid.plot(vertices = self.container.vertices, cell_d = 18, inside_ids = ids, save=True, name=f'triangle_{round(angle,2)}_{dx}_{dy}.png', title = f'Ukupan broj ćelija u spremniku: {len(ids)}, kut: {round(angle,2)}, dx:{dx}, dy={dy}')

                    if len(ids) > max_count:
                        max_count = len(ids)
                        best_v = self.container.vertices
                        best_c_ids = ids
                        # best_angle=angle
                        # best_dx=dx
                        # best_dy=dy
        #self.grid.plot(vertices =best_v, cell_d = 18, inside_ids = best_c_ids, save=True, name=f'best_triangle.png', title = f'Ukupan broj ćelija u spremniku: {len(best_c_ids)}, kut: {round(best_angle,2)}, dx:{best_dx}, dy={best_dy}')

        return best_c_ids, best_v, self.grid

    def get_alignment_angles(self, vertices):
        """racuna kutove potrebne da svaka stranica trokuta bude paralelna s celijama.

        argumenti:
        vertices (list): lista točaka (x, y) koje definiraju vrhove trokuta

        povratna vrijednost:
        list: lista kutova u stupnjevima za poravnanje
        """
        angles = []
        n = len(vertices)
        for i in range(n):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % n]
            angle_rad = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
            angle_deg = math.degrees(angle_rad)
            angles.append(-angle_deg)
        return angles
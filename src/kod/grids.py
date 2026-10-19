import math
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


class Cell:
    """klasa koja predstavlja jednu celiju u baterijskom paketu."""

    def __init__(self, cell_id, row, col, x, y):
        # cell_id (int): jedinstveni identifikator celije
        # row (int): indeks retka u matrici
        # col (int): indeks stupca u matrici
        # x (float): x koordinata u mm
        # y (float): y koordinata u mm
        self.id = cell_id
        self.row = row
        self.col = col
        self.x = x
        self.y = y
        self.group = None


class Grid:
    """bazna klasa za mrezu baterijskih celija."""

    def __init__(self, cell_distance, width, height):
        # cell_distance (float): razmak/promjer celija u mm
        # width (float): sirina baterijskog paketa u mm
        # height (float): visina baterijskog paketa u mm
        self.d = cell_distance
        self.H = height
        self.W = width

        self.rows = 0
        self.cells_per_row_1 = 0
        self.cells_per_row_2 = 0
        self.total_cells = 0

        self.matrix = None
        self.cells = []
        self.cell_matrix = None

    def get_all_cell_coordinates(self):
        """dohvaca koordinate svih celija u mrezi.

        povratna vrijednost:

        list: lista parova koordinate (x, y) za svaku celiju.
        """
        return [(cell.x, cell.y) for cell in self.cells]

    def plot(self, cell_to_group=None, save=False, name="battery_pack.png", title="", vertices=None, inside_ids=None, cell_d=None):
        """iscrtava baterijski paket i celije pomocu matplotliba.

        argumenti:
        cell_to_group (list, optional): lista grupa po celijama.
        save (bool): spremiti sliku na disk ili prikazati.
        name (str): naziv datoteke ako se sprema.
        title (str): naslov prikaza.
        vertices (list, optional): vrhovi poligona za prikaz.
        inside_ids (list, optional): id-ovi celija unutar poligona.
        """
        if not self.cells:
            print("fit optimal pack first!")
            return
        if cell_d is None:
            cell_d=self.d

        base_scale = 12
        if self.W >= self.H:
            width_fig = base_scale
            height_fig = base_scale * (self.H / self.W)
        else:
            height_fig = base_scale
            width_fig = base_scale * (self.W / self.H)

        width_fig = max(width_fig, 4)
        height_fig = max(height_fig, 4)

        fig, ax = plt.subplots(figsize=(width_fig, height_fig))

        # okvir baterije
        rect = patches.Rectangle((0, 0), self.W, self.H, linewidth=2, edgecolor="black", facecolor="none")
        ax.add_patch(rect)

        if vertices is not None:
            triangle_patch = Polygon(vertices, closed=True, fill=None, edgecolor="blue", linewidth=2, linestyle="--")
            ax.add_patch(triangle_patch)

        # pretvaramo u set radi brze pretrage o(1)
        inside_set = set(inside_ids) if inside_ids is not None else set()
        added_labels = set()
        for i, cell in enumerate(self.cells):
            y = cell.y
            x = cell.x
            current_group = cell_to_group[i] if cell_to_group is not None else cell.group

            if 0 <= x <= self.W and 0 <= y <= self.H:
                if current_group is None or current_group == -1:
                    if cell.id in inside_set or inside_ids is None:
                        c_label = 'Ćelije unutar spremnika' if 'INSIDE' not in added_labels else None
                        if inside_ids is None:
                            circle_color= "grey"
                        else:
                            circle_color = "purple"
                        added_labels.add('INSIDE')
                    else:
                        c_label = 'Ćelije izvan spremnika' if 'OUTSIDE' not in added_labels else None
                        circle_color='grey'
                        added_labels.add('OUTSIDE')
                else:
                    circle_color = "tab:blue"

                circle = patches.Circle((x, y), cell_d / 2, edgecolor="black", facecolor=circle_color, alpha=0.6, label = c_label)
                ax.add_patch(circle)

        ax.legend(loc="upper right", fontsize=14)
        ax.set_xlim(-5, self.W + 0.1 * self.W)
        ax.set_ylim(-5, self.H + 0.1 * self.H)
        ax.set_xlabel('Širina (mm)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Visina (mm)', fontsize=16, fontweight='bold')
        ax.set_aspect("equal")
        ax.set_title(str(title), fontsize=18)

        if save:
            plt.savefig(name, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def create_subgrid(self, inside_ids):
        """uklanja celije koje nisu u inside_ids i smanjuje dimenzije matrice.

        argumenti:
        inside_ids (list/set): popis id-ova celija koje ostaju u mrezi.
        """
        inside_set = set(inside_ids)
        cells_to_keep = []

        # uklanjanje stanica iz matrica koje nisu zadrzane
        for cell in self.cells:
            if cell.id in inside_set:
                cells_to_keep.append(cell)
            else:
                self.matrix[cell.row][cell.col] = "x"
                self.cell_matrix[cell.row][cell.col] = None

        # filtriranje praznih redaka i stupaca
        row_mask = np.any(self.matrix != "x", axis=1)
        col_mask = np.any(self.matrix != "x", axis=0)

        # osiguranje za prazan subgrid
        if not np.any(row_mask) or not np.any(col_mask):
            self.matrix = np.empty((0, 0))
            self.cell_matrix = np.empty((0, 0), dtype=object)
            self.cells = []
            self.total_cells = 0
            self.rows = 0
            return

        self.matrix = self.matrix[row_mask][:, col_mask]
        self.cell_matrix = self.cell_matrix[row_mask][:, col_mask]
        self.cells = cells_to_keep

        # azuriranje redaka i stupaca celija u objektima
        new_rows, new_cols = self.cell_matrix.shape
        for r in range(new_rows):
            for c in range(new_cols):
                cell_obj = self.cell_matrix[r][c]
                if cell_obj is not None:
                    cell_obj.row = r
                    cell_obj.col = c

        # azuriranje celijskih id-ova i broja redaka
        for i, cell in enumerate(self.cells):
            cell.id = i

        self.total_cells = len(self.cells)
        self.rows = new_rows


class HexGrid(Grid):
    """heksagonalni raspored baterijskih celija."""

    def fit(self):
        """proracunava raspored i postavlja celije u heksagonalnu mrezu."""
        self.cells = []
        self.cell_matrix = None

        self.cells_per_row_1, self.cells_per_row_2, self.rows, self.total_cells = self.calculate_max_no_of_cells(self.d, self.H, self.W)

        self.matrix = self.create_config_matrix()
        self.cell_matrix = np.full(self.matrix.shape, None, dtype=object)

        cell_counter = 0
        r_h = self.d * np.sqrt(3) / 2
        n_larger_rows = (self.rows - 1) // 2 + 1
        x_offset = (n_larger_rows - 1) * self.d

        for i in range(self.rows):
            for j in range(self.matrix.shape[1]):
                if self.matrix[i][j] == "1":
                    y = (self.rows - 1 - i) * r_h + (self.d / 2)
                    x = (j * self.d) + (i * self.d / 2) - x_offset + (self.d / 2)

                    new_cell = Cell(cell_counter, i, j, x, y)
                    self.cells.append(new_cell)
                    self.cell_matrix[i][j] = new_cell
                    cell_counter += 1

    def calculate_max_no_of_cells(self, cell_distance, height, width):
        """racuna maksimalni broj celija u heksagonalnom rasporedu.

        argumenti:
        cell_distance (float): razmak celija u mm.
        height (float): visina okvira u mm.
        width (float): sirina okvira u mm.

        povratna vrijednost:
        tuple: (cells_per_row_1, cells_per_row_2, rows, total_cells)
        """
        cells_per_row_1 = math.floor(width / cell_distance)
        extra_space = width - cells_per_row_1 * cell_distance

        if extra_space >= cell_distance / 2:
            cells_per_row_2 = cells_per_row_1
        else:
            cells_per_row_2 = cells_per_row_1 - 1

        row_height = cell_distance * np.sqrt(3) / 2
        num_extra_rows = math.floor((height - cell_distance) / row_height)
        rows = num_extra_rows + 1

        n_odd = (rows + 1) // 2
        n_even = rows // 2
        total_cells = (n_odd * cells_per_row_1) + (n_even * cells_per_row_2)

        return cells_per_row_1, cells_per_row_2, rows, total_cells

    def create_config_matrix(self):
        """kreira matricu sa 1 i x oznakama za heksagonalnu mrezu.

        povratna vrijednost:
        numpy.ndarray: 2d matrica sa string oznakama.
        """
        n_larger_rows = (self.rows - 1) // 2 + 1
        cols = self.cells_per_row_1 + n_larger_rows - 1
        matrix = np.full((self.rows, cols), "1")

        right_margin = cols - n_larger_rows
        left_margin = n_larger_rows

        if self.cells_per_row_1 != self.cells_per_row_2:
            if self.rows % 2 == 1:
                counter, counter2 = 0, 2
            else:
                counter, counter2 = 1, 1
                right_margin -= 1
        else:
            if self.rows % 2 == 1:
                counter, counter2 = 0, 1
            else:
                counter, counter2 = 1, 2

        for j in range(cols):
            if j < left_margin:
                for i in range(self.rows):
                    if i < (self.rows - 1 - counter):
                        matrix[i][j] = "x"
                counter += 2
            if j > right_margin:
                for i in range(self.rows):
                    if i >= (self.rows - counter2):
                        matrix[i][j] = "x"
                counter2 += 2
        return matrix

    def get_neighbouring_cells(self, cell_id):
        """dohvaca susjedne celije za zadanu heksagonalnu celiju.

        argumenti:
        cell_id (int): id celije za koju se traze susjedi.

        povratna vrijednost:
        list: lista susjednih cell objekata.
        """
        if cell_id < 0 or cell_id >= len(self.cells):
            return []

        cell = self.cells[cell_id]
        neighbours = []
        rows, cols = self.matrix.shape
        directions = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

        for dr, dc in directions:
            r, c = cell.row + dr, cell.col + dc
            if 0 <= r < rows and 0 <= c < cols:
                neighbour = self.cell_matrix[r][c]
                if neighbour is not None:
                    neighbours.append(neighbour)
        return neighbours


class RectangularGrid(Grid):
    """pravokutni raspored baterijskih celija."""

    def fit(self):
        """proracunava raspored i postavlja celije u pravokutnu mrezu."""
        self.cells = []
        self.cells_per_row_1 = math.floor(self.W / self.d)
        self.cells_per_row_2 = self.cells_per_row_1
        self.rows = math.floor(self.H / self.d)
        self.total_cells = self.cells_per_row_1 * self.rows

        self.matrix = np.full((self.rows, self.cells_per_row_1), "1")
        self.cell_matrix = np.full(self.matrix.shape, None, dtype=object)
        cell_counter = 0

        for i in range(self.rows):
            for j in range(self.cells_per_row_1):
                x = (j * self.d) + (self.d / 2)
                y = (self.rows - 1 - i) * self.d + (self.d / 2)
                new_cell = Cell(cell_counter, i, j, x, y)
                self.cell_matrix[i][j] = new_cell
                self.cells.append(new_cell)
                cell_counter += 1

    def get_neighbouring_cells(self, cell_id):
        """dohvaca susjedne celije za zadanu pravokutnu celiju.

        argumenti:
        cell_id (int): id celije za koju se traze susjedi.

        povratna vrijednost:
        list: lista susjednih cell objekata.
        """
        if cell_id < 0 or cell_id >= len(self.cells):
            return []

        cell = self.cells[cell_id]
        neighbours = []
        rows, cols = self.matrix.shape
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dr, dc in directions:
            r, c = cell.row + dr, cell.col + dc
            if 0 <= r < rows and 0 <= c < cols:
                neighbour = self.cell_matrix[r][c]
                if neighbour is not None:
                    neighbours.append(neighbour)
        return neighbours
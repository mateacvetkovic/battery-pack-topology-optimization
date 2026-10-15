"""
Definira klase i pomocne matematicke funkcije za rad s geometrijom kontejnera.
Primarno sluzi za izracun vrhova trokutastog kontejnera na temelju duljina stranica,
rotaciju i translaciju trokuta u ravnini, te provjeru koje se celije (baterije) 
u potpunosti nalaze unutar zadanog trokutastog prostora.
"""

import math
import numpy as np
from matplotlib.path import Path

class Container:
    def __init__(self, dimensions):
        """
        Inicijalizacija baznog kontejnera.
        Argumenti:
          - dimensions: lista dimenzija u cm
        Vraca:
          - None
        """
        # pretvaranje dimenzija iz cm u mm
        self.dimensions = [d * 10 for d in dimensions]


class TriangleContainer(Container):
    def __init__(self, dimensions):
        """
        Inicijalizacija trokutastog kontejnera i provjera valjanosti.
        Argumenti:
          - dimensions: lista s 3 duljine stranica trokuta
        Vraca:
          - None
        """
        super().__init__(dimensions)
        
        # provjera ima li trokut tocno 3 stranice i jesu li sve vece od 0
        if len(self.dimensions) != 3 or any(d <= 0 for d in self.dimensions):
            raise ValueError("Trokut mora imati tri stranice duljine > 0.")
        
        a, b, c = self.dimensions
        # provjera nejednakosti trokuta (svaka stranica mora biti kraca od zbroja preostale dvije)
        if a >= b + c or b >= a + c or c >= a + b:
            raise ValueError("Neispravne dimenzije! Zbroj dviju stranica mora biti veci od trece.")
        
        # racunanje pocetnih vrhova trokuta
        self.vertices = self.calculate_vertices(self.dimensions[0], self.dimensions[1], self.dimensions[2])
        self.original_vertices = self.vertices

    def calculate_vertices(self, a, b, c):
        """
        Racuna koordinate vrhova trokuta u ravnini na temelju duljina stranica.
        Argumenti:
          - a: duljina prve stranice
          - b: duljina druge stranice
          - c: duljina trece stranice
        Vraca:
          - lista od 3 koordinate vrhova trokuta [(x1,y1), (x2,y2), (x3,y3)]
        """
        # tocka A je u ishodistu (0,0)
        A = (0, 0)
        # tocka B lezi na x osi
        B = (c, 0)
        # racunanje kuta i koordinata za tocku C
        cos_alpha = (b**2 + c**2 - a**2) / (2 * b * c)
        x_c = b * cos_alpha
        y_c = b * math.sqrt(1 - cos_alpha**2)
        C = (x_c, y_c)
        return [A, B, C] 

    def calculate_triangle_grid_size(self, d):
        """
        Racuna potrebnu velicinu mreze za trokut. Mreza mora biti
        dovoljne velicione da pokrije sve moguce polozaje trokuta pri rotaciji.
        Argumenti:
          - d: udaljenost ili promjer celije
        Vraca:
          - broj koji predstavlja minimalnu dimenziju mreze
        """
        # uzima se najduza stranica trokuta za procjenu velicine mreze
        max_side = max(self.dimensions)
        return max_side + d + 1

    def get_cells_in_shape(self, grid_cell_coordinates, d):
        """
        Vraca listu indeksa celija koje u potpunosti staju unutar trokuta.
        Argumenti:
          - grid_cell_coordinates: lista (x,y) koordinata svake celije
          - d: promjer ili udaljenost izmedu celija baterije
        Vraca:
          - lista indeksa (int) celija koje staju unutar trokuta
        """
        cells_inside = []
        polygon_path = Path(self.vertices)
        for i, coord in enumerate(grid_cell_coordinates):
            #oblik mora sadrzavati cijelu celiju, ne samo njezino srediste
            left = (coord[0]-d/2, coord[1])
            right = (coord[0]+d/2, coord[1])
            up = (coord[0], coord[1]+d/2)
            down = (coord[0], coord[1]-d/2)
            if is_cell_completely_inside(polygon_path, self.vertices, coord, d):
                cells_inside.append(i)
        return cells_inside
    
    def update_position(self, angle, dx, dy):
        """
        Rotira trokut i pomice ga na novu poziciju.
        Argumenti:
          - angle: kut rotacije u stupnjevima
          - dx: pomak po x osi
          - dy: pomak po y osi
        Vraca:
          - None (azurira self.vertices)
        """
        # rotacija originalnih vrhova i njihovo pomicanje u prvi kvadrant
        rotated_vertices = rotate_and_shift_triangle(self.original_vertices, angle)
        # dodavanje konacnog translacijskog pomaka
        self.vertices = [(v[0] + dx, v[1] + dy) for v in rotated_vertices]


def get_dist_to_segment(p, a, b):
    """
    Racuna okomitu udaljenost tocke od pravca definiranog tockama A i B.
    Argumenti:
      - p: koordinate tocke (x, y)
      - a: pocetna tocka duzine (x, y)
      - b: zavrsna tocka duzine (x, y)
    Vraca:
      - float vrijednost okomite udaljenosti
    """
    p = np.array(p)
    a = np.array(a)
    b = np.array(b)
    ab = b - a
    ap = p - a
    L = np.linalg.norm(ab)
    distance = abs(np.cross(ab, ap)) / L
    
    return distance


def is_cell_completely_inside(polygon_path, vertices, center, d, epsilon=0.05):
    """
    Provjerava nalazi li se cijela celija (krug/stranice) unutar trokuta.
    Argumenti:
      - polygon_path: matplotlib Path objekt trokuta
      - vertices: lista vrhova trokuta
      - center: koordinate sredista celije (x, y)
      - d: promjer celije
      - epsilon: mala tolerancija za rubove
    Vraca:
      - True ako je celija potpuno unutar trokuta, inace False
    """
    # prvo provjeri je li samo srediste unutar trokuta
    if not polygon_path.contains_point(center, radius=epsilon):
        return False
    
    # provjera udaljenosti sredista od svih stranica trokuta
    for i in range(len(vertices)):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % len(vertices)] 
        
        dist = get_dist_to_segment(center, v1, v2)
        # ako je srediste preblizu stranici celija izlazi van trokuta
        if dist < (d/2 - epsilon):
            return False
    return True

def rotate_point(point, angle_deg, pivot):
    """
    Rotira jednu tocku oko zadane tocke (pivot) za zadani kut.
    Argumenti:
      - point: koordinate tocke koja se rotira (x, y)
      - angle_deg: kut rotacije u stupnjevima
      - pivot: koordinate tocke oko koje se rotira (x, y)
    Vraca:
      - tupl s novim koordinatama rotirane tocke (qx, qy)
    """
    angle_rad = math.radians(angle_deg)
    px, py = point
    ox, oy = pivot

    # primjena rotacijske matrice
    qx = ox + math.cos(angle_rad) * (px - ox) - math.sin(angle_rad) * (py - oy)
    qy = oy + math.sin(angle_rad) * (px - ox) + math.cos(angle_rad) * (py - oy)
    
    return qx, qy

def rotate_triangle(vertices, angle_deg):
    """
    Rotira sve vrhove trokuta oko njegovog tezista.
    Argumenti:
      - vertices: lista vrhova trokuta
      - angle_deg: kut rotacije u stupnjevima
    Vraca:
      - lista novih rotiranih koordinata vrhova [(x1,y1), (x2,y2), (x3,y3)]
    """
    # pronalazenje tezista trokuta
    cx = sum(v[0] for v in vertices) / 3
    cy = sum(v[1] for v in vertices) / 3
    centroid = (cx, cy)

    # rotiranje svih vrhova oko izracunatog tezista
    rotated_vertices = [rotate_point(v, angle_deg, centroid) for v in vertices]
    
    return rotated_vertices

def rotate_and_shift_triangle(vertices, angle):
    """
    Rotira trokut i poravnava ga natrag u prvi kvadrant (pozitivne koordinate).
    Argumenti:
      - vertices: lista vrhova trokuta
      - angle: kut rotacije u stupnjevima
    Vraca:
      - lista vrhova trokuta koji su rotirani i pomaknuti u prvi kvadrant
    """
    rotated = rotate_triangle(vertices, angle)
    
    # pronalazenje najmanjih vrijednosti koordinata nakon rotacije
    min_x = min(v[0] for v in rotated)
    min_y = min(v[1] for v in rotated)
    
    # pomak trokuta tako da dira osi i nema negativne koordinate
    shift_x = -min_x
    shift_y = -min_y
    
    shifted_vertices = [(v[0] + shift_x, v[1] + shift_y) for v in rotated]
    
    return shifted_vertices
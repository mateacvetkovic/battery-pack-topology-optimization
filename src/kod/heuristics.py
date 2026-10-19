from collections import deque
import math
import random
from src.kod.BatterySolution import BatterySolution
from src.kod.grids import HexGrid, RectangularGrid
import numpy as np


def snake_pattern_rows(pack):
    """Generira rjesenje popunjavanjem po redovima u snake uzorku.

    Argumenti:
        pack (BatteryPack): objekt baterijskog paketa s mrezom i parametrima.

    Vraca:
        BatterySolution | None: generirano rjesenje ili None ako je mreza prazna.
    """
    # provjeravamo ima li uopce celija u mrezi
    if not pack.grid.cells:
        return None

    sol = BatterySolution(pack)
    ordered_cells = []
    num_rows, num_cols = pack.grid.cell_matrix.shape

    # idemo red po red i skupljamo celije
    for i in range(num_rows):
        row_cells = []
        for j in range(num_cols):
            cell = pack.grid.cell_matrix[i, j]
            if cell is not None:
                row_cells.append(cell)

        # okreci svaki drugi red da dobijemo snake efekt
        if i % 2 == 1:
            row_cells.reverse()

        ordered_cells.extend(row_cells)

    # dodjeljujemo celije grupama po skupljenom redoslijedu
    total_needed = pack.no_groups * pack.no_cells_per_group
    for idx, cell in enumerate(ordered_cells):
        if idx < total_needed:
            group_id = idx // pack.no_cells_per_group
            sol.assign_cell_to_group(cell.id, group_id)

    return sol


def snake_pattern_columns(pack):
    """Generira rjesenje popunjavanjem po stupcima u snake uzorku.

    Argumenti:
        pack (BatteryPack): objekt baterijskog paketa.

    Vraca:
        BatterySolution | None: generirano rjesenje ili None ako nema celija.
    """
    # provjera prazne mreze
    if not pack.grid.cells:
        return None

    sol = BatterySolution(pack)
    column_ordered_cells = []
    num_rows, num_cols = pack.grid.cell_matrix.shape

    # idemo stupac po stupac
    for j in range(num_cols):
        row_indices = list(range(num_rows))
        # okreni smjer redova za svaki drugi stupac
        if j % 2 == 1:
            row_indices.reverse()

        for i in row_indices:
            cell = pack.grid.cell_matrix[i, j]
            if cell is not None:
                column_ordered_cells.append(cell)

    total_needed = pack.no_groups * pack.no_cells_per_group
    for idx, cell in enumerate(column_ordered_cells):
        if idx < total_needed:
            group_id = idx // pack.no_cells_per_group
            sol.assign_cell_to_group(cell.id, group_id)

    return sol


def snake_pattern_bottom_up(pack):
    """Generira snake uzorak po redovima krecuci od dna prema vrhu.

    Argumenti:
        pack (BatteryPack): objekt baterijskog paketa.

    Vraca:
        BatterySolution | None: generirano rjesenje ili None ako nema celija.
    """
    if not pack.grid.cells:
        return None

    sol = BatterySolution(pack)
    # okreni matricu naopako da krenemo s dna
    flipped_matrix = np.flipud(pack.grid.cell_matrix)
    ordered_cells = []
    num_rows, num_cols = flipped_matrix.shape

    # kupimo celije od dna prema vrhu
    for i in range(num_rows):
        row_cells = []
        for j in range(num_cols):
            cell = flipped_matrix[i, j]
            if cell is not None:
                row_cells.append(cell)
        if i % 2 == 1:
            row_cells.reverse()
        ordered_cells.extend(row_cells)

    # dodijeli grupe
    total_needed = pack.no_groups * pack.no_cells_per_group
    for idx, cell in enumerate(ordered_cells):
        if idx < total_needed:
            group_id = idx // pack.no_cells_per_group
            sol.assign_cell_to_group(cell.id, group_id)

    return sol

def max_shared_boundary_distance_based_universal(
    pack,
    void_id=-1,
    temperature=0,
    p_stochastic=0.2,
    strategy="horizontal",
    direction="top-down",
    diagonal_type="backslash",
    reverse_columns=False,
):
    """Univerzalni algoritam temeljen na optimizaciji udaljenosti i granica.

    Argumenti:
        pack (BatteryPack): paket baterija.
        void_id (int, optional): id za neiskorištene celije (praznine).
          default: -1.
        temperature (float, optional): parametar za stohasticki odabir. default:
          0.
        p_stochastic (float, optional): vjerojatnost stohastickog izbora.
          default: 0.2.
        strategy (str, optional): smjer slaganja ("horizontal", "vertical",
          "diagonal"). default: "horizontal".
        direction (str, optional): smjer unutar strategije ("top-down",
          "bottom-up", "left-to-right", "right-to-left"). default: "top-down".
        diagonal_type (str, optional): tip dijagonale ("backslash", "slash").
          default: "backslash".
        reverse_columns (bool, optional): obrni redoslijed stupaca kod
          dijagonale. default: False.

    Vraca:
        BatterySolution | None: rjesenje ili None ako algoritam zapne.
    """
    if not pack.grid.cells:
        return None

    sol = BatterySolution(pack)
    cells_per_group = pack.no_cells_per_group
    total_groups = pack.no_groups

    used_cells = set()
    previous_group_cells = []

    is_hex = type(pack.grid).__name__ == "HexGrid"

    # pomocne funkcije za pozicije
    def get_col_key(c):
        return (2 * c.col + c.row) if is_hex else c.col

    def get_diag_index(c):
        if diagonal_type == "slash":
            idx = c.row + c.col
        else:
            idx = c.col
        return -idx if reverse_columns else idx

    # sortiramo celije prema odabranoj strategiji
    if strategy == "horizontal":
        if direction == "bottom-up":
            ordered_cells = sorted(pack.grid.cells, key=lambda c: (c.y, c.x))
        else:
            ordered_cells = sorted(pack.grid.cells, key=lambda c: (-c.y, c.x))

    elif strategy == "vertical":
        col_dir = 1 if direction == "left-to-right" else -1
        ordered_cells = sorted(
            [c for c in pack.grid.cells if c is not None],
            key=lambda c: (col_dir * get_col_key(c), c.row),
        )

    elif strategy == "diagonal":
        if diagonal_type == "slash":
            ordered_cells = sorted(pack.grid.cells, key=lambda c: (get_diag_index(c), -c.row if not reverse_columns else c.row))

        else:
            ordered_cells = sorted(pack.grid.cells, key=lambda c: (get_diag_index(c), c.row))


    # provjera ima li dovoljno povezanih slobodnih celija
    def get_connected_free_cluster_size(start_cell):
        visited = {start_cell.id}
        queue = deque([start_cell])
        cluster_size = 0
        while queue:
            curr = queue.popleft()
            cluster_size += 1
            if cluster_size >= cells_per_group:
                return cluster_size
            for nbr in pack.grid.get_neighbouring_cells(curr.id):
                if nbr.id not in used_cells and nbr.id not in visited:
                    visited.add(nbr.id)
                    queue.append(nbr)
        return cluster_size

    # glavno sklapanje grupa
    for group_id in range(total_groups):
        current_group_cells = []
        seed_cell = None

        # biramo seed iz susjedstva prosle grupe
        if previous_group_cells:
            last_prev_cell = previous_group_cells[-1]
            candidates_prev = []

            for prev_cell in previous_group_cells:
                for nbr in pack.grid.get_neighbouring_cells(prev_cell.id):
                    if nbr.id not in used_cells:
                        candidates_prev.append(nbr)

            if candidates_prev:
                if strategy == "horizontal":
                    if direction == "bottom-up":
                        candidates_prev.sort(key=lambda c: (c.y, c.x))
                    else:
                        candidates_prev.sort(key=lambda c: (-c.y, c.x))

                elif strategy == "vertical":
                    candidates_prev = [
                        (abs(get_col_key(nbr) - get_col_key(last_prev_cell)) * 1000 + math.hypot(nbr.row - last_prev_cell.row, nbr.col - last_prev_cell.col), nbr)
                        for nbr in candidates_prev
                    ]
                    candidates_prev.sort(key=lambda x: x[0])
                    candidates_prev = [cand[1] for cand in candidates_prev]

                elif strategy == "diagonal":
                                    last_prev_diag = get_diag_index(last_prev_cell)
                                    candidates_prev.sort(key=lambda c: (
                                        abs(get_diag_index(c) - last_prev_diag), 
                                        math.hypot(c.row - last_prev_cell.row, c.col - last_prev_cell.col)
                                    ))

                for cand in candidates_prev:
                    if get_connected_free_cluster_size(cand) >= cells_per_group:

                        seed_cell = cand
                        break

        # ako nismo nasli seed medju susjedima, uzmi prvu slobodnu iz liste
        if not seed_cell:
            for cell in ordered_cells:
                if cell.id not in used_cells:
                    if get_connected_free_cluster_size(cell) >= cells_per_group:

                        seed_cell = cell
                        break

        if not seed_cell:
            return None

        current_group_cells.append(seed_cell)
        used_cells.add(seed_cell.id)
        sol.assign_cell_to_group(seed_cell.id, group_id)

        # siri grupu dok se ne napuni
        while len(current_group_cells) < cells_per_group:
            candidates = set()
            for cg in current_group_cells:
                for nbr in pack.grid.get_neighbouring_cells(cg.id):
                    if nbr.id not in used_cells:
                        candidates.add(nbr)

            if not candidates:
                break

            scored_candidates = []
            last_added = current_group_cells[-1]

            # racunanje bodova za svaku kandidatsku celiju
            for cand in candidates:
                if strategy == "horizontal":
                    if previous_group_cells:
                        min_dist_to_prev = min(math.hypot(cand.x - p.x, cand.y - p.y) for p in previous_group_cells)
                        total_dist_to_prev = sum(math.hypot(cand.x - p.x, cand.y - p.y) for p in previous_group_cells)

                        score = min_dist_to_prev * 100 + total_dist_to_prev
                    else:
                        score = (cand.y * 10 + cand.x) if direction == "bottom-up" else (-cand.y * 10 + cand.x)


                elif strategy == "vertical":
                    if group_id == 0:
                        g0_seed_col = get_col_key(current_group_cells[0])
                        col_distance = abs(get_col_key(cand) - g0_seed_col)
                        row_distance = abs(cand.row - last_added.row)
                        score = (col_distance * 1_000_000) + row_distance
                    else:
                        min_dist_to_prev = min(math.hypot(cand.x - p.x, cand.y - p.y) for p in previous_group_cells)
                        total_dist_to_prev = sum(math.hypot(cand.x - p.x, cand.y - p.y) for p in previous_group_cells)
                        dist_to_last = math.hypot(cand.x - last_added.x, cand.y - last_added.y)
                        score = (min_dist_to_prev * 100) + total_dist_to_prev + (dist_to_last * 10)


                elif strategy == "diagonal":
                    if group_id == 0:
                        diag_diff = abs(get_diag_index(cand) - get_diag_index(last_added))
                        same_diag_penalty = 0 if diag_diff == 0 else 1000
                        dist_along_line = math.hypot(cand.row - last_added.row, cand.col - last_added.col)
                        score = same_diag_penalty + dist_along_line
                    else:
                        min_dist_curr = min(math.hypot(cand.x - cg.x, cand.y - cg.y) for cg in current_group_cells)
                        min_dist_prev = min(math.hypot(cand.x - pr.x, cand.y - pr.y) for pr in previous_group_cells)
                        score = min_dist_curr * 4 + min_dist_prev


                scored_candidates.append((score, cand))

            scored_candidates.sort(key=lambda x: x[0])

            # malo stohastike s temperaturom ako treba
            if temperature > 0 and len(scored_candidates) > 1 and random.random() < p_stochastic:

                best_score = scored_candidates[0][0]
                temp_multiplier = 2.0 if strategy == "diagonal" else 10.0

                good_candidates = [
                    cand for score, cand in scored_candidates
                    if score <= best_score + (temperature * temp_multiplier)
                ]
                selected_candidate = random.choice(good_candidates)
            else:
                selected_candidate = scored_candidates[0][1]

            current_group_cells.append(selected_candidate)
            used_cells.add(selected_candidate.id)
            sol.assign_cell_to_group(selected_candidate.id, group_id)

        if len(current_group_cells) < cells_per_group:
            return None

        previous_group_cells = current_group_cells

    # ostale neiskoristene postavi kao praznine
    for cell in pack.grid.cells:
        if cell is not None and cell.id not in used_cells:
            sol.assign_cell_to_group(cell.id, void_id)

    return sol


def max_shared_boundary_distance_based_universal_smart(pack, void_id=-1, temperature=0, p_stochastic=0.2, strategy="horizontal", direction="top-down", diagonal_type="backslash",reverse_columns=False):
    """Napredna verzija univerzalnog algoritma koja pazi na bms celije i sprjecava
    fragmentaciju slobodnog prostora.

    Argumenti:
        pack (BatteryPack): paket baterija sa definiranim bms celijama.
        void_id (int, optional): id za neiskoristene celije. default: -1.
        temperature (float, optional): parametar za stohasticki odabir. default:
          0.
        p_stochastic (float, optional): vjerojatnost stohastickog odabira.
          default: 0.2.
        strategy (str, optional): strategija slaganja ("horizontal", "vertical",
          "diagonal"). default: "horizontal".
        direction (str, optional): smjer u strategiji ("top-down", "bottom-up",
          "left-to-right", "right-to-left"). default: "top-down".
        diagonal_type (str, optional): vrst dijagonale ("backslash", "slash").
          default: "backslash".
        reverse_columns (bool, optional): okreni stupce za dijagonalu. default:
          False.

    Vraca:
        BatterySolution | None: rjesenje ili None ako nije moguce poslagati sve
        grupe.
    """
    if not pack.grid.cells:
        return None

    sol = BatterySolution(pack)
    cells_per_group = pack.no_cells_per_group
    total_groups = pack.no_groups

    used_cells = set()

    # bms celije i unaprijed rezervirane stavljamo u grupu -2
    for cell in pack.grid.cells:
        if cell.id in pack.bms_cell_ids or getattr(cell, "group", None) == -2:
            used_cells.add(cell.id)
            sol.assign_cell_to_group(cell.id, -2)

    previous_group_cells = []
    is_hex = type(pack.grid).__name__ == "HexGrid"

    def get_col_key(c):
        return (2 * c.col + c.row) if is_hex else c.col

    def get_diag_index(c):
        if diagonal_type == "slash":
            idx = c.row + c.col
        else:
            idx = c.col
        return -idx if reverse_columns else idx

    # sortiranje celija (bms celije preskacemo)
    if strategy == "horizontal":
        if direction == "bottom-up":
            
            ordered_cells = sorted([c for c in pack.grid.cells if c.id not in pack.bms_cell_ids], key=lambda c: (c.y, c.x))
        else:
            ordered_cells = sorted([c for c in pack.grid.cells if c.id not in pack.bms_cell_ids], key=lambda c: (-c.y, c.x))
            
    elif strategy == "vertical":
        col_dir = 1 if direction == "left-to-right" else -1
        ordered_cells = sorted(
            [c for c in pack.grid.cells if c is not None and c.id not in pack.bms_cell_ids],
            key=lambda c: (col_dir * get_col_key(c), c.row)
        )
        
    elif strategy == "diagonal":
        if diagonal_type == "slash":
            ordered_cells = sorted([c for c in pack.grid.cells if c.id not in pack.bms_cell_ids], key=lambda c: (get_diag_index(c), -c.row if not reverse_columns else c.row))
        else:
            ordered_cells = sorted([c for c in pack.grid.cells if c.id not in pack.bms_cell_ids], key=lambda c: (get_diag_index(c), c.row))


    # provjere velicine klastera da ne ostavimo nepovezane "rupe"
    def get_connected_free_cluster_size(start_cell, cells_left):
        neighbors = pack.grid.get_neighbouring_cells(start_cell.id)
        visited = {start_cell.id}
        max_cluster_size = 0
        n = neighbors.pop() if neighbors else None
        while n is not None:
            if (n.id not in visited) and (n.id not in used_cells):
                visited.add(n.id)
                queue = deque(pack.grid.get_neighbouring_cells(n.id))
                cluster_size = 1
                while queue:
                    curr = queue.popleft()
                    if curr.id in visited or curr.id in used_cells:
                        continue
                    visited.add(curr.id)
                    cluster_size += 1
                    if cluster_size >= cells_left:
                        return cluster_size
                    for nbr in pack.grid.get_neighbouring_cells(curr.id):
                        if nbr.id not in used_cells and nbr.id not in visited:
                            queue.append(nbr)
                if cluster_size > max_cluster_size:
                    max_cluster_size = cluster_size
            n = neighbors.pop() if neighbors else None

        return max_cluster_size

    def get_connected_free_cluster_size2(candidate, neighbors, cells_left):
        visited = {candidate.id}
        max_cluster_size = 0
        n = neighbors.pop() if neighbors else None
        while n is not None:
            if (n.id not in visited) and (n.id not in used_cells):
                visited.add(n.id)
                queue = deque(pack.grid.get_neighbouring_cells(n.id))
                cluster_size = 1
                while queue:
                    curr = queue.popleft()
                    if curr.id in visited or curr.id in used_cells:
                        continue
                    visited.add(curr.id)
                    cluster_size += 1
                    if cluster_size >= cells_left:
                        return cluster_size
                    for nbr in pack.grid.get_neighbouring_cells(curr.id):
                        if nbr.id not in used_cells and nbr.id not in visited:
                            queue.append(nbr)
                if cluster_size > max_cluster_size:
                    max_cluster_size = cluster_size
            n = neighbors.pop() if neighbors else None

        return max_cluster_size

    # idemo grupu po grupu
    for group_id in range(total_groups):
        current_group_cells = []
        seed_cell = None
        # preostali broj celija za sve buduce grupe
        cells_left = cells_per_group * (total_groups - group_id - 1)

        # trazenje seed celije
        if previous_group_cells:
            last_prev_cell = previous_group_cells[-1]
            candidates_prev = []

            for prev_cell in previous_group_cells:
                for nbr in pack.grid.get_neighbouring_cells(prev_cell.id):
                    if nbr.id not in used_cells:
                        candidates_prev.append(nbr)

            if candidates_prev:
                if strategy == "horizontal":
                    if direction == "bottom-up":
                        candidates_prev.sort(key=lambda c: (c.y, c.x))
                    else:
                        candidates_prev.sort(key=lambda c: (-c.y, c.x))

                elif strategy == "vertical":
                    candidates_prev = [
                        (
                            abs(get_col_key(nbr) - get_col_key(last_prev_cell))
                            * 1000
                            + math.hypot(
                                nbr.row - last_prev_cell.row,
                                nbr.col - last_prev_cell.col,
                            ),
                            nbr,
                        )
                        for nbr in candidates_prev
                    ]
                    candidates_prev.sort(key=lambda x: x[0])
                    candidates_prev = [cand[1] for cand in candidates_prev]

                elif strategy == "diagonal":
                    last_prev_diag = get_diag_index(last_prev_cell)
                    candidates_prev.sort(
                        key=lambda c: (
                            abs(get_diag_index(c) - last_prev_diag),
                            math.hypot(
                                c.row - last_prev_cell.row,
                                c.col - last_prev_cell.col,
                            ),
                        )
                    )

                for cand in candidates_prev:
                    if get_connected_free_cluster_size(
                        cand, cells_left
                    ) >= cells_left and get_connected_free_cluster_size(
                        cand, cells_per_group - 1
                    ) >= (
                        cells_per_group - 1
                    ):
                        seed_cell = cand
                        break

        # pricuva ako nismo nasli seed medju susjedima
        if not seed_cell:
            for cell in ordered_cells:
                if cell.id not in used_cells:
                    if get_connected_free_cluster_size(
                        cell, cells_left
                    ) >= cells_left and get_connected_free_cluster_size(
                        cell, cells_per_group - 1
                    ) >= (
                        cells_per_group - 1
                    ):
                        seed_cell = cell
                        break

        if not seed_cell:
            return None

        current_group_cells.append(seed_cell)
        used_cells.add(seed_cell.id)
        sol.assign_cell_to_group(seed_cell.id, group_id)

        # punjenje trenutne grupe celijama
        while len(current_group_cells) < cells_per_group:
            candidates = set()
            for cg in current_group_cells:
                for nbr in pack.grid.get_neighbouring_cells(cg.id):
                    if nbr.id not in used_cells:
                        candidates.add(nbr)

            if not candidates:
                break

            scored_candidates = []
            last_added = current_group_cells[-1]

            # izracunajte score za kandidate
            for cand in candidates:
                if strategy == "horizontal":
                    if previous_group_cells:
                        min_dist_to_prev = min(
                            math.hypot(cand.x - p.x, cand.y - p.y)
                            for p in previous_group_cells
                        )
                        total_dist_to_prev = sum(
                            math.hypot(cand.x - p.x, cand.y - p.y)
                            for p in previous_group_cells
                        )
                        score = min_dist_to_prev * 100 + total_dist_to_prev
                    else:
                        score = (
                            (cand.y * 10 + cand.x)
                            if direction == "bottom-up"
                            else (-cand.y * 10 + cand.x)
                        )

                elif strategy == "vertical":
                    if group_id == 0:
                        g0_seed_col = get_col_key(current_group_cells[0])
                        col_distance = abs(get_col_key(cand) - g0_seed_col)
                        row_distance = abs(cand.row - last_added.row)
                        score = (col_distance * 1_000_000) + row_distance
                    else:
                        min_dist_to_prev = min(
                            math.hypot(cand.x - p.x, cand.y - p.y)
                            for p in previous_group_cells
                        )
                        total_dist_to_prev = sum(
                            math.hypot(cand.x - p.x, cand.y - p.y)
                            for p in previous_group_cells
                        )
                        dist_to_last = math.hypot(
                            cand.x - last_added.x, cand.y - last_added.y
                        )
                        score = (
                            (min_dist_to_prev * 100)
                            + total_dist_to_prev
                            + (dist_to_last * 10)
                        )

                elif strategy == "diagonal":
                    if group_id == 0:
                        diag_diff = abs(
                            get_diag_index(cand) - get_diag_index(last_added)
                        )
                        same_diag_penalty = 0 if diag_diff == 0 else 1000
                        dist_along_line = math.hypot(
                            cand.row - last_added.row, cand.col - last_added.col
                        )
                        score = same_diag_penalty + dist_along_line
                    else:
                        min_dist_curr = min(
                            math.hypot(cand.x - cg.x, cand.y - cg.y)
                            for cg in current_group_cells
                        )
                        min_dist_prev = min(
                            math.hypot(cand.x - pr.x, cand.y - pr.y)
                            for pr in previous_group_cells
                        )
                        score = min_dist_curr * 4 + min_dist_prev

                scored_candidates.append((score, cand))

            scored_candidates.sort(key=lambda x: x[0])
            valid_candidate = False

            # provjera ostavlja li kandidat dovoljno prohodnog prostora
            while not valid_candidate and scored_candidates:
                if (
                    temperature > 0
                    and len(scored_candidates) > 1
                    and random.random() < p_stochastic
                ):
                    best_score = scored_candidates[0][0]
                    temp_multiplier = 2.0 if strategy == "diagonal" else 10.0

                    good_candidates = [
                        cand
                        for score, cand in scored_candidates
                        if score <= best_score + (temperature * temp_multiplier)
                    ]
                    selected_candidate = random.choice(good_candidates)
                else:
                    selected_candidate = scored_candidates[0][1]

                group_nbrs = candidates.copy()
                group_nbrs.remove(selected_candidate)
                for nbr in pack.grid.get_neighbouring_cells(
                    selected_candidate.id
                ):
                    if nbr.id not in used_cells:
                        group_nbrs.add(nbr)

                cluster_Size = get_connected_free_cluster_size2(
                    selected_candidate, group_nbrs, cells_left
                )
                if cluster_Size >= cells_left:
                    valid_candidate = True

                # ako kandidat zablokira prostor, odbacujemo ga
                scored_candidates = [item for item in scored_candidates if item[1] != selected_candidate]

            current_group_cells.append(selected_candidate)
            used_cells.add(selected_candidate.id)
            sol.assign_cell_to_group(selected_candidate.id, group_id)

        if len(current_group_cells) < cells_per_group:
            return None

        previous_group_cells = current_group_cells

    # nepovezane / neiskoristene stavljamo u void
    for cell in pack.grid.cells:
        if cell is not None and cell.id not in used_cells:
            sol.assign_cell_to_group(cell.id, void_id)

    return sol


def check_simple_solution(pack):
    """Brzi algoritam za pravokutne konfiguracije gdje grupa stane u jedan red
    ili stupac.

    Argumenti:
        pack (BatteryPack): paket baterija.

    Vraca:
        BatterySolution | None: brzo izgenerirano rjesenje ili None ako stane
        jednostavno.
    """
    # provjeravamo stane li svaka grupa tocno u svoj red
    if (pack.grid.rows >= pack.no_groups) and (
        pack.grid.cells_per_row_1 >= pack.no_cells_per_group
        and pack.grid.cells_per_row_2 >= pack.no_cells_per_group
    ):
        sol = BatterySolution(pack)
        cells_per_group = pack.no_cells_per_group

        for group_id in range(pack.no_groups):
            row_cells = [
                cell
                for cell in pack.grid.cell_matrix[group_id]
                if cell is not None
            ]
            selected_cells = row_cells[:cells_per_group]

            for cell in selected_cells:
                sol.assign_cell_to_group(cell.id, group_id)
        return sol

    # provjeravamo stane li svaka grupa u svoj stupac
    elif (pack.grid.cells_per_row_1 >= pack.no_groups) and (
        pack.grid.rows >= pack.no_cells_per_group
    ):
        sol = BatterySolution(pack)
        cells_per_group = pack.no_cells_per_group

        for group_id in range(pack.no_groups):
            col_cells = [
                pack.grid.cell_matrix[r][group_id]
                for r in range(pack.grid.rows)
                if pack.grid.cell_matrix[r][group_id] is not None
            ]
            selected_cells = col_cells[:cells_per_group]

            for cell in selected_cells:
                sol.assign_cell_to_group(cell.id, group_id)
        return sol

    return None
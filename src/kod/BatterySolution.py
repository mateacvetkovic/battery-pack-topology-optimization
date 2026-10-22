from src.kod.visuals import plotGroups, plotBridges

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

class BatterySolution:
    """
    razred koji predstavlja rjesenje rasporeda celija unutar baterijskog paketa.
    sadrzi metode za provjeru valjanosti, racunanje fitnessa i analizu mostova.
    """
    def __init__(self, pack, id=None):
        """
        inicijalizacija rjesenja baterijskog paketa.

        argumenti:
            pack: objekt baterijskog paketa koji sadrzi mrezu i geometriju
            id: identifikator rjesenja
        """
        self.solution_id = id
        self.pack = pack
        self.cell_to_group = [-1] * self.pack.grid.total_cells
        self.group_dict = {i: set() for i in range(self.pack.no_groups)}
        self._fitness = None
        self._eval_info = None
        self.mark_bms_cells(self.pack.bms_cell_ids)

    @property
    def fitness(self):
        """
        dohvaca fitness rjesenja. izracunava ga ako vec nije spremljen.

        povratna vrijednost:
            float: vrijednost fitnessa rjesenja
        """
        if self._fitness is None:
            self._fitness = self.calculate_total_fitness()
        return self._fitness

    @fitness.setter
    def fitness(self, value):
        self._fitness = value

    @property
    def eval_info(self):
        """
        dohvaca informacije o evaluaciji rjesenja.

        povratna vrijednost:
            dict: rjecnik s detaljima evaluacije
        """
        if self._eval_info is None:
            self._eval_info = self.evaluate_solution()
        return self._eval_info

    @eval_info.setter
    def eval_info(self, value):
        self._eval_info = value

    def is_group_connected(self, group_id):
        """
        provjerava jesu li sve celije u zdanoj grupi međusobno povezane.

        argumenti:
            group_id (int): identifikator grupe koja se provjerava

        povratna vrijednost:
            bool: true ako je grupa povezana, false inace
        """
        cells_in_group = self.group_dict.get(group_id, set())
        if not cells_in_group:
            return True

        # IZMJENA: isti DFS kao u originalu, samo izvucen u pomocnu funkciju da se moze
        # primijeniti i na uniju dviju grupa, a ne samo na jednu grupu
        def connected(cells):
            remaining = set(cells)
            stack = [remaining.pop()]
            while stack:
                current_id = stack.pop()
                for neighbour_id in self.pack.get_neighbours_ids(current_id):
                    if neighbour_id in remaining:
                        remaining.remove(neighbour_id)
                        stack.append(neighbour_id)
            return not remaining

        # IZMJENA: prva i zadnja grupa ostaju kompaktne jer im je druga strana izvod paketa
        # if group_id in (0, self.pack.no_groups - 1) and not connected(cells_in_group):
        #     return False

        # IZMJENA: traka izmedju susjednih grupa je jedan komad metala koji mora spojiti SVE celije
        # obje grupe, pa unija grupa g i g+1 (isto tako g-1 i g) mora biti povezana.
        # Sama grupa NE mora biti kompaktna. Za kompaktne grupe ovo je isto sto je is_valid
        # vec provjeravao (grupe se dodiruju), pa se postojece ponasanje ne mijenja.
        for other in (group_id - 1, group_id + 1):
            if 0 <= other < self.pack.no_groups and self.group_dict.get(other):
                if not connected(cells_in_group | self.group_dict[other]):
                    return False
                # IZMJENA: dvije susjedne grupe ne smiju istovremeno biti razdvojene. Time se
                # izbjegavaju slozene trake i raspad trake izmedju dvije razdvojene grupe.
                # if not connected(cells_in_group) and not connected(self.group_dict[other]):
                #     return False
                # NOVO: odvojeni dijelovi grupe moraju imati dovoljno razlicitih spojeva prema susjednoj
                # grupi, u OBA smjera (vidi _pieces_have_own_bridges). Ovdje se provjerava jer mutacije
                # pozivaju upravo ovu funkciju, pa nevaljana particija nikad ne nastane.
                if not self._pieces_have_own_bridges(group_id, other) or not self._pieces_have_own_bridges(other, group_id):
                    return False
        return True

    # NOVO: dijeli skup celija na povezane komponente (kompaktne dijelove)
    def get_components(self, cells):
        remaining = set(cells)
        components = []
        while remaining:
            component = {remaining.pop()}
            stack = list(component)
            while stack:
                current_id = stack.pop()
                for neighbour_id in self.pack.get_neighbours_ids(current_id):
                    if neighbour_id in remaining:
                        remaining.remove(neighbour_id)
                        component.add(neighbour_id)
                        stack.append(neighbour_id)
            components.append(component)
        return components

    # NOVO: pravilo za razdvojene grupe. Ako je grupa g razbijena na vise kompaktnih dijelova, SVAKI dio
    # (i najveci) od k celija mora imati k RAZLICITIH spojeva prema grupi target (maksimalno uparivanje
    # pokriva svaku celiju dijela; svaka celija s obje strane koristi se najvise jednom). Dio od 3
    # celija s manje od 3 spoja nije dopusten. Dijelovi se gledaju svaki za sebe: smiju dijeliti iste
    # celije iduce grupe (struja tada ide visestruko kroz isti most), a to se vec kaznjava u fitnessu
    # kroz manji broj jedinstvenih mostova. Nerazbijena grupa nema novo ogranicenje.
    def _pieces_have_own_bridges(self, g, target):
        cells = self.group_dict.get(g, set())
        target_cells = self.group_dict.get(target, set())
        if len(cells) < 2 or not target_cells:
            return True
        components = self.get_components(cells)
        if len(components) < 2:
            return True
        for component in components:
            cells_c = list(component)
            t_index = {}
            rows, cols = [], []
            for i, c in enumerate(cells_c):
                for n in self.pack.get_neighbours_ids(c):
                    if n in target_cells:
                        rows.append(i)
                        cols.append(t_index.setdefault(n, len(t_index)))
            if len(t_index) < len(cells_c):
                return False
            graph = csr_matrix((np.ones(len(rows), dtype=int), (rows, cols)), shape=(len(cells_c), len(t_index)))
            if int(np.sum(maximum_bipartite_matching(graph, perm_type='column') != -1)) < len(cells_c):
                return False
        return True

    def count_components(self, group_id):
        """
        IZMJENA: vraca broj povezanih komponenti grupe (1 = kompaktna grupa). Koristi se za kaznu
        u fitnessu.
        """
        remaining = set(self.group_dict.get(group_id, set()))
        count = 0
        while remaining:
            count += 1
            stack = [remaining.pop()]
            while stack:
                current_id = stack.pop()
                for neighbour_id in self.pack.get_neighbours_ids(current_id):
                    if neighbour_id in remaining:
                        remaining.remove(neighbour_id)
                        stack.append(neighbour_id)
        return count

    def is_valid(self):
        """
        provjerava cjelokupnu valjanost rjesenja (broj grupa, broj celija po grupi, povezanost i dodirivanje uzastopnih grupa).

        povratna vrijednost:
            bool: true ako je rjesenje valjano, false inace
        """
        valid_groups = [g for g in self.group_dict.keys() if g != -1]
        if len(valid_groups) != self.pack.no_groups:
            return False
            
        for group in valid_groups:
            if len(self.group_dict[group]) != self.pack.no_cells_per_group:
                return False
            if not self.is_group_connected(group):
                return False
                
        sorted_groups = sorted(valid_groups)
        for i in range(len(sorted_groups) - 1):
            g1 = sorted_groups[i]
            g2 = sorted_groups[i + 1]
            
            cells_g2 = self.group_dict[g2]
            touching = False
            
            for cell_id in self.group_dict[g1]:
                neighbors = self.pack.grid.get_neighbouring_cells(cell_id)
                if any(nbr.id in cells_g2 for nbr in neighbors):
                    touching = True
                    break
                    
            if not touching:
                return False
                
        return True


    def assign_cell_to_group(self, cell_id, group_id):
        """
        dodjeljuje celiju odredjenoj grupi i azurira unutarnje strukture.

        argumenti:
            cell_id (int): identifikator celije
            group_id (int): identifikator grupe
        """
        self._fitness = None
        self.eval_info = None
        self.pack.grid.cells[cell_id].group = group_id
        if cell_id in self.pack.bms_cell_ids:
            return
        
        old_group = self.cell_to_group[cell_id]

        if old_group != -1 and old_group != -2:
            self.group_dict[old_group].discard(cell_id)
        if old_group != -2:
            self.cell_to_group[cell_id] = group_id
        if group_id != -1 and group_id != -2:
            self.group_dict[group_id].add(cell_id)

    def display(self, save=False, show = True, name=None):
        """
        prikazuje grupe rjesenja na mrezi.

        argumenti:
            save (bool): zastavica za spremanje slike
            name (str): naziv izlazne datoteke
        """
        if(self.pack.no_groups is not None and self.pack.no_cells_per_group is not None):
            title = f'Konfiguracija: {self.pack.no_groups}S {self.pack.no_cells_per_group}P'
        plotGroups(self.pack.grid.cells, self.pack.grid.W, self.pack.grid.H, self.pack.d, self.cell_to_group, self.pack.no_groups, title=title,cell_d=self.pack.cell_d, vertices=self.pack.vertices, save=save, show=show, name=name)

    def displayBridges(self, save=False, name=None):
        """
        prikazuje mostove izmedju grupa na mrezi.

        argumenti:
            save (bool): zastavica za spremanje slike
            name (str): naziv izlazne datoteke
        """
        eval_res = self.evaluate_solution()
        plotBridges(self.pack.grid.cells, self.pack.grid.W, self.pack.grid.H, self.pack.d, self.cell_to_group, self.pack.no_groups, vertices=self.pack.vertices, save=save, name=name, bridges=eval_res['bridges'])

    def copy(self):
        """
        stvara duboku kopiju trenutnog objekta rjesenja.

        povratna vrijednost:
            BatterySolution: novi kopirani objekt rjesenja
        """
        copy_obj = BatterySolution(self.pack, self.solution_id)
        copy_obj.cell_to_group = self.cell_to_group[:]
        copy_obj.group_dict = {g_id: cell_ids.copy() for g_id, cell_ids in self.group_dict.items()}
        copy_obj._fitness = None
        copy_obj.eval_info = None
        return copy_obj

    def count_effective_serial_bridges(self, g1_id, g2_id):
        """
        racuna efektivni broj serijskih mostova izmedju dvije grupe.
        broji koliko JEDINSTVENIH celija iz G1 ima kontakt s G2, i koliko JEDINSTVENIH celija iz G2 ima kontakt s G1, te uzima manji broj jer jedna celija ne moze pružiti vise veza
        argumenti:
            g1_id (int): identifikator prve grupe
            g2_id (int): identifikator druge grupe

        povratna vrijednost:
            int: minimalni broj jedinstvenih spojnih celija iz obje grupe
        """
        cells_g1 = self.group_dict.get(g1_id, set())
        cells_g2 = self.group_dict.get(g2_id, set())

        if not cells_g1 or not cells_g2:
            return 0

        g1_connecting_cells = set()
        g2_connecting_cells = set()

        for cell_1 in cells_g1:
            neighbors = self.pack.get_neighbours_ids(cell_1)
            for cell_2 in neighbors:
                if cell_2 in cells_g2:
                    g1_connecting_cells.add(cell_1)
                    g2_connecting_cells.add(cell_2)

        return min(len(g1_connecting_cells), len(g2_connecting_cells))

    def count_independent_serial_bridges(self, g1_id, g2_id):
        """
        racuna broj neovisnih serijskih mostova koristeci maksimalno bipartitno uparivanje.

        argumenti:
            g1_id (int): identifikator prve grupe
            g2_id (int): identifikator druge grupe

        povratna vrijednost:
            int: broj neovisnih mostova
        """
        cells_g1 = list(self.group_dict.get(g1_id, set()))
        cells_g2 = list(self.group_dict.get(g2_id, set()))

        if not cells_g1 or not cells_g2:
            return 0

        graph_matrix = np.zeros((len(cells_g1), len(cells_g2)), dtype=int)

        for i, cell_1_id in enumerate(cells_g1):
            neighbors = self.pack.get_neighbours_ids(cell_1_id)
            for j, cell_2_id in enumerate(cells_g2):
                if cell_2_id in neighbors:
                    graph_matrix[i, j] = 1

        bi_graph = csr_matrix(graph_matrix)
        matching = maximum_bipartite_matching(bi_graph, perm_type='column')
        num_independent_bridges = np.sum(matching != -1)

        return num_independent_bridges

    def calculate_bridge_distance_metrics(self):
        """
        racuna prosjecnu udaljenost i varijancu udaljenosti do serijskih mostova za sve grupe.

        povratna vrijednost:
            tuple: (max_variance, total_avg_distance, group_stats)
        """
        group_stats = {}
        all_variances = []
        all_distances = []

        for group_id in range(self.pack.no_groups):
            distances = self._get_group_bridge_distances(group_id)
            
            if distances and float('inf') not in distances:
                mean_dist = np.mean(distances)
                var_dist = np.var(distances)
            else:
                mean_dist = float('inf')
                var_dist = float('inf')

            group_stats[group_id] = {
                'distances': distances,
                'mean_distance': mean_dist,
                'variance': var_dist
            }
            
            if var_dist != float('inf'):
                all_variances.append(var_dist)
                all_distances.extend(distances)

        total_avg_distance = np.mean(all_distances) if all_distances else float('inf')
        max_variance = max(all_variances) if all_variances else float('inf')

        return max_variance, total_avg_distance, group_stats

    def _get_group_bridge_distances(self, group_id):
        """
        dohvaca udaljenosti celija u grupi do najblizeg mosta.

        argumenti:
            group_id (int): identifikator grupe

        povratna vrijednost:
            list: lista udaljenosti za svaku celiju u grupi
        """
        cells_in_group = list(self.group_dict.get(group_id, set()))
        if not cells_in_group:
            return []

        if group_id == self.pack.no_groups - 1:
            target_group = group_id - 1
        else:
            target_group = group_id + 1

        if target_group < 0 or target_group >= self.pack.no_groups:
            return [float('inf')] * len(cells_in_group)

        g_x = [self.pack.grid.cells[c].x for c in cells_in_group]
        g_y = [self.pack.grid.cells[c].y for c in cells_in_group]
        center_x, center_y = sum(g_x) / len(g_x), sum(g_y) / len(g_y)

        def dist_to_center(cell_id):
            c = self.pack.grid.cells[cell_id]
            return (c.x - center_x)**2 + (c.y - center_y)**2

        candidates = []
        for cell_1 in cells_in_group:
            neighbors = self.pack.get_neighbours_ids(cell_1)
            for cell_2 in neighbors:
                if self.cell_to_group[cell_2] == target_group:
                    candidates.append((cell_1, cell_2, dist_to_center(cell_1)))

        candidates.sort(key=lambda x: x[2])

        actual_bridge_cells = set()
        used_target_cells = set()

        for cell_1, cell_2, _ in candidates:
            if cell_1 not in actual_bridge_cells and cell_2 not in used_target_cells:
                actual_bridge_cells.add(cell_1)
                used_target_cells.add(cell_2)

        if not actual_bridge_cells:
            return [float('inf')] * len(cells_in_group)

        distances = {c_id: float('inf') for c_id in cells_in_group}
        queue = []

        for b_id in actual_bridge_cells:
            distances[b_id] = 0
            queue.append(b_id)

        while queue:
            current_id = queue.pop(0)
            current_dist = distances[current_id]

            for neighbor_id in self.pack.get_neighbours_ids(current_id):
                if neighbor_id in cells_in_group:
                    if distances[neighbor_id] > current_dist + 1:
                        distances[neighbor_id] = current_dist + 1
                        queue.append(neighbor_id)

        return list(distances.values())

    def _run_bfs(self, cells_in_group, bridge_sources):
        """
        pokrece pretragu u sirinu (bfs) za izracun udaljenosti celija od izvornih mostova.

        argumenti:
            cells_in_group (list): lista celija u grupi
            bridge_sources (set): skup celija koje sluze kao izvori (mostovi)

        povratna vrijednost:
            dict: rjecnik s udaljenostima celija od izvora
        """
        distances = {c_id: float('inf') for c_id in cells_in_group}
        queue = []

        for b_id in bridge_sources:
            if b_id in distances:
                distances[b_id] = 0
                queue.append(b_id)

        while queue:
            current_id = queue.pop(0)
            current_dist = distances[current_id]

            for neighbor_id in self.pack.get_neighbours_ids(current_id):
                if neighbor_id in distances:
                    if distances[neighbor_id] > current_dist + 1:
                        distances[neighbor_id] = current_dist + 1
                        queue.append(neighbor_id)

        # IZMJENA: celije koje BFS nije dosegao su odvojene komponente grupe. Bez ovoga bi ostale
        # inf, np.var bi dao nan i fitness bi se pokvario. Dobivaju prosjek udaljenosti dosegnutih
        # celija, pa odvajanje ne snizava prosjecnu udaljenost i fitness ga time ne nagradjuje.
        reached = [dist for dist in distances.values() if dist != float('inf')]
        neutral = float(np.mean(reached)) if reached else 0
        for c_id in distances:
            if distances[c_id] == float('inf'):
                distances[c_id] = neutral

        return distances

    def find_optimal_bridges_and_score(self, group_id, target_group, w_mean=1.0, w_var=2.0, w_load=1.5, verbose=True):
        """
        pronalazi optimalnu kombinaciju neovisnih mostova i ocjenjuje obje grupe.

        argumenti:
            group_id (int): identifikator prve grupe
            target_group (int): identifikator ciljane grupe
            w_mean (float): tezina prosjecne udaljenosti
            w_var (float): tezina varijance udaljenosti
            w_load (float): tezina opterecenja mosta
            verbose (bool): zastavica za ispis detalja

        povratna vrijednost:
            dict: detaljni rezultati o najboljoj kombinaciji mostova ili none ako nema parova
        """
        cells_g1 = list(self.group_dict.get(group_id, set()))
        cells_g2 = list(self.group_dict.get(target_group, set()))
        if not cells_g1 or not cells_g2:
            if verbose:
                print(f"[BRIDGES] Jedna od grupa je prazna: G{group_id} ili G{target_group}")
            return None

        possible_pairs = []
        for c1 in cells_g1:
            for c2 in self.pack.get_neighbours_ids(c1):
                if self.cell_to_group[c2] == target_group:
                    possible_pairs.append((c1, c2))

        if not possible_pairs:
            print('nema parova')
            return None

        valid_bridge_combinations = all_max_matchings(possible_pairs)

        best_score = -float('inf')
        best_result = None
        evaluated_candidates = []

        for comb in valid_bridge_combinations:
            bridge_sources_g1 = set(p[0] for p in comb)
            bridge_sources_g2 = set(p[1] for p in comb)
            
            distances_g1 = self._run_bfs(cells_g1, bridge_sources_g1)
            distances_g2 = self._run_bfs(cells_g2, bridge_sources_g2)
            
            dist_vals_g1 = list(distances_g1.values())
            dist_vals_g2 = list(distances_g2.values())
            
            mean_d1 = float(np.mean(dist_vals_g1))
            var_d1 = float(np.var(dist_vals_g1))
            
            mean_d2 = float(np.mean(dist_vals_g2))
            var_d2 = float(np.var(dist_vals_g2))
            
            total_mean_d = (mean_d1 + mean_d2) / 2.0
            total_var_d = max(var_d1, var_d2)

            # NOVO: "daleka" celija = vise od 1 skoka (unutar grupe) od celije koja ima most. Celija na 1 skoku
            # je ok, svaka na 2+ skoka kosta jednako (ravno, bez kvadrata). Zbraja se po svim parovima, ne
            # prosjecuje, da se jedan dugi lanac ne izgubi u prosjeku. Bez lanaca je vrijednost 0, pa se
            # kazna sama "iskljuci" kad lanaca nema (tipicno pravokutnik); djeluje na trokutu.
            far_cost = sum(1 for d in dist_vals_g1 if d >= 2) + sum(1 for d in dist_vals_g2 if d >= 2)
            
            num_bridges = len(comb)
            max_bridge_load = max(len(cells_g1) / num_bridges, len(cells_g2) / num_bridges)

            combo_score = -(w_mean * total_mean_d) - (w_var * total_var_d) - (w_load * max_bridge_load)

            candidate_info = {
                'combo_score': combo_score,
                'comb': comb,
                'num_bridges': num_bridges,
                'mean_d1': mean_d1,
                'mean_d2': mean_d2,
                'combined_mean_d': total_mean_d,
                'combined_max_var': total_var_d,
                'max_bridge_load': max_bridge_load,
            }
            evaluated_candidates.append(candidate_info)

            if verbose:
                comb_str = str(comb) if len(str(comb)) <= 28 else str(comb)[:25] + "..."
                print(f"{num_bridges:<12} | {comb_str:<30} | {total_mean_d:<8.2f} | {total_var_d:<8.2f} | {max_bridge_load:<8.2f}")

            if combo_score > best_score:
                best_score = combo_score
                best_result = {
                    'best_bridges': comb,
                    'bridge_cells_g1': list(bridge_sources_g1),
                    'bridge_cells_g2': list(bridge_sources_g2),
                    'num_bridges': num_bridges,
                    'cell_distances_g1': distances_g1,
                    'cell_distances_g2': distances_g2,
                    'mean_distance_g1': mean_d1,
                    'mean_distance_g2': mean_d2,
                    'combined_mean_distance': total_mean_d,
                    'combined_max_variance': total_var_d,
                    'max_bridge_load': max_bridge_load,
                    'bridge_score': combo_score,
                    'far_cost': far_cost
                }

        if verbose and best_result:
            print("Grupa", group_id)
            print(f" POBJEDNIČKA KOMBINACIJA: {best_result['best_bridges']}")
            print(f" No_bridges: {best_result['num_bridges']}, Score: {best_result['bridge_score']:.2f} | Mean Dist: {best_result['combined_mean_distance']:.2f} | Max Var: {best_result['combined_max_variance']:.2f} | Max Load: {best_result['max_bridge_load']:.2f}")
            print("="*80 + "\n")

        return best_result

    # NOVO: zadana tezina kazne za daleke celije. 0 = iskljuceno. IALG.run je postavlja na 2 SAMO ako konacno
    # rjesenje faze 1 ima celije na 2+ skoka od mosta (vidi ClonalgOptimizer.run).
    W_FAR = 0.0

    def calculate_total_fitness(self, w_score=2.0, w_min_bridges=5.0, w_total_bridges=1.0, w_bridge_balance=2.0, w_split=0, w_far=None):
        """
        racuna ukupnu ocjenu (fitness) cjelokupnog rasporeda grupa u bateriji.

        argumenti:
            w_score (float): tezina prosjecnog rezultata mostova
            w_min_bridges (float): tezina minimalnog broja mostova
            w_total_bridges (float): tezina ukupnog broja mostova
            w_bridge_balance (float): kazna za neujednacenost broja mostova
            w_split (float): kazna za svaku dodatnu komponentu (odvojenu cijelu) u grupama

        povratna vrijednost:
            float: izracunati ukupni fitness
        """
        if not self.is_valid():
            self.fitness = -float('inf')
            return self.fitness

        bridge_results = []
        valid_groups = sorted(g for g in self.group_dict.keys() if g != -1)

        for i in range(len(valid_groups) - 1):
            g1 = valid_groups[i]
            g2 = valid_groups[i + 1]

            bridge_result = self.find_optimal_bridges_and_score(group_id=g1, target_group=g2, w_mean=1.0, w_var=2.0, w_load=1.5, verbose=False)

            if bridge_result is None:
                self.fitness = -float('inf')
                return self.fitness

            bridge_results.append(bridge_result)

        if not bridge_results:
            self.fitness = -float('inf')
            return self.fitness

        bridge_counts = [r['num_bridges'] for r in bridge_results]
        bridge_scores = [r['bridge_score'] for r in bridge_results]

        min_bridges = min(bridge_counts)
        total_bridges = sum(bridge_counts)
        mean_bridges = np.mean(bridge_counts)
        bridge_variance = np.var(bridge_counts)
        mean_bridge_score = np.mean(bridge_scores)

        # IZMJENA: kazna za odvojene celije. Isplati se samo ako time naraste broj mostova
        # (jedan dodatni most vrijedi barem w_total_bridges = 1, a kazna je 0.5 po komponenti).
        extra_components = sum(self.count_components(g) - 1 for g in valid_groups)
        # NOVO: kazna za daleke celije (2+ skoka od mosta); aktivna samo kad je W_FAR > 0 (postavlja je IALG.run).
        far_total = sum(r['far_cost'] for r in bridge_results)
        self.far_total = far_total  # IALG.run po tome odlucuje treba li ukljuciti kaznu
        if w_far is None:
            w_far = self.W_FAR
        self.fitness = (w_min_bridges * min_bridges + w_total_bridges * total_bridges + w_score * mean_bridge_score - w_bridge_balance * bridge_variance - w_split * extra_components
                        - w_far * far_total)
        self.bridge_results = bridge_results

        self.fitness_details = {
            'bridge_counts': bridge_counts,
            'min_bridges': min_bridges,
            'total_bridges': total_bridges,
            'mean_bridges': mean_bridges,
            'bridge_variance': bridge_variance,
            'mean_bridge_score': mean_bridge_score,
            'fitness': self.fitness,
        }

        return self.fitness

    def evaluate_solution(self, w_mean=1.0, w_var=2.0, w_load=4, verbose=False):
        """
        evaluira rjesenje i vraca detaljne informacije o kvalitetama veza izmedju grupa.

        argumenti:
            w_mean (float): tezina prosjecne udaljenosti
            w_var (float): tezina varijance udaljenosti
            w_load (float): tezina opterecenja
            verbose (bool): zastavica za ispis

        povratna vrijednost:
            dict: rjecnik s detaljnim metrikama evaluacije
        """
        if not self.is_valid():
            return {
                'bridges': None,
                'total_score': -float('inf'),
                'group_scores': {},
                'connection_results': {},
                'worst_group': None,
                'worst_connection': None,
                'bridge_counts': [],
                'min_bridges': 0,
                'total_bridges': 0,
                'bridge_variance': 0.0
            }
        group_ids = sorted([g_id for g_id in self.group_dict.keys() if g_id != -1])
        bridges = set()
        total_score = 0.0
        
        group_sum = {g_id: 0.0 for g_id in group_ids}
        group_count = {g_id: 0 for g_id in group_ids}
        group_total_bridges = {g_id: 0 for g_id in group_ids}
        
        connection_results = {}
        bridge_counts = []

        for i in range(len(group_ids) - 1):
            g1 = group_ids[i]
            g2 = group_ids[i + 1]
            
            res = self.find_optimal_bridges_and_score(g1, g2, verbose=False)
            
            if res is not None:
                connection_results[(g1, g2)] = res
                score = res.get('bridge_score', 0.0)
                num_bridges = res.get('num_bridges', 0)
                
                bridges.update(res.get('best_bridges', []))
                total_score += score
                bridge_counts.append(num_bridges)
                
                group_sum[g1] += score
                group_count[g1] += 1
                group_sum[g2] += score
                group_count[g2] += 1

                group_total_bridges[g1] += num_bridges
                group_total_bridges[g2] += num_bridges

        group_scores = {}
        for g_id in group_ids:
            if group_count[g_id] > 0:
                group_scores[g_id] = group_sum[g_id] / group_count[g_id]
            else:
                group_scores[g_id] = 0.0

        min_bridges = min(bridge_counts) if bridge_counts else 0
        total_bridges = sum(bridge_counts) if bridge_counts else 0
        bridge_variance = float(np.var(bridge_counts)) if bridge_counts else 0.0

        worst_group = min(group_total_bridges, key=group_total_bridges.get) if group_total_bridges else None
        worst_connection = min(connection_results, key=lambda k: connection_results[k]['bridge_score']) if connection_results else None

        return {
            'total_score': total_score,        
            'group_scores': group_scores,      
            'connection_results': connection_results,
            'worst_group': worst_group,
            'worst_connection': worst_connection,
            'bridges': bridges,
            'bridge_counts': bridge_counts,
            'min_bridges': min_bridges,
            'total_bridges': total_bridges,
            'bridge_variance': bridge_variance
        }

    def __eq__(self, other):
        """
        provjerava jednakost dva rjesenja usporedbom njihovih otisaka.

        argumenti:
            other (object): drugi objekt za usporedbu

        povratna vrijednost:
            bool: true ako su rjesenja jednaka, false inace
        """
        if not isinstance(other, BatterySolution):
            return False
        return self.get_fingerprint() == other.get_fingerprint()

    def get_fingerprint(self):
        """
        generira jedinstveni otisak rjesenja u obliku sortiranog tuple-a.

        povratna vrijednost:
            tuple: sortirani tuple parova (cell_id, group_id)
        """
        if isinstance(self.cell_to_group, list):
            items = enumerate(self.cell_to_group)
        else:
            items = self.cell_to_group.items()
            
        return tuple(sorted(items))

    def mark_bms_cells(self, bms_cell_ids):
        """
        postavlja rezervirane bms celije u nepromjenjivu grupu -2.

        argumenti:
            bms_cell_ids (list): lista identifikatora bms celija
        """
        for cell_id in bms_cell_ids:
            self.assign_cell_to_group(cell_id, group_id=-2)

    def _get_valid_bridge_combinations(self, group_id, target_group):
        """
        vraca listu svih validnih kombinacija mostova izmedju dviju grupa.

        argumenti:
            group_id (int): identifikator prve grupe
            target_group (int): identifikator druge grupe

        povratna vrijednost:
            list: lista kombinacija najjacih uparivanja
        """
        cells_g1 = list(self.group_dict.get(group_id, set()))
        cells_g2 = list(self.group_dict.get(target_group, set()))
        if not cells_g1 or not cells_g2:
            return []

        possible_pairs = [(c1, c2) for c1 in cells_g1 for c2 in self.pack.get_neighbours_ids(c1) if self.cell_to_group[c2] == target_group]
        
        return all_max_matchings(possible_pairs) if possible_pairs else []

    def find_optimal_bridges_number(self, group_id, target_group, verbose=True):
        """
        vraca velicinu optimalne kombinacije mostova izmedju grupa.

        argumenti:
            group_id (int): identifikator prve grupe
            target_group (int): identifikator druge grupe
            verbose (bool): zastavica za ispis

        povratna vrijednost:
            int: broj mostova u optimalnoj kombinaciji
        """
        valid_bridge_combinations = self._get_valid_bridge_combinations(group_id, target_group)
        if not valid_bridge_combinations:
            if verbose:
                print(f"[BRIDGES] Nema mogućih mostova između G{group_id} i G{target_group}")
            return 0
        return len(valid_bridge_combinations[0])

    def find_optimal_bridges(self, group_id, target_group, w_mean=1.0, w_var=2.0, w_load=1.5, verbose=True):
        """
        izvrsava punu optimizaciju i detaljnu evaluaciju mostova izmedju dvije grupe.

        argumenti:
            group_id (int): identifikator prve grupe
            target_group (int): identifikator druge grupe
            w_mean (float): tezina prosjecne udaljenosti
            w_var (float): tezina varijance udaljenosti
            w_load (float): tezina opterecenja
            verbose (bool): zastavica za detaljan ispis

        povratna vrijednost:
            dict: rjecnik s podacima o najboljoj kombinaciji
        """
        cells_g1 = list(self.group_dict.get(group_id, set()))
        cells_g2 = list(self.group_dict.get(target_group, set()))
        
        valid_bridge_combinations = self._get_valid_bridge_combinations(group_id, target_group)
        if not valid_bridge_combinations:
            if verbose:
                print(f"[BRIDGES] Nema mogućih mostova između G{group_id} i G{target_group}")
            return None

        best_score = -float('inf')
        best_result = None

        if verbose:
            print(f"\nEvaluacija mostova između G{group_id} i G{target_group}:")
            print(f"{'Br. mostova':<12} | {'Kombinacija':<30} | {'Mean D':<8} | {'Var D':<8} | {'Load':<8}")
            print("-" * 75)

        for comb in valid_bridge_combinations:
            bridge_g1, bridge_g2 = set(p[0] for p in comb), set(p[1] for p in comb)
            
            d1, d2 = self._run_bfs(cells_g1, bridge_g1), self._run_bfs(cells_g2, bridge_g2)
            vals_d1, vals_d2 = list(d1.values()), list(d2.values())
            
            mean_d1, var_d1 = float(np.mean(vals_d1)), float(np.var(vals_d1))
            mean_d2, var_d2 = float(np.mean(vals_d2)), float(np.var(vals_d2))
            
            total_mean_d = (mean_d1 + mean_d2) / 2.0
            total_var_d = max(var_d1, var_d2)
            num_bridges = len(comb)
            max_bridge_load = max(len(cells_g1), len(cells_g2)) / num_bridges

            combo_score = -(w_mean * total_mean_d) - (w_var * total_var_d) - (w_load * max_bridge_load)

            if verbose:
                comb_str = str(comb) if len(str(comb)) <= 28 else str(comb)[:25] + "..."
                print(f"{num_bridges:<12} | {comb_str:<30} | {total_mean_d:<8.2f} | {total_var_d:<8.2f} | {max_bridge_load:<8.2f}")

            if combo_score > best_score:
                best_score = combo_score
                best_result = {
                    'best_bridges': comb,
                    'bridge_cells_g1': list(bridge_g1),
                    'bridge_cells_g2': list(bridge_g2),
                    'num_bridges': num_bridges,
                    'cell_distances_g1': d1,
                    'cell_distances_g2': d2,
                    'mean_distance_g1': mean_d1,
                    'mean_distance_g2': mean_d2,
                    'combined_mean_distance': total_mean_d,
                    'combined_max_variance': total_var_d,
                    'max_bridge_load': max_bridge_load,
                    'bridge_score': combo_score
                }

        if verbose and best_result:
            print("=" * 75)
            print(f"POBJEDNIČKA KOMB. (G{group_id}): No_bridges: {best_result['num_bridges']}, Score: {best_result['bridge_score']:.2f}")
            print("=" * 75 + "\n")

        return best_result

def all_max_matchings(possible_pairs):
    """
    pronalazi sve kombinacije maksimalnih uparivanja iz liste ponudjenih parova.

    argumenti:
        possible_pairs (list): lista parova celija (c1, c2)

    povratna vrijednost:
        list: lista svih optimalnih kombinacija uparivanja
    """
    pairs = list(set(possible_pairs))

    by_g1 = {}
    for g1, g2 in pairs:
        by_g1.setdefault(g1, []).append(g2)

    g1_values = list(by_g1)

    best_size = 0
    best_matchings = []

    def search(i, used_g2, current):
        nonlocal best_size, best_matchings

        if i == len(g1_values):
            size = len(current)

            if size > best_size:
                best_size = size
                best_matchings = [tuple(current)]

            elif size == best_size:
                best_matchings.append(tuple(current))

            return

        remaining = len(g1_values) - i

        if len(current) + remaining < best_size:
            return

        g1 = g1_values[i]

        search(i + 1, used_g2, current)

        for g2 in by_g1[g1]:
            if g2 in used_g2:
                continue

            used_g2.add(g2)
            current.append((g1, g2))

            search(i + 1, used_g2, current)

            current.pop()
            used_g2.remove(g2)

    search(0, set(), [])

    return best_matchings
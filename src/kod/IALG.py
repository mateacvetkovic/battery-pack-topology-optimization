import random
from src.kod.heuristics import skewed_sweep,check_simple_solution, snake_pattern_rows, snake_pattern_bottom_up, snake_pattern_columns, max_shared_boundary_distance_based_universal_smart


def remove_duplicates(population):
    """
    Uklanja duplikate iz populacije na temelju otiska prsta (fingerprint) rjesenja.

    Argumenti:
        population (list): lista rjesenja koja se filtrira.

    Vraca:
        list: lista jedinstvenih rjesenja bez duplikata.
    """
    unique_population = []
    seen_fingerprints = set()
    
    for sol in population:
        fp = sol.get_fingerprint()
        if fp not in seen_fingerprints:
            seen_fingerprints.add(fp)
            unique_population.append(sol)
            
    return unique_population


def generate_initial_population(pack, population_size=10, snakes=True):
    """
    kreira raznoliku pocetnu populaciju bez duplikata generirajuci nova rjesenja
    sve dok se ne popuni zeljeni broj jedinki ili ne potrosi maksimalan broj pokusaja.

    Argumenti:
        pack (object): objekt koji definira problem pakiranja.
        population_size (int): zeljena velicina populacije.
        snakes (bool): zastavica koja odredjuje hoce li se koristiti zmijoliki uzorci.

    Vraca:
        list: sortirana lista pocetnih rjesenja po fitnessu.
    """
    population = []
    seen_fingerprints = set() 
    
    def try_add_solution(sol):
        """
        pomocna funkcija za validaciju, izracun fitnessa i dodavanje rjesenja u skup.

        Argumenti:
            sol (object): rjesenje koje se pokusava dodati.

        Vraca:
            bool: true ako je rjesenje uspjesno dodano, inace false.
        """
        if sol and sol.is_valid():
            fp = sol.get_fingerprint()
            if fp not in seen_fingerprints:
                seen_fingerprints.add(fp)
                sol.calculate_total_fitness()
                population.append(sol)
                return True
        return False

    if pack.type == 'rectangle':
        sol = check_simple_solution(pack)
        try_add_solution(sol)

    snake_generators = [
        lambda p: snake_pattern_rows(p),
        lambda p: snake_pattern_bottom_up(p),
        lambda p: snake_pattern_columns(p)
    ]

    if snakes == True:
        for gen_func in snake_generators:
            sol = gen_func(pack)
            try_add_solution(sol)

    strategies = ["horizontal", "vertical", "diagonal"]
    for _ in range(population_size * 3):
        sol = skewed_sweep(pack, axis=random.choice(["x", "y"]), reverse=random.choice([False, True]),
                           amp=random.choice([0.5, 1.0, 1.5, 2.0]), noise=random.choice([0.0, 0.3]))
        try_add_solution(sol)
    directions = {
        "horizontal": ["top-down", "bottom-up"],
        "vertical": ["left-to-right", "right-to-left"],
        "diagonal": ["top-down", "bottom-up"]
    }
    diagonal_types = ["backslash", "slash"]

    temperatures = [0, 15, 25]            
    p_stochastics = [0.0, 0.2, 0.5]     

    attempts = 0
    max_attempts = population_size * 40  
    
    while len(population) < population_size * 4 and attempts < max_attempts:
        attempts += 1
        
        strat = random.choice(strategies)
        dir = random.choice(directions[strat])
        temp = random.choice(temperatures)
        p_stoch = random.choice(p_stochastics)
        diag_type = random.choice(diagonal_types)

        if strat == 'diagonal':
            rev_cols = random.choice([True, False]) 
        else:
            rev_cols = False

        sol = max_shared_boundary_distance_based_universal_smart(pack, strategy=strat, direction=dir, diagonal_type=diag_type, reverse_columns=rev_cols, p_stochastic=p_stoch, temperature=temp)
        
        try_add_solution(sol)

    population.sort(key=lambda s: s.fitness, reverse=True)
    
    if len(population) >= population_size:
        return population[:population_size]
    else:
        return population


class ClonalgOptimizer:
    """
    Razred koji implementira imunoloski algoritam clonalg za optimizaciju rjesenja pakiranja.

    """
    def __init__(self, pack, max_iterations=50, population_size=50, clones_per_antibody=5, d_new_individuals=12, beta=10):
        """
        Argumenti:
        pack (object): objekt koji opisuje mrežu
        max_iterations (int): maksimalni broj iteracija optimizacije.
        population_size (int): ciljana velicina populacije.
        clones_per_antibody (int): bazni broj klonova po antitijelu.
        d_new_individuals (int): broj novih jedinki koje se unose u svaku populaciju.
        beta (int): faktor za skaliranje broja klonova u ovisnosti o rangu.
        """
        self.pack = pack
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.clones_per_antibody = clones_per_antibody
        self.d_new_individuals = d_new_individuals
        self.beta = beta
        
        self.start_population = []
        self.best_ever_solution = None
        self.best_ever_fitness = -float('inf')
        self.history = []
        self.mutation_stats_history = []
        self.final_population=[]

        self.avg_fitness = []
    def run(self):
        """
        pokrece glavnu petlju algoritma s evolucijskim operacijama.

        Vraca:
            object: najbolje pronadjeno rjesenje (jedinka) tijekom izvodjenja.
        """
        population = generate_initial_population(self.pack, self.population_size, snakes=True)
        self.start_population = [ind.copy() for ind in population]
        stag = 0
        
        mutation_stats = {
            'swap_boundary': {'tried': 0, 'improved': 0},
            'cyclic': {'tried': 0, 'improved': 0},
            'steal': {'tried': 0, 'improved': 0}
        }

        for t in range(self.max_iterations):
            
            population.sort(key=lambda s: s.fitness, reverse=True)
            
            current_best = population[0]
            if current_best.fitness > self.best_ever_fitness:
                stag = 0
                self.best_ever_fitness = current_best.fitness
                self.best_ever_solution = current_best.copy()
            else:
                stag += 1
                if stag > 100:
                    print(f"[CLONALG] rani prekid u iteraciji {t+1} zbog stagnacije.")
                    break
                    
            if current_best.eval_info['total_bridges'] == (self.pack.no_groups - 1) * self.pack.no_cells_per_group:
                return [current_best]

            clonal_pool = []
            # odabir broja klonova u ovisnosti o kvaliteti rjesenja - bolja rjesenja, vise klonova
            for i, antibody in enumerate(population):
                affinity_rank_factor = (len(population) - i) / len(population)
                
                num_clones = int(self.beta * affinity_rank_factor)
                num_clones = max(1, num_clones)

                for _ in range(num_clones):
                    clone = antibody.copy() 
                    clonal_pool.append((i, clone))
                    
            mutated_pool = []
            n_pop = len(population)

            valid_fitnesses = [s.fitness for s in population if s.fitness != -float('inf')]

            if valid_fitnesses:
                self.avg_fitness.append(sum(valid_fitnesses) / len(valid_fitnesses))
            else:
                self.avg_fitness.append(0)
                        
            for rank, (parent_rank, clone) in enumerate(clonal_pool):

                parent_fitness = population[parent_rank].fitness                
                rank_ratio = parent_rank / (n_pop - 1) if n_pop > 1 else 0.0

                # bolji rade 1 mutaciju, losiji mogu odraditi vise njih u nizu (npr. do 4)
                num_mutations = 1 + int(3 * rank_ratio)

                for _ in range(num_mutations):
                    r = random.random()
                    if r < 0.5:
                        mut_type = 'swap_boundary'
                        mutate_swap_boundary(clone)
                    elif r < 0.75:
                        mut_type = 'cyclic'
                        mutate_cyclic_3way_swap(clone)
                    else:
                        mut_type = 'steal'
                        mutate_steal_and_shift(clone)
                
                    mutation_stats[mut_type]['tried'] += 1
                
                clone.calculate_total_fitness()
                
                if clone.fitness > parent_fitness:
                    mutation_stats[mut_type]['improved'] += 1
                    
                mutated_pool.append(clone)
            #combined_pool = population + mutated_pool
            num_elite = 5
            next_population = [ind.copy() for ind in population[:num_elite]]
        
            unique_mutants = []
            seen_fps = {ind.get_fingerprint() for ind in next_population}

            for ind in mutated_pool:
                if ind:
                    fp = ind.get_fingerprint()
                    if fp not in seen_fps:
                        seen_fps.add(fp)
                        unique_mutants.append(ind)
                        
            unique_mutants.sort(key=lambda s: s.fitness, reverse=True)
            
            target_size_without_new = self.population_size - self.d_new_individuals
            slots_for_mutants = target_size_without_new - len(next_population)
            if slots_for_mutants > 0:
                # next_population.extend(unique_mutants[:slots_for_mutants])
                #chosen_mutants = roulette_wheel_selection(unique_mutants, slots_for_mutants)

                chosen_mutants = hybrid_elitist_random_selection(unique_mutants, slots_for_mutants)
                next_population.extend(chosen_mutants)

            needed_new = self.d_new_individuals
            if needed_new > 0:
                new_bloods = generate_initial_population(self.pack, population_size=needed_new, snakes=False)
                for nb in new_bloods:
                    if nb:
                        next_population.append(nb)
                        
            while len(next_population) < self.population_size and population:
                next_population.append(population[0].copy())
                
            population = next_population
            valid_fitnesses = [s.fitness for s in population if s.fitness != -float('inf')]

            if valid_fitnesses:
                self.avg_fitness.append(sum(valid_fitnesses) / len(valid_fitnesses))
            else:
                self.avg_fitness.append(0)
            self.mutation_stats_history.append({
                'iteration': t + 1,
                'stats': {k: v.copy() for k, v in mutation_stats.items()}
            })
            
            self.history.append({
                'iteration': t + 1,
                'best_population_fitness': self.best_ever_fitness
            })
            if t % 10 == 0:
                print(f"iteracija {t+1}/{self.max_iterations} | najbolji fitness u populaciji: {self.best_ever_fitness:.2f}")
            # print("   -> Uspješnost mutacija:")
            # for m_name, stats in mutation_stats.items():
            #     tried = stats['tried']
            #     improved = stats['improved']
            #     rate = (improved / tried * 100) if tried > 0 else 0
            #     print(f"      {m_name}: {improved}/{tried} ({rate:.1f}%)")
        self.final_population = [ind.copy() for ind in population]

        best_solutions = [ind.copy() for ind in self.final_population if abs(ind.fitness - self.best_ever_fitness) < 1e-9]
        
        if not best_solutions and self.best_ever_solution is not None:
            best_solutions = [self.best_ever_solution]
            
        return best_solutions


def mutate_swap_boundary(solution):
    """
    ide kroz sve celije odabrane grupe, pronade susjedne celije (one koje su granica),
    zamjenjuje neke od njih i vraca samo validno rjesenje.

    Argumenti:
        solution (object): rjesenje na kojem se provodi mutacija.

    Vraca:
        bool: true ako je mutacija uspjesno primijenjena, inace false.
    """
    selected_g = select_groups_for_mutation(solution)
    
    if selected_g is None or selected_g == -1:
        return False

    selected_cells = list(solution.group_dict.get(selected_g, set()))
    
    swap_candidates = []
    for c_A in selected_cells:
        for c_B in solution.pack.get_neighbours_ids(c_A):
            neighbor_g = solution.cell_to_group[c_B]
            
            if neighbor_g is not None and neighbor_g != -1 and neighbor_g != selected_g:
                swap_candidates.append(('swap', c_A, c_B, neighbor_g))
                
            elif neighbor_g == -1 or neighbor_g is None:
                swap_candidates.append(('empty', c_A, c_B, selected_g))

    if not swap_candidates:
        return False
        
    random.shuffle(swap_candidates)
    for action_type, c_A, c_B, target_g in swap_candidates:
        if action_type == 'swap':
            solution.assign_cell_to_group(c_A, target_g)
            solution.assign_cell_to_group(c_B, selected_g)
            
            is_selected_ok = solution.is_group_connected(selected_g)
            is_neighbor_ok = solution.is_group_connected(target_g)
            
            if is_selected_ok and is_neighbor_ok:
                return True
                
            solution.assign_cell_to_group(c_A, selected_g)
            solution.assign_cell_to_group(c_B, target_g)
            
        elif action_type == 'empty':
            solution.assign_cell_to_group(c_A, -1)
            solution.assign_cell_to_group(c_B, selected_g)
            
            if solution.is_group_connected(selected_g):
                return True
            solution.assign_cell_to_group(c_A, selected_g)
            solution.assign_cell_to_group(c_B, -1)
    return False


def mutate_cyclic_3way_swap(solution):
    """
    provodi trostranu kruznu zamjenu celija izmedju tri susjedne grupe radi poticanja lokalne promjene.

    Argumenti:
        solution (object): rjesenje na kojem se izvrsava zamjena celija.

    Vraca:
        bool: true ako je zamjena uspjela uz ocuvanu povezanost, inace false.
    """
    g1 = select_groups_for_mutation(solution)
    if g1 is None:
        return False

    g1_cells = list(solution.group_dict.get(g1, set()))
    
    g2_candidates = set()
    for c1 in g1_cells:
        for nbr in solution.pack.get_neighbours_ids(c1):
            g_nbr = solution.cell_to_group[nbr]
            if g_nbr is not None and g_nbr != -1 and g_nbr != g1:
                g2_candidates.add(g_nbr)
                
    if not g2_candidates:
        return False
    g2 = random.choice(list(g2_candidates))
    
    g2_cells = list(solution.group_dict.get(g2, set()))
    g3_candidates = set()
    for c2 in g2_cells:
        for nbr in solution.pack.get_neighbours_ids(c2):
            g_nbr = solution.cell_to_group[nbr]
            if g_nbr is not None and g_nbr != -1 and g_nbr not in (g1, g2):
                g3_candidates.add(g_nbr)
                
    if not g3_candidates:
        return False
    g3 = random.choice(list(g3_candidates))

    candidates_c1 = [c for c in g1_cells if any(solution.cell_to_group[n] == g2 for n in solution.pack.get_neighbours_ids(c))]
    candidates_c2 = [c for c in g2_cells if any(solution.cell_to_group[n] == g3 for n in solution.pack.get_neighbours_ids(c))]
    candidates_c3 = [c for c in solution.group_dict.get(g3, set()) if any(solution.cell_to_group[n] == g1 for n in solution.pack.get_neighbours_ids(c))]

    if not candidates_c1 or not candidates_c2 or not candidates_c3:
        return False

    random.shuffle(candidates_c1)
    random.shuffle(candidates_c2)
    random.shuffle(candidates_c3)

    for c1 in candidates_c1[:5]:
        for c2 in candidates_c2[:5]:
            for c3 in candidates_c3[:5]:
                solution.group_dict[g1].remove(c1)
                solution.group_dict[g1].add(c3)
                solution.group_dict[g2].remove(c2)
                solution.group_dict[g2].add(c1)
                solution.group_dict[g3].remove(c3)
                solution.group_dict[g3].add(c2)
                
                solution.cell_to_group[c1] = g2
                solution.cell_to_group[c2] = g3
                solution.cell_to_group[c3] = g1
                
                if solution.is_group_connected(g1) and solution.is_group_connected(g2) and solution.is_group_connected(g3):
                    return True
                
                solution.group_dict[g1].remove(c3)
                solution.group_dict[g1].add(c1)
                solution.group_dict[g2].remove(c1)
                solution.group_dict[g2].add(c2)
                solution.group_dict[g3].remove(c2)
                solution.group_dict[g3].add(c3)
                
                solution.cell_to_group[c1] = g1
                solution.cell_to_group[c2] = g2
                solution.cell_to_group[c3] = g3

    return False


def mutate_steal_and_shift(solution):
    """
    mutacija 'kradi i gurni' (steal & shift):
    1. odabire grupu a i pronalazi njezinu rubnu celiju koja granici s grupom b.
    2. provjerava ima li grupa b u svojoj okolini prazno mjesto (-1).
    3. ako ima, grupa a uzima poziciju od grupe b, a celija grupe b klizi u prazno mjesto.

    Argumenti:
        solution (object): rjesenje na kojem se primjenjuje mutacija.

    Vraca:
        bool: true ako je mutacija bila uspjesna, inace false.
    """
    all_groups = [g for g in solution.group_dict.keys() if g != -1]
    if not all_groups:
        return False

    group_a = select_groups_for_mutation(solution)
    cells_a = list(solution.group_dict.get(group_a, set()))
    if not cells_a:
        return False

    candidates_for_steal = []
    for cell_a in cells_a:
        for neighbor_id in solution.pack.get_neighbours_ids(cell_a):
            neighbor_g = solution.cell_to_group[neighbor_id] if neighbor_id < len(solution.cell_to_group) else -1
            if neighbor_g != -1 and neighbor_g != group_a:
                candidates_for_steal.append((neighbor_id, neighbor_g))

    if not candidates_for_steal:
        return False

    cell_to_steal, group_b = random.choice(candidates_for_steal) #celija iz grupe b koju cemo ubacit u a
    cells_b = list(solution.group_dict.get(group_b, set()))
    cell_from_b_to_shift = None
    empty_spot = None

    for cell_b in cells_b:
        if cell_b == cell_to_steal:
            continue
        for neighbor_id in solution.pack.get_neighbours_ids(cell_b):
            neighbor_g = solution.cell_to_group[neighbor_id] if neighbor_id < len(solution.cell_to_group) else -1
            if neighbor_g == -1:
                cell_from_b_to_shift = cell_b
                empty_spot = neighbor_id
                break
        if empty_spot is not None:
            break

    if empty_spot is None:
        return False


    cell_from_a_to_free = None
    for cell_a in cells_a:
        if cell_a == cell_to_steal:
            continue
        for neighbor_id in solution.pack.get_neighbours_ids(cell_a):
            neighbor_g = solution.cell_to_group[neighbor_id] if neighbor_id < len(solution.cell_to_group) else -1
            if neighbor_g == -1:
                cell_from_a_to_free = neighbor_id
                cell_to_drop = cell_a
                break
        if cell_from_a_to_free is not None:
            break

    if cell_from_a_to_free is None:
        return False

    backup_steal = solution.cell_to_group[cell_to_steal]
    backup_shift = solution.cell_to_group[cell_from_b_to_shift]
    backup_empty = solution.cell_to_group[empty_spot]
    backup_drop = solution.cell_to_group[cell_to_drop]
    backup_drop_spot = solution.cell_to_group[cell_from_a_to_free]



    solution.assign_cell_to_group(cell_to_steal, group_a)
    solution.assign_cell_to_group(empty_spot, group_b)
    solution.assign_cell_to_group(cell_to_drop, -1)
    if solution.is_group_connected(group_a) and solution.is_group_connected(group_b):
        return True

    solution.assign_cell_to_group(cell_to_steal, backup_steal)
    solution.assign_cell_to_group(empty_spot, backup_empty)
    solution.assign_cell_to_group(cell_to_drop, backup_drop)

    return False


def select_groups_for_mutation(solution, random_prob=0.5):
    """
    odabire grupu za mutaciju na temelju vjerojatnosti: nasumicno ili najgoru grupu prema fitnessu.

    Argumenti:
        solution (object): rjesenje iz kojeg se bira grupa.
        random_prob (float): vjerojatnost odabira potpuno nasumicne grupe.

    Vraca:
        int/object: identifikator odabrane grupe ili none ako nema grupa.
    """
    all_groups = [g for g in solution.group_dict.keys() if g != -1]    
    if random.random() < random_prob and all_groups:
        return random.choice(all_groups)
            
    worst_g = solution.eval_info.get('worst_group') if solution.eval_info else None
    if worst_g is not None and worst_g != -1 and worst_g in solution.group_dict:
        return worst_g
        
    return random.choice(all_groups) if all_groups else None


def roulette_wheel_selection(mutants, num_to_select):
    """
    bira 'num_to_select' jedinki iz liste 'mutants' metodom ruleta 
    (fitness proportionate selection) s prilagodbom za negativne vrijednosti.

    Argumenti:
        mutants (list): lista kandidata mutanata za odabir.
        num_to_select (int): broj jedinki koje treba odabrati.

    Vraca:
        list: odabrane jedinke mutanata.
    """
    if not mutants or num_to_select <= 0:
        return []
        
    if len(mutants) <= num_to_select:
        return list(mutants)

    fitnesses = [m.fitness for m in mutants]
    
    valid_fitnesses = [f for f in fitnesses if f != -float('inf')]
    min_f = min(valid_fitnesses) if valid_fitnesses else 0
    
    shift = abs(min_f) + 0.1 if min_f < 0 else 0.1
    adjusted_fitnesses = [(f + shift) if f != -float('inf') else 0.0001 for f in fitnesses]
    
    total_fitness = sum(adjusted_fitnesses)
    
    if total_fitness <= 0:
        return random.sample(mutants, num_to_select)
        
    chosen_mutants = []
    for _ in range(num_to_select):
        pick = random.uniform(0, total_fitness)
        current_sum = 0
        for idx, weight in enumerate(adjusted_fitnesses):
            current_sum += weight
            if current_sum >= pick:
                chosen_mutants.append(mutants[idx])
                break
        if len(chosen_mutants) <= _:
            chosen_mutants.append(mutants[-1])
            
    return chosen_mutants


def hybrid_elitist_random_selection(mutants, num_to_select, elite_ratio=0.8):
    """
    bira 'num_to_select' jedinki iz liste mutanata kombinirajuci elitizam i nasumicni odabir.

    Argumenti:
        mutants (list): lista kandidata mutanata.
        num_to_select (int): ukupan broj jedinki koje je potrebno odabrati.
        elite_ratio (float): udio najboljih jedinki koje se obavezno zadrzavaju.

    Vraca:
        list: odabrane jedinke kombinacijom elitizma i nasumicnog izbora.
    """
    if not mutants or num_to_select <= 0:
        return []
        
    if len(mutants) <= num_to_select:
        return list(mutants)

    sorted_mutants = sorted(mutants, key=lambda s: s.fitness, reverse=True)
    
    num_elite = int(num_to_select * elite_ratio)
    num_random = num_to_select - num_elite
    
    if num_elite == 0 and num_to_select > 0:
        num_elite = 1
        num_random = num_to_select - 1

    chosen_mutants = sorted_mutants[:num_elite]
    
    remainder = sorted_mutants[num_elite:]
    
    if remainder and num_random > 0:
        if len(remainder) <= num_random:
            chosen_mutants.extend(remainder)
        else:
            chosen_mutants.extend(random.sample(remainder, num_random))
            
    while len(chosen_mutants) < num_to_select and remainder:
        for m in remainder:
            if m not in chosen_mutants:
                chosen_mutants.append(m)
                break
        else:
            break

    return chosen_mutants
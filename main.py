import os
import sys
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt

from src.kod.grids import HexGrid, RectangularGrid
from src.kod.containers import TriangleContainer
from src.kod.BatteryPack import BatteryPack
from src.kod.BatterySolution import BatterySolution
from src.kod.optimizers import TriangleContainerOptimizer
from src.kod.IALG import ClonalgOptimizer

CELL_SPECS = {
    '18650': {'diameter': 18, 'nominal_voltage': 3.6}, # mm, V
    '21700': {'diameter': 21, 'nominal_voltage': 3.6}  # mm, V
}

ALLOWED_NET_TYPES = ['saćasti', 'pravokutni']
ALLOWED_CELL_TYPES = ['18650', '21700']


def get_user_inputs():
    print("=" * 60)
    print("     OPTIMIZACIJA I DIZAJN BATERIJSKOG PAKETA")
    print("=" * 60)

    while True:
        print("\n[1/4] Odaberite tip ćelije:")
        for idx, c_type in enumerate(ALLOWED_CELL_TYPES, 1):
            print(f"  {idx}. {c_type}")
        try:
            c_choice = int(input("Unos: ")) - 1
            if 0 <= c_choice < len(ALLOWED_CELL_TYPES):
                cell_type = ALLOWED_CELL_TYPES[c_choice]
                break
            else:
                print("Krivi odabir. Odaberite ponuđeni broj.")
        except ValueError:
            print("Unesite cijeli broj.")

    while True:
        print("\n[2/4] Unesite udaljenost među ćelijama(mm): ")
        try:
            d = int(input("Unos: "))
            if CELL_SPECS[cell_type]['diameter'] <= d:
                break
            else:
                print("Unesite ispravnu vrijednost udaljenosti.")
        except ValueError:
            print("Unesite cijeli broj.")

    while True:
        print("\n[3/4] Odaberite tip mreže (grid):")
        for idx, net in enumerate(ALLOWED_NET_TYPES, 1):
            print(f"  {idx}. {net}")
        try:
            n_choice = int(input("Unos: ")) - 1
            if 0 <= n_choice < len(ALLOWED_NET_TYPES):
                net_type = ALLOWED_NET_TYPES[n_choice]
                break
            else:
                print("Odaberite ponuđeni broj.")
        except ValueError:
            print("Unesite cijeli broj.")

    while True:
        print("\n[4/4] Odaberite oblik paketa:")
        print("  1. Pravokutnik (Širina x Visina)")
        print("  2. Trokut (3 stranice)")
        try:
            shape_choice = int(input("Unos: "))
            if shape_choice in [1, 2]:
                break
            else:
                print("Krivi odabir. Unesite 1 ili 2.")
        except ValueError:
            print("Unesite cijeli broj.")

    while True:
        try:
            if shape_choice == 1:
                shape_type = 'rectangle'
                print("\n[4/4] Unesite dimenzije pravokutnika:")
                w_cm = float(input("Unesite širinu paketa (cm): "))
                h_cm = float(input("Unesite visinu paketa (cm): "))
                if w_cm > 0 and h_cm > 0:
                    dimensions_cm = [w_cm, h_cm]
                    break
                else:
                    print("Dimenzije moraju biti veće od 0.")
            else:
                shape_type = 'triangle'
                print("\n[4/4] Unesite dimenzije trokuta:")
                a_cm = float(input("Unesite stranicu a (cm): "))
                b_cm = float(input("Unesite stranicu b (cm): "))
                c_cm = float(input("Unesite stranicu c (cm): "))
                if a_cm <= 0 or b_cm <= 0 or c_cm <= 0:
                    print("Sve stranice moraju biti veće od 0.")
                elif (a_cm + b_cm > c_cm) and (a_cm + c_cm > b_cm) and (b_cm + c_cm > a_cm):
                    dimensions_cm = [a_cm, b_cm, c_cm]
                    break
                else:
                    print("Neispravne dimenzije! Zbroj dviju stranica mora biti veći od treće.")
        except ValueError:
            print("Unesite ispravnu brojčanu vrijednost.")

    return {
        'cell_type': cell_type,
        'cell_distance':d,
        'net_type': net_type,
        'shape_type': shape_type,
        'dimensions_cm': dimensions_cm
    }

def setup_battery_pack(inputs):
    c_spec = CELL_SPECS[inputs['cell_type']]
    d = inputs['cell_distance']

    dims_mm = [dim * 10.0 for dim in inputs['dimensions_cm']]

    if inputs['shape_type'] == 'rectangle':
        v = None
        W, H = dims_mm[0], dims_mm[1]
        if inputs['net_type'] == 'saćasti':
            grid_h = HexGrid(d, W, H)
            grid_h.fit()
            count_h = grid_h.total_cells

            grid_v = HexGrid(d, H, W)
            grid_v.fit()
            count_v = grid_v.total_cells

            if count_h >= count_v:
                grid = grid_h
            else:
                grid = grid_v
        else:
            grid = RectangularGrid(cell_distance=d, width=W, height=H)
            grid.fit()

    else: 
        container = TriangleContainer(inputs['dimensions_cm'])        
        max_dim = container.calculate_triangle_grid_size(d) 
        W, H = max_dim, max_dim

        if inputs['net_type'] == 'saćasti':
            grid = HexGrid(cell_distance=d, width=W, height=H)
        else:
            grid = RectangularGrid(cell_distance=d, width=W, height=H)
        grid.fit()
        to = TriangleContainerOptimizer(container, grid)
        cells_inside_ids, v, grid = to.solve(15)

        grid.create_subgrid(cells_inside_ids)

    try:
        pack = BatteryPack(
            grid=grid, 
            cell_type=inputs['cell_type'], 
            vertices=v,
            type=inputs['shape_type']
        )
        return pack

    except ValueError as e:
        print(e)
        sys.exit()

def interactive_cell_removal_and_voltage_setup(pack):
    print("\n" + "="*60)
    print("CRTANJE INICIJALNOG GRID-A...")
    print("Prikazuje se prozor s brojevima (ID-jevima) ćelija...")
    print("="*60)
    
    pack.display()  

    print("\n[BMS] Uklanjanje ćelija za BMS, šarafe i nosače:")
    remove_input = input("Unesite ID-jeve ćelija koje želite ukloniti (odvojene zarezom, npr. 0,1,12) ili stisnite ENTER za nastavljanje bez uklanjanja ćelija: ")

    if remove_input.strip():
        try:
            bms_ids = [int(x.strip()) for x in remove_input.split(',')]
            pack.set_bms_cells(bms_ids)  
            print(f"-> Uspješno uklonjene ćelije za BMS/komponente.")
            
            print("Prikaz mreže s označenim BMS mjestima...")
            pack.display()  
        except ValueError:
            print("Neispravan unos ID-jeva. Nastavlja se bez uklanjanja.")

    print(f"\nStanje paketa:")
    print(f" - Ukupno iskoristivih ćelija: {pack.usable_cells}")

    cell_v = CELL_SPECS[pack.cell_type]['nominal_voltage']

    # ispis mogućih konfiguracija
    print("\n" + "="*60)
    print("MOGUĆE S/P KONFIGURACIJE PAKETA:")
    print("="*60)
    configs = pack.get_possible_configurations(cell_v)
    for cfg in configs:
        print(f" - {cfg['S']}S {cfg['P']}P -> Napon: {cfg['voltage']:.1f} V (Iskorišteno ćelija: {cfg['used_cells']}/{pack.usable_cells})")

    # unos napona paketa
    while True:
        print("\nElektrični parametri:")
        try:
            target_pack_voltage = float(input("Unesite željeni napon paketa (V) [npr. 36 ili 48]: "))
            if target_pack_voltage > 0 and target_pack_voltage >= cell_v:
                try:
                    pack.set_voltage(target_pack_voltage, cell_v)
                    break
                except ValueError as e:
                    print(f"Greška pri postavljanju napona: {e}")
            else:
                print("Neispravan napon paketa.")
        except ValueError:
            print("Unesite ispravan broj.")

    print(f"\nOdabrana konfiguracija:")
    print(f" - Konfiguracija: {pack.no_groups}S {pack.no_cells_per_group}P")  
    
    if pack.usable_cells < pack.no_groups:
        print("\n[GREŠKA] Nedovoljno ćelija u paketu za zadani napon! Program prekinut.")
        sys.exit()

def run_optimization(pack, inputs):
    print("\n" + "="*60)
    print("KREĆE OPTIMIZACIJA I EVOLUCIJA POVEZIVANJA GRUPA...")
    print("="*60)
    current_dir = os.getcwd()

    grid_type = inputs['net_type']
    shape_type = inputs['shape_type']
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if shape_type == 'rectangle':
        W, H = inputs['dimensions_cm']
        file_name = f"baterija_{grid_type}_{shape_type}_W{W}_H{H}_{timestamp}"
    else:
        a,b,c = inputs['dimensions_cm']
        file_name = f"baterija_{grid_type}_{shape_type}_A{a}_B{b}_C{c}_{timestamp}"
    file_name = file_name.replace(".", ",")

    best_sol_ever = None
    best_fitness_ever = 0

    for i in range(3):
        print(f"{i+1}/3 pokušaja")
        optimizer = ClonalgOptimizer(pack, max_iterations=500, population_size=50)
        best_solutions = optimizer.run()  
        print(f"Pronađeno {len(best_solutions)} rješenja.")
        for id, best_solution in enumerate(best_solutions):
            print(f"\n[EVALUACIJA RJEŠENJA {id+1}/{len(best_solutions)}]")
            fitness_val = best_solution.fitness  
            eval_info = best_solution.evaluate_solution(verbose=True)  
            if (best_fitness_ever < best_solution.fitness):
                best_fitness_ever = best_solution.fitness
                best_sol_ever = best_solution

            file_name_it = f"{file_name}_it{i+1}_id{id}.png"

            print(f"\n-> Ukupna ocjena kvalitete paketa (Fitness): {fitness_val:.2f}")  
            print(f"-> Minimalan broj direktnih spojeva između ikojih susjednih grupa: {eval_info['min_bridges']}")  
            print(f"-> Ukupan broj direktnih serijskih spojeva u paketu/idealan broj spojeva: {eval_info['total_bridges']}/{pack.no_cells_per_group*(pack.no_groups-1)}")  
            if eval_info['total_bridges'] == pack.no_cells_per_group*(pack.no_groups-1):
                best_solution.display(save=True, name=file_name_it)
                print(f"-> Slika spremljena na lokaciju: {os.path.join(current_dir, file_name_it)}")
                break
            if eval_info['worst_group'] is not None:
                print(f"-> Najkritičnija grupa s najmanje direktnih spojeva: Grupa {eval_info['worst_group']}")  

            print("\nGeneriranje grafičkog prikaza konačnog paketa...")
            if (id==0):
                best_solution.display(save=True, name=file_name_it)
            else:
                best_solution.display(save=True, show=False,name=file_name_it)
            print(f"-> Slika spremljena na lokaciju: {os.path.join(current_dir, file_name_it)}")

    input("\nProgram je završio i prikazao rezultate. Za izlaz i zatvaranje svih prozora pritisnite ENTER... ")
    plt.close('all')

if __name__ == "__main__":
    inputs = get_user_inputs()
    pack = setup_battery_pack(inputs)
    interactive_cell_removal_and_voltage_setup(pack)
    run_optimization(pack, inputs)

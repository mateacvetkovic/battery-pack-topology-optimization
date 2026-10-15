import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.patches import Polygon

plt.ion()  


def _init_plot(W, H, vertices, show):
    base_scale = 12 
    if W >= H:
        width_fig = base_scale
        height_fig = base_scale * (H / W)
    else:
        height_fig = base_scale
        width_fig = base_scale * (W / H)
        
    if not show:
        plt.ioff()

    fig, ax = plt.subplots(figsize=(max(width_fig, 4), max(height_fig, 4)))  
    
    if not show:
        plt.ion() # Vratimo natrag na ion stanje
    ax.add_patch(patches.Rectangle((0, 0), W, H, linewidth=2, edgecolor='black', facecolor='none'))
    if vertices is not None:
        ax.add_patch(Polygon(vertices, closed=True, fill=None, edgecolor='blue', linewidth=2, linestyle='--'))
    return fig, ax


def _finalize_plot(ax, W, H, title, save, name, show):
    ax.set_xlim(-5, W + 0.1 * W)
    ax.set_ylim(-5, H + 0.1 * H)
    ax.set_xlabel('Širina (mm)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Visina (mm)', fontsize=16, fontweight='bold')
    ax.set_aspect('equal')
    ax.set_title('{}'.format(title), fontsize=16)
    if save:
        plt.savefig(name,bbox_inches="tight")
    if show:
        plt.draw()
        plt.pause(0.1)


def plotGrid(cells, W, H, d, cell_d=None, save=False, show=True, name='battery_pack.png', title=None, vertices=None, higlight_ids=None):
    if not cells:
        print("U paketu nema ćelija!")
        return
    if cell_d is None:
        cell_d = d
    _, ax = _init_plot(W, H, vertices, show)
    title_text = title if title is not None else f'Dimenzije baterije: {W}x{H} cm'
    added_labels = set()
    for i, cell in enumerate(cells):
        y, x = cell.y, cell.x
        if 0 <= x <= W and 0 <= y <= H:
            circle_color = 'grey'
            if higlight_ids is not None and cell.id in higlight_ids:
                c_label = 'BMS' if 'BMS' not in added_labels else None
                added_labels.add('BMS')
                circle_color = 'black'
                ax.text(x, y, "BMS", fontsize=12, ha='center', va='center', color='white', fontweight='bold', zorder=7)
            else:
                c_label = 'Slobodne ćelije' if 'FREE' not in added_labels else None
                added_labels.add('FREE')
                ax.text(x, y, f"{cell.id}", fontsize=12, ha='center', va='center', color='grey', fontweight='bold', zorder=7)
            
            ax.add_patch(patches.Circle((x, y), cell_d/2, edgecolor='black', facecolor=circle_color, alpha=0.6, label = c_label))

    ax.legend(loc="upper right", fontsize=14)

    ax.tick_params(axis='both', which='major', labelsize=12)

    _finalize_plot(ax, W, H, title_text, save, name, show)


def plotGroups(cells, W, H, d, cell_to_group, no_groups,  cell_d=None, save=False, show=True, name='battery_pack.png', title='', vertices=None, inside_ids=None):
    if not cells:
        print("Nema ćelija za crtanje!")
        return
    
    if cell_d is None:
        cell_d = d
    _, ax = _init_plot(W, H, vertices, show)
    cmap = plt.get_cmap('nipy_spectral', no_groups if no_groups > 0 else 1)
    added_labels = set()
    for i, cell in enumerate(cells):
        y, x = cell.y, cell.x
        if getattr(cell, 'group', None) in [-2, 17]:
            current_group = cell.group
        else:
            current_group = cell_to_group[i] if cell_to_group is not None else cell.group
        if 0 <= x <= W and 0 <= y <= H:
            if current_group == -2:
                circle_color = 'black'
                c_label = 'BMS' if 'BMS' not in added_labels else None
                added_labels.add('BMS')
                ax.text(x, y, "BMS", fontsize=8, color='white', ha='center', va='center', fontweight='bold', zorder=7)
            elif current_group is None or current_group == -1:
                circle_color = 'grey'
                c_label = 'Slobodne ćelije' if 'FREE' not in added_labels else None
                added_labels.add('FREE')
                ax.text(x, y, f"C{cell.id}", fontsize=12, ha='center', va='center', color='black', fontweight='bold', zorder=7)
            else:
                norm_group = current_group / max(1, no_groups)
                circle_color = cmap(norm_group)
                group_key = f'G_{current_group}'
                c_label = f'Grupa {current_group}' if group_key not in added_labels else None
                added_labels.add(group_key)
                ax.text(x, y, f"G{current_group}", fontsize=12, ha='center', va='center', zorder=6)
                #ax.text(x, y, f"C{cell.id}", fontsize=12, ha='center', va='center', zorder=6, label=f'Grupa {current_group}')
            ax.add_patch(patches.Circle((x, y), cell_d/2, edgecolor='black', facecolor=circle_color, alpha=0.6, label=c_label))
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        sorted_pairs = sorted(zip(handles, labels), key=lambda pair: pair[1])
        handles, labels = zip(*sorted_pairs)
        handles, labels = list(handles), list(labels)

    ax.legend(
        handles, 
        labels, 
        title="Električni spoj:\n- unutar grupe: PARALELA\n- između grupa: SERIJA", 
        loc='upper left', 
        bbox_to_anchor=(1.02, 1), 
        borderaxespad=0,
        fontsize=14,
        title_fontsize=14
    )
    ax.tick_params(axis='both', which='major', labelsize=12)
    _finalize_plot(ax, W, H, title, save, name, show)


def plotBridges(cells, W, H, d, cell_to_group, no_groups, save=False, show=True, name='battery_pack.png', title='', vertices=None, inside_ids=None, bridges=None):
    if not cells:
        print("No cells to plot!")
        return

    _, ax = _init_plot(W, H, vertices)
    cmap = plt.get_cmap('nipy_spectral', no_groups if no_groups > 0 else 1)
    cell_coords = {}
    added_labels = set()

    for i, cell in enumerate(cells):
        y, x = cell.y, cell.x
        c_id = getattr(cell, 'id', i)
        cell_coords[c_id] = (x, y)
        current_group = cell_to_group[i] if cell_to_group is not None else getattr(cell, 'group', None)

        if 0 <= x <= W and 0 <= y <= H:
            if getattr(cell, 'group', None) == -2:
                circle_color = 'black'
                c_label = 'BMS' if 'BMS' not in added_labels else None
                added_labels.add('BMS')
                ax.text(x, y, "BMS", fontsize=8, color='white', ha='center', va='center', fontweight='bold', zorder=7)
            elif current_group is None or current_group == -1:
                circle_color = 'grey'
                c_label = 'Slobodne ćelije' if 'FREE' not in added_labels else None
                added_labels.add('FREE')
                ax.text(x, y, f"C{c_id}", fontsize=12, ha='center', va='center', color='black', fontweight='bold', zorder=7)
            else:
                norm_group = current_group / max(1, no_groups)
                circle_color = cmap(norm_group)
                group_key = f'G_{current_group}'
                c_label = f'Grupa {current_group}' if group_key not in added_labels else None
                added_labels.add(group_key)
                ax.text(x, y, f"G{current_group}", fontsize=8, ha='center', va='center', zorder=6, label=f'Grupa {current_group}')

            ax.add_patch(patches.Circle((x, y), d/2, edgecolor='black', facecolor=circle_color, alpha=0.6, label=c_label))

    if bridges is not None:
        for c1_id, c2_id in bridges:
            if c1_id in cell_coords and c2_id in cell_coords:
                x1, y1 = cell_coords[c1_id]
                x2, y2 = cell_coords[c2_id]
                label_text = 'Spojevi između grupa' if 'BRIDGE' not in added_labels else None
                added_labels.add('BRIDGE')
                ax.plot([x1, x2], [y1, y2], color='red', linewidth=2.5, linestyle='-', zorder=8, label=label_text)

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        sorted_pairs = sorted(zip(handles, labels), key=lambda pair: pair[1])
        handles, labels = zip(*sorted_pairs)
        handles, labels = list(handles), list(labels)

    ax.legend(
        handles, 
        labels, 
        title="Električni spoj:\n- unutar grupe: PARALELA\n- između grupa: SERIJA", 
        loc='upper left', 
        bbox_to_anchor=(1.02, 1), 
        borderaxespad=0
    )

    _finalize_plot(ax, W, H, title, save, name, show)

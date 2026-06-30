import pandas as pd

from .cabinet_geometry import CONFIG_DIR

AIR_ZONE_TYPE = 'air_zone'
BOUNDARY_NODE_TYPES = {'cabinet_boundary', 'ventilation'}


def _distance_mm(node_pos, src, dst):
    p1, p2 = node_pos[src], node_pos[dst]
    return sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5


def _validate_edges(table_name, table, component_nodes):
    missing_nodes = sorted(
        ({row.src for row in table.itertuples(index=False)} | {row.dst for row in table.itertuples(index=False)})
        - component_nodes
    )
    if missing_nodes:
        raise ValueError(f'{table_name} references unknown nodes: {missing_nodes}')


def _build_thermal_topology(nodes, node_type, link_conductance, air_neighbors):
    """Pre-compute thermal-network adjacency: solid-solid conduction, solid->air-zone mapping,
    zone-to-zone air links, zone outlets, and the cabinet-boundary partition.
    """
    air_zones = [n for n in nodes if node_type[n] == AIR_ZONE_TYPE]
    boundary_nodes = [n for n in nodes if node_type[n] in BOUNDARY_NODE_TYPES]

    # Solid-solid conduction. We also treat any air_zone<->cabinet_boundary thermal_edge as
    # an air<->wall convective coupling, so we drop it from the conduction list and reroute
    # it into air_zone_wall_neighbors below.
    solid_thermal_neighbors = {n: [] for n in nodes}
    air_zone_wall_neighbors = {z: [] for z in air_zones}
    wall_zone = {}
    for (a, b), k_th in link_conductance.items():
        a_is_air = node_type[a] == AIR_ZONE_TYPE
        b_is_air = node_type[b] == AIR_ZONE_TYPE
        a_is_wall = node_type[a] in BOUNDARY_NODE_TYPES
        b_is_wall = node_type[b] in BOUNDARY_NODE_TYPES
        if a_is_air and b_is_wall:
            air_zone_wall_neighbors[a].append((b, k_th))
            wall_zone.setdefault(b, a)
            continue
        if b_is_air and a_is_wall:
            air_zone_wall_neighbors[b].append((a, k_th))
            wall_zone.setdefault(a, b)
            continue
        if a_is_air or b_is_air:
            # Pure solid<->air conduction is folded into convection via solid_air_zone.
            continue
        solid_thermal_neighbors[a].append((b, k_th))
        solid_thermal_neighbors[b].append((a, k_th))

    # solid -> air zone: pick the air_zone neighbour with the highest a_air weight.
    solid_air_zone = {}
    for node in nodes:
        if node_type[node] == AIR_ZONE_TYPE:
            continue
        best = None
        for nbr, a_air, _d_gas, _flow_dir in air_neighbors.get(node, []):
            if node_type[nbr] != AIR_ZONE_TYPE:
                continue
            if best is None or a_air > best[1]:
                best = (nbr, a_air)
        if best is not None:
            solid_air_zone[node] = best[0]
        elif node in wall_zone:
            solid_air_zone[node] = wall_zone[node]

    # Air-zone reverse member list (only true solid components — boundaries are handled separately).
    air_zone_members = {z: [] for z in air_zones}
    for node, z in solid_air_zone.items():
        if node_type[node] in BOUNDARY_NODE_TYPES:
            continue
        air_zone_members[z].append(node)

    # Air-zone to air-zone advection links and to-boundary outlet weights.
    air_zone_air_neighbors = {z: [] for z in air_zones}
    air_zone_outlet_a = {z: 0.0 for z in air_zones}
    seen_zone_pairs = set()
    for zone in air_zones:
        for nbr, a_air, _d_gas, flow_dir in air_neighbors.get(zone, []):
            if node_type[nbr] == AIR_ZONE_TYPE:
                key = tuple(sorted((zone, nbr)))
                if key in seen_zone_pairs:
                    continue
                seen_zone_pairs.add(key)
                air_zone_air_neighbors[zone].append((nbr, float(a_air), flow_dir))
                air_zone_air_neighbors[nbr].append((zone, float(a_air), flow_dir))
            elif node_type[nbr] in BOUNDARY_NODE_TYPES:
                air_zone_outlet_a[zone] += float(a_air)

    return {
        'solid_thermal_neighbors': solid_thermal_neighbors,
        'solid_air_zone': solid_air_zone,
        'air_zone_members': air_zone_members,
        'air_zone_air_neighbors': air_zone_air_neighbors,
        'air_zone_outlet_a': air_zone_outlet_a,
        'air_zone_nodes': air_zones,
        'cabinet_boundary_nodes': boundary_nodes,
        'air_zone_wall_neighbors': air_zone_wall_neighbors,
    }


def build_graph():
    components = pd.read_csv(CONFIG_DIR / 'component_table.csv')
    electrical_edges = pd.read_csv(CONFIG_DIR / 'electrical_edges.csv')
    thermal_edges = pd.read_csv(CONFIG_DIR / 'thermal_edges.csv')
    air_edges = pd.read_csv(CONFIG_DIR / 'air_edges.csv')

    component_nodes = set(components['node'])
    _validate_edges('electrical_edges.csv', electrical_edges, component_nodes)
    _validate_edges('thermal_edges.csv', thermal_edges, component_nodes)
    _validate_edges('air_edges.csv', air_edges, component_nodes)

    node_pos = {
        row.node: (float(row.x_mm), float(row.y_mm), float(row.z_mm))
        for row in components.itertuples(index=False)
    }
    nodes = list(node_pos.keys())
    copper_links = [
        (row.src, row.dst)
        for row in electrical_edges.itertuples(index=False)
    ]
    link_conductance = {
        (row.src, row.dst): float(row.k_th)
        for row in thermal_edges.itertuples(index=False)
        if float(row.k_th) > 0.0
    }
    spread_link_factor = {
        (row.src, row.dst): max(0.70, min(1.60, 0.75 + 2.0 * float(row.view_factor)))
        for row in thermal_edges.itertuples(index=False)
        if float(row.k_th) > 0.0
    }

    node_type = {row.node: row.node_type for row in components.itertuples(index=False)}
    fault_source_level = {row.node: row.fault_source_level for row in components.itertuples(index=False)}
    surface_area = {row.node: float(row.surface_area) for row in components.itertuples(index=False)}
    conv_hA = {row.node: float(row.conv_hA) for row in components.itertuples(index=False)}
    c_th = {row.node: float(row.C_th) for row in components.itertuples(index=False)}
    node_resistance = {row.node: float(row.base_resistance_ohm) for row in components.itertuples(index=False)}
    series_path_nodes = [row.node for row in components.itertuples(index=False) if int(row.is_series_path) == 1]
    fuel_mass_total = {row.node: float(row.fuel_mass) for row in components.itertuples(index=False)}
    burn_base_rate = {row.node: float(row.burn_rate) for row in components.itertuples(index=False)}
    spread_material_factor = {row.node: float(row.spread_material_factor) for row in components.itertuples(index=False)}
    spread_heat_threshold = {row.node: float(row.spread_heat_threshold) for row in components.itertuples(index=False)}
    spread_exposure_threshold = {row.node: float(row.spread_exposure_threshold) for row in components.itertuples(index=False)}

    edge_map = {}
    for row in electrical_edges.itertuples(index=False):
        key = (row.src, row.dst)
        item = edge_map.setdefault(key, {
            'src': row.src,
            'dst': row.dst,
            'edge_type': [],
            'distance_mm': _distance_mm(node_pos, row.src, row.dst),
            'k_th': 0.0,
            'g_elec': 0.0,
            'a_air': 0.0,
            'R_line': None,
            'D_gas': None,
            'topology_weight': 0.0,
        })
        item['edge_type'].append('electrical')
        item['R_line'] = float(row.R_line)
        item['g_elec'] = 1.0 / max(float(row.R_line), 1e-6)
        item['topology_weight'] = max(item['topology_weight'], 1.0)

    for row in thermal_edges.itertuples(index=False):
        key = (row.src, row.dst)
        item = edge_map.setdefault(key, {
            'src': row.src,
            'dst': row.dst,
            'edge_type': [],
            'distance_mm': _distance_mm(node_pos, row.src, row.dst),
            'k_th': 0.0,
            'g_elec': 0.0,
            'a_air': 0.0,
            'R_line': None,
            'D_gas': None,
            'topology_weight': 0.0,
        })
        item['edge_type'].append('thermal')
        item['distance_mm'] = float(row.d_mm) if float(row.d_mm) > 0.0 else item['distance_mm']
        item['k_th'] = float(row.k_th)
        item['view_factor'] = float(row.view_factor)
        item['topology_weight'] = max(item['topology_weight'], float(row.view_factor))

    air_neighbors = {node: [] for node in nodes}
    for row in air_edges.itertuples(index=False):
        key = (row.src, row.dst)
        item = edge_map.setdefault(key, {
            'src': row.src,
            'dst': row.dst,
            'edge_type': [],
            'distance_mm': _distance_mm(node_pos, row.src, row.dst),
            'k_th': 0.0,
            'g_elec': 0.0,
            'a_air': 0.0,
            'R_line': None,
            'D_gas': None,
            'topology_weight': 0.0,
        })
        item['edge_type'].append('air')
        item['a_air'] = float(row.a_air)
        item['D_gas'] = float(row.D_gas)
        item['flow_direction'] = row.flow_direction
        item['topology_weight'] = max(item['topology_weight'], float(row.a_air))
        air_neighbors[row.src].append((row.dst, float(row.a_air), float(row.D_gas), row.flow_direction))
        air_neighbors[row.dst].append((row.src, float(row.a_air), float(row.D_gas), 'reverse_' + row.flow_direction))

    graph_edges = []
    for item in edge_map.values():
        item = item.copy()
        item['edge_type'] = '+'.join(dict.fromkeys(item['edge_type']))
        graph_edges.append(item)

    electrical_edges = electrical_edges.copy()
    electrical_edges['g_elec'] = electrical_edges['R_line'].apply(lambda value: 1.0 / max(float(value), 1e-6))
    electrical_edge_rows = electrical_edges.to_dict('records')
    thermal_edge_rows = thermal_edges.to_dict('records')
    air_edge_rows = air_edges.to_dict('records')

    thermal_topology = _build_thermal_topology(nodes, node_type, link_conductance, air_neighbors)

    return {
        'node_pos': node_pos,
        'nodes': nodes,
        'node_type': node_type,
        'fault_source_level': fault_source_level,
        'copper_links': copper_links,
        'link_conductance': link_conductance,
        'graph_edges': graph_edges,
        'electrical_edges': electrical_edge_rows,
        'thermal_edges': thermal_edge_rows,
        'air_edges': air_edge_rows,
        'air_neighbors': air_neighbors,
        'surface_area': surface_area,
        'conv_hA': conv_hA,
        'C_th': c_th,
        'node_resistance': node_resistance,
        'series_path_nodes': series_path_nodes,
        'fuel_mass_total': fuel_mass_total,
        'burn_base_rate': burn_base_rate,
        'spread_material_factor': spread_material_factor,
        'spread_heat_threshold': spread_heat_threshold,
        'spread_exposure_threshold': spread_exposure_threshold,
        'spread_link_factor': spread_link_factor,
        'solid_thermal_neighbors': thermal_topology['solid_thermal_neighbors'],
        'solid_air_zone': thermal_topology['solid_air_zone'],
        'air_zone_members': thermal_topology['air_zone_members'],
        'air_zone_air_neighbors': thermal_topology['air_zone_air_neighbors'],
        'air_zone_outlet_a': thermal_topology['air_zone_outlet_a'],
        'air_zone_nodes': thermal_topology['air_zone_nodes'],
        'cabinet_boundary_nodes': thermal_topology['cabinet_boundary_nodes'],
        'air_zone_wall_neighbors': thermal_topology['air_zone_wall_neighbors'],
    }

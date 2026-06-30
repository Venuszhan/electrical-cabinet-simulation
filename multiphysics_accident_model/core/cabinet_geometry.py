from pathlib import Path

import pandas as pd


PACKAGE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = PACKAGE_DIR / 'config'


def load_cabinet_table():
    table = pd.read_csv(CONFIG_DIR / 'cabinet_table.csv')
    raw = {row.key: row.value for row in table.itertuples(index=False)}
    return {
        'width_mm': float(raw['width_mm']),
        'depth_mm': float(raw['depth_mm']),
        'height_mm': float(raw['height_mm']),
        'material': raw['material'],
        'ground_resistance_max_ohm': float(raw['ground_resistance_max_ohm']),
    }, float(raw['vent_area_m2']), float(raw['ambient_temp_c'])


def load_zone_table():
    table = pd.read_csv(CONFIG_DIR / 'zone_table.csv')
    zones = {}
    for row in table.itertuples(index=False):
        zones[row.zone_name] = {
            'bounds_mm': {
                'x_min': float(row.x_min_mm),
                'x_max': float(row.x_max_mm),
                'y_min': float(row.y_min_mm),
                'y_max': float(row.y_max_mm),
                'z_min': float(row.z_min_mm),
                'z_max': float(row.z_max_mm),
            },
            'zone_type': row.zone_type,
            'primary_role': row.primary_role,
            'source_doc': row.source_doc,
            'note': row.note,
        }
    return zones


def find_zones_for_point(zones, point_mm):
    x_mm, y_mm, z_mm = point_mm
    matched = []
    for name, zone in zones.items():
        bounds = zone['bounds_mm']
        if (
            bounds['x_min'] <= x_mm <= bounds['x_max']
            and bounds['y_min'] <= y_mm <= bounds['y_max']
            and bounds['z_min'] <= z_mm <= bounds['z_max']
        ):
            matched.append(name)
    return matched


CABINET_DIMS, VENT_AREA_M2, AMBIENT_TEMP_C = load_cabinet_table()
CABINET_ZONES = load_zone_table()


def get_dist(sim, n1, n2):
    p1, p2 = sim.node_pos[n1], sim.node_pos[n2]
    return sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5

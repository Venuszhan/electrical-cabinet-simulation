import numpy as np


NODE_TYPE_SENSITIVITY = {
    'switch': 1.00,
    'breaker': 1.00,
    'contactor': 1.18,
    'contactor_terminal': 1.28,
    'protector': 0.95,
    'relay': 0.92,
    'terminal': 1.22,
    'connector': 1.25,
    'harness': 1.16,
    'ground': 0.70,
    'air_zone': 0.40,
    'cabinet_boundary': 0.45,
    'ventilation': 0.70,
}


class AgingProfile:
    def __init__(self, service_years, node_type, rng=None):
        self.tau = float(np.clip(service_years, 0.0, 60.0))
        self.node_type = node_type
        self.rng = rng

        self.eta_contact = 1 - np.exp(-self.tau / 18.0)
        self.eta_oxide = 1 - np.exp(-self.tau / 10.0)
        self.eta_insulation = 1 - np.exp(-(self.tau / 25.0) ** 1.4)
        self.eta_fuel = 1 - np.exp(-(self.tau / 30.0) ** 1.2)

        self.eta_total = (
            0.30 * self.eta_contact
            + 0.25 * self.eta_oxide
            + 0.30 * self.eta_insulation
            + 0.15 * self.eta_fuel
        )
        self.type_sensitivity = NODE_TYPE_SENSITIVITY.get(node_type, 1.0)
        self.eta_effective = float(np.clip(self.eta_total * self.type_sensitivity, 0.0, 0.98))

        self.Ea = self.compute_Ea()

    def compute_Ea(self):
        # Cable combustion tests show delayed ignition after thermal aging.
        # Keep the increase conservative because service years are not the
        # accelerated-aging days used by the transferable experiments.
        Ea0 = 135e3
        return Ea0 + 5e3 * self.eta_insulation * self.type_sensitivity

    @property
    def contact_pressure_rate_multiplier(self):
        return 1.0 + 2.4 * self.eta_contact * self.type_sensitivity

    @property
    def oxide_growth_multiplier(self):
        return 1.0 + 2.8 * self.eta_oxide * self.type_sensitivity

    @property
    def contact_resistance_multiplier(self):
        return 1.0 + 1.6 * self.eta_contact + 2.0 * self.eta_oxide * self.type_sensitivity

    @property
    def pyrolysis_rate_multiplier(self):
        return 1.0

    @property
    def ignition_temp_rise_c(self):
        return 15.0 * self.eta_insulation + 5.0 * self.eta_fuel

    @property
    def gas_threshold_multiplier(self):
        return float(np.clip(1.0 + 0.10 * self.eta_insulation, 1.0, 1.10))

    @property
    def arc_energy_multiplier(self):
        # No transferred experiment resolves arc-to-flame energy thresholds.
        return 1.0

    @property
    def combustion_rate_multiplier(self):
        return float(np.clip(1.0 - 0.20 * self.eta_insulation, 0.80, 1.0))

    def to_dict(self):
        return {
            'service_years': self.tau,
            'node_type': self.node_type,
            'eta_contact': self.eta_contact,
            'eta_oxide': self.eta_oxide,
            'eta_insulation': self.eta_insulation,
            'eta_fuel': self.eta_fuel,
            'eta_total': self.eta_total,
            'eta_effective': self.eta_effective,
            'Ea_J_per_mol': self.Ea,
            'contact_pressure_rate_multiplier': self.contact_pressure_rate_multiplier,
            'oxide_growth_multiplier': self.oxide_growth_multiplier,
            'contact_resistance_multiplier': self.contact_resistance_multiplier,
            'pyrolysis_rate_multiplier': self.pyrolysis_rate_multiplier,
            'ignition_temp_rise_c': self.ignition_temp_rise_c,
            'gas_threshold_multiplier': self.gas_threshold_multiplier,
            'arc_energy_multiplier': self.arc_energy_multiplier,
            'combustion_rate_multiplier': self.combustion_rate_multiplier,
        }


def init_aging_parameters(sim, service_days=None, service_years=None, rng=None):
    if service_years is None:
        service_days = 30 if service_days is None else service_days
        service_years = float(service_days) / 365.0

    sim.service_years = float(np.clip(service_years, 0.0, 60.0))
    sim.service_days = sim.service_years * 365.0
    sim.aging_profiles = {
        node: AgingProfile(sim.service_years, sim.node_type[node], rng)
        for node in sim.nodes
    }
    sim.aging_profile = sim.aging_profiles[sim.f_node]
    sim.gamma = sim.aging_profile.eta_total
    sim.Ea_ref = sim.aging_profile.Ea / 1000.0
    sim.A_pyro_base = 2.2e10
    sim.A_pyro = sim.A_pyro_base * sim.aging_profile.pyrolysis_rate_multiplier


def aged_initial_resistance(sim, node):
    base_resistance = sim.node_resistance[node]
    if base_resistance <= 0.0:
        return 0.0
    return base_resistance * sim.aging_profiles[node].contact_resistance_multiplier


def apply_fault_aging_effects(sim):
    profile = sim.aging_profile
    sim.R_metal = sim.R_metal_base * profile.contact_resistance_multiplier
    sim.T_ig = min(420.0, sim.T_ig + profile.ignition_temp_rise_c)
    sim.gas_ignition_threshold *= profile.gas_threshold_multiplier
    sim.gas_spread_threshold *= profile.gas_threshold_multiplier
    sim.arc_energy_zone_low *= profile.arc_energy_multiplier
    sim.arc_energy_zone_high *= profile.arc_energy_multiplier

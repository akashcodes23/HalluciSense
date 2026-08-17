"""Phase 8A — Scientific Adversarial Benchmark Dataset Builder.

Constructs N=175 scientific adversarial claims across:
  5 domains × 7 categories × 5 samples = 175

Domains: Physics, Chemistry, Biology, Medicine, Mathematics

Categories:
  TRUE_CONTROL                  — factual correct claim (GT=0)
  NUMERICAL_PRECISION           — wrong number in otherwise factual claim (GT=1)
  UNIT_SCALE                    — wrong unit or scale (GT=1)
  NEGATION                      — negated truth (GT=1)
  CAUSAL_INVERSION              — causal direction reversed (GT=1)
  OUTDATED_SCIENTIFIC_CLAIM     — superseded/outdated fact (GT=1)
  TRUE_CORE_FALSE_ELABORATION   — factual core + fabricated specific detail (GT=1)

Ground truth comes from cited authoritative references.
Ground truth is NEVER derived from running HalluciSense.
"""

from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "reports" / "phase8" / "8A"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOMAINS = ["Physics", "Chemistry", "Biology", "Medicine", "Mathematics"]
CATEGORIES = [
    "TRUE_CONTROL",
    "NUMERICAL_PRECISION",
    "UNIT_SCALE",
    "NEGATION",
    "CAUSAL_INVERSION",
    "OUTDATED_SCIENTIFIC_CLAIM",
    "TRUE_CORE_FALSE_ELABORATION",
]

# fmt: off
# Each entry: (domain, category, claim, ground_truth, difficulty,
#              source, source_url, source_type, provenance)
CLAIMS = [

    # ══════════════════════════════════════════════════════════════════════
    # PHYSICS
    # ══════════════════════════════════════════════════════════════════════

    # TRUE_CONTROL — Physics (GT=0)
    ("Physics","TRUE_CONTROL","The speed of light in a vacuum is approximately 3×10⁸ metres per second.",
     0,"easy","Wikipedia: Speed of light",
     "https://en.wikipedia.org/wiki/Speed_of_light","encyclopedia",
     "Exact value 299,792,458 m/s; commonly approximated as 3×10⁸ m/s. Factual."),

    ("Physics","TRUE_CONTROL","An electron has a rest mass of approximately 9.11×10⁻³¹ kilograms.",
     0,"easy","NIST CODATA 2018",
     "https://physics.nist.gov/cgi-bin/cuu/Value?me","standards_body",
     "CODATA recommended value: 9.1093837015×10⁻³¹ kg. Factual."),

    ("Physics","TRUE_CONTROL","The gravitational acceleration at Earth's surface is approximately 9.8 m/s².",
     0,"easy","Wikipedia: Gravitational acceleration",
     "https://en.wikipedia.org/wiki/Gravitational_acceleration","encyclopedia",
     "Standard gravity g=9.80665 m/s². Commonly stated as 9.8 m/s². Factual."),

    ("Physics","TRUE_CONTROL","Planck's constant has a value of approximately 6.626×10⁻³⁴ joule-seconds.",
     0,"medium","NIST CODATA 2018",
     "https://physics.nist.gov/cgi-bin/cuu/Value?h","standards_body",
     "Exact defined value: 6.62607015×10⁻³⁴ J·s. Factual."),

    ("Physics","TRUE_CONTROL","The wavelength of visible light ranges from approximately 380 nm to 740 nm.",
     0,"easy","Wikipedia: Visible spectrum",
     "https://en.wikipedia.org/wiki/Visible_spectrum","encyclopedia",
     "Visible range: ~380–740 nm. Factual."),

    # NUMERICAL_PRECISION — Physics (GT=1, wrong number)
    ("Physics","NUMERICAL_PRECISION","The speed of light in a vacuum is approximately 3×10⁶ metres per second.",
     1,"medium","Wikipedia: Speed of light",
     "https://en.wikipedia.org/wiki/Speed_of_light","encyclopedia",
     "Correct value is ~3×10⁸ m/s. Claim states 10⁶ — off by two orders of magnitude."),

    ("Physics","NUMERICAL_PRECISION","An electron has a rest mass of approximately 9.11×10⁻²⁷ kilograms.",
     1,"hard","NIST CODATA 2018",
     "https://physics.nist.gov/cgi-bin/cuu/Value?me","standards_body",
     "Correct exponent is 10⁻³¹. Claim uses 10⁻²⁷ — four orders of magnitude error."),

    ("Physics","NUMERICAL_PRECISION","Planck's constant is approximately 6.626×10⁻³⁰ joule-seconds.",
     1,"hard","NIST CODATA 2018",
     "https://physics.nist.gov/cgi-bin/cuu/Value?h","standards_body",
     "Correct exponent is 10⁻³⁴. Claim states 10⁻³⁰ — four orders of magnitude error."),

    ("Physics","NUMERICAL_PRECISION","The gravitational acceleration at Earth's surface is approximately 19.6 m/s².",
     1,"easy","Wikipedia: Gravitational acceleration",
     "https://en.wikipedia.org/wiki/Gravitational_acceleration","encyclopedia",
     "Standard g≈9.8 m/s². Claim doubles the correct value."),

    ("Physics","NUMERICAL_PRECISION","Visible light has a wavelength range of approximately 200 nm to 400 nm.",
     1,"medium","Wikipedia: Visible spectrum",
     "https://en.wikipedia.org/wiki/Visible_spectrum","encyclopedia",
     "Visible range is 380–740 nm. The claim describes UV, not visible light."),

    # UNIT_SCALE — Physics (GT=1)
    ("Physics","UNIT_SCALE","The mass of the electron is approximately 9.11×10⁻³¹ grams.",
     1,"medium","NIST CODATA 2018",
     "https://physics.nist.gov/cgi-bin/cuu/Value?me","standards_body",
     "Correct unit is kilograms, not grams. 9.11×10⁻³¹ grams = 9.11×10⁻³⁴ kg — wrong by 10³."),

    ("Physics","UNIT_SCALE","Planck's constant is approximately 6.626×10⁻³⁴ electron-volts per second.",
     1,"hard","NIST CODATA 2018",
     "https://physics.nist.gov/cgi-bin/cuu/Value?h","standards_body",
     "Planck's constant is in J·s. In eV·s it is ~4.136×10⁻¹⁵ eV·s. Claim is dimensionally wrong."),

    ("Physics","UNIT_SCALE","The speed of light is approximately 3×10⁸ kilometres per second.",
     1,"medium","Wikipedia: Speed of light",
     "https://en.wikipedia.org/wiki/Speed_of_light","encyclopedia",
     "Correct unit is metres per second, not kilometres per second."),

    ("Physics","UNIT_SCALE","Standard gravitational acceleration is approximately 9.8 cm/s².",
     1,"easy","Wikipedia: Gravitational acceleration",
     "https://en.wikipedia.org/wiki/Gravitational_acceleration","encyclopedia",
     "g≈9.8 m/s², not cm/s². Claim is off by a factor of 100."),

    ("Physics","UNIT_SCALE","Visible light wavelengths range from 380 to 740 micrometres.",
     1,"medium","Wikipedia: Visible spectrum",
     "https://en.wikipedia.org/wiki/Visible_spectrum","encyclopedia",
     "Correct unit is nanometres (nm). Micrometres (μm) would be infrared range."),

    # NEGATION — Physics (GT=1)
    ("Physics","NEGATION","The speed of light is not constant in a vacuum.",
     1,"easy","Wikipedia: Speed of light",
     "https://en.wikipedia.org/wiki/Speed_of_light","encyclopedia",
     "The speed of light is constant in a vacuum (c = 299,792,458 m/s exactly). Claim negates this."),

    ("Physics","NEGATION","Photons do not have zero rest mass.",
     1,"easy","Wikipedia: Photon",
     "https://en.wikipedia.org/wiki/Photon","encyclopedia",
     "Photons have zero rest mass. The claim negates this established fact."),

    ("Physics","NEGATION","The gravitational force between two masses does not depend on the distance between them.",
     1,"medium","Wikipedia: Newton's law of universal gravitation",
     "https://en.wikipedia.org/wiki/Newton%27s_law_of_universal_gravitation","encyclopedia",
     "Gravitational force is inversely proportional to the square of distance. The negation is false."),

    ("Physics","NEGATION","Electrons do not carry a negative electric charge.",
     1,"easy","Wikipedia: Electron",
     "https://en.wikipedia.org/wiki/Electron","encyclopedia",
     "Electrons carry a negative charge of −1.602×10⁻¹⁹ C. Claim is false."),

    ("Physics","NEGATION","An object at rest does not tend to stay at rest unless acted upon by a net force.",
     1,"easy","Wikipedia: Newton's laws of motion",
     "https://en.wikipedia.org/wiki/Newton%27s_laws_of_motion","encyclopedia",
     "Newton's first law states objects at rest stay at rest unless acted upon. Negation is false."),

    # CAUSAL_INVERSION — Physics (GT=1)
    ("Physics","CAUSAL_INVERSION","An increase in temperature causes a decrease in the thermal energy of an ideal gas.",
     1,"medium","Wikipedia: Kinetic theory of gases",
     "https://en.wikipedia.org/wiki/Kinetic_theory_of_gases","encyclopedia",
     "Temperature increase causes INCREASE, not decrease, in thermal energy. Causal direction reversed."),

    ("Physics","CAUSAL_INVERSION","Higher frequency electromagnetic radiation results in lower photon energy.",
     1,"medium","Wikipedia: Photon energy",
     "https://en.wikipedia.org/wiki/Photon_energy","encyclopedia",
     "E=hf: higher frequency → higher energy. Claim reverses the causal direction."),

    ("Physics","CAUSAL_INVERSION","Stronger magnetic fields cause slower charged particle precession in NMR.",
     1,"hard","Wikipedia: Nuclear magnetic resonance",
     "https://en.wikipedia.org/wiki/Nuclear_magnetic_resonance","encyclopedia",
     "NMR precession frequency (Larmor) increases with field strength. Direction reversed."),

    ("Physics","CAUSAL_INVERSION","Increasing electrical resistance in a circuit causes current to increase when voltage is constant.",
     1,"easy","Wikipedia: Ohm's law",
     "https://en.wikipedia.org/wiki/Ohm%27s_law","encyclopedia",
     "I=V/R: more resistance → less current. Claim inverts Ohm's law."),

    ("Physics","CAUSAL_INVERSION","Compressing a gas adiabatically causes its temperature to decrease.",
     1,"medium","Wikipedia: Adiabatic process",
     "https://en.wikipedia.org/wiki/Adiabatic_process","encyclopedia",
     "Adiabatic compression raises temperature. Claim states the opposite."),

    # OUTDATED_SCIENTIFIC_CLAIM — Physics (GT=1)
    ("Physics","OUTDATED_SCIENTIFIC_CLAIM","The atom is the smallest indivisible unit of matter.",
     1,"easy","Wikipedia: Atom",
     "https://en.wikipedia.org/wiki/Atom","encyclopedia",
     "Atoms are divisible into protons, neutrons, and electrons; these into quarks. Outdated 19th-century claim."),

    ("Physics","OUTDATED_SCIENTIFIC_CLAIM","The luminiferous ether is the medium through which light propagates.",
     1,"easy","Wikipedia: Luminiferous aether",
     "https://en.wikipedia.org/wiki/Luminiferous_aether","encyclopedia",
     "Ether concept was experimentally disproved by Michelson-Morley (1887) and superseded by special relativity."),

    ("Physics","OUTDATED_SCIENTIFIC_CLAIM","Newtonian mechanics provides a complete description of motion at all speeds and scales.",
     1,"medium","Wikipedia: Newton's laws of motion",
     "https://en.wikipedia.org/wiki/Newton%27s_laws_of_motion","encyclopedia",
     "Superseded by special relativity at high speeds and quantum mechanics at small scales."),

    ("Physics","OUTDATED_SCIENTIFIC_CLAIM","The universe exists in a steady state with no beginning or end.",
     1,"medium","Wikipedia: Steady State theory",
     "https://en.wikipedia.org/wiki/Steady-state_model","encyclopedia",
     "Steady state theory replaced by Big Bang cosmology following CMB discovery (1965)."),

    ("Physics","OUTDATED_SCIENTIFIC_CLAIM","Heat is a fluid substance called caloric that flows from hot to cold objects.",
     1,"easy","Wikipedia: Caloric theory",
     "https://en.wikipedia.org/wiki/Caloric_theory","encyclopedia",
     "Caloric theory disproved by Rumford (1798) and Joule (1843); heat is energy transfer."),

    # TRUE_CORE_FALSE_ELABORATION — Physics (GT=1)
    ("Physics","TRUE_CORE_FALSE_ELABORATION","The photoelectric effect, discovered by Albert Einstein in 1905, shows that electrons are emitted from metal surfaces when exposed to UV light of any frequency.",
     1,"medium","Wikipedia: Photoelectric effect",
     "https://en.wikipedia.org/wiki/Photoelectric_effect","encyclopedia",
     "Core true: Einstein explained photoelectric effect (1905). Fabrication: requires frequency ABOVE a threshold, not any frequency."),

    ("Physics","TRUE_CORE_FALSE_ELABORATION","Special relativity, formulated by Einstein in 1905, predicts that mass increases with velocity and that objects can exceed the speed of light with sufficient energy.",
     1,"hard","Wikipedia: Special relativity",
     "https://en.wikipedia.org/wiki/Special_relativity","encyclopedia",
     "Core true: SR formulated 1905. Fabrication: objects with mass cannot reach, let alone exceed, c."),

    ("Physics","TRUE_CORE_FALSE_ELABORATION","Superconductivity, observed in some materials below a critical temperature, allows electrical resistance to become exactly zero and also increases magnetic permeability.",
     1,"hard","Wikipedia: Superconductivity",
     "https://en.wikipedia.org/wiki/Superconductivity","encyclopedia",
     "Core true: zero resistance below Tc. Fabrication: superconductors EXPEL magnetic fields (Meissner effect), not increase permeability."),

    ("Physics","TRUE_CORE_FALSE_ELABORATION","Black holes, regions of spacetime where gravity prevents even light from escaping, were first observed directly by the Hubble Space Telescope in 1995.",
     1,"medium","Wikipedia: Black hole",
     "https://en.wikipedia.org/wiki/Black_hole","encyclopedia",
     "Core true: black holes prevent light escape. Fabrication: first image by Event Horizon Telescope in 2019, not Hubble in 1995."),

    ("Physics","TRUE_CORE_FALSE_ELABORATION","Nuclear fission, the process of splitting heavy nuclei to release energy, was first demonstrated by Marie Curie in 1934.",
     1,"medium","Wikipedia: Nuclear fission",
     "https://en.wikipedia.org/wiki/Nuclear_fission","encyclopedia",
     "Core true: fission releases energy. Fabrication: first demonstrated by Hahn, Strassmann, Meitner, Frisch in 1938–1939, not Curie."),

    # ══════════════════════════════════════════════════════════════════════
    # CHEMISTRY
    # ══════════════════════════════════════════════════════════════════════

    ("Chemistry","TRUE_CONTROL","Water has a molecular formula of H₂O and a molar mass of approximately 18 g/mol.",
     0,"easy","Wikipedia: Water",
     "https://en.wikipedia.org/wiki/Water","encyclopedia",
     "H₂O, molar mass 18.015 g/mol. Factual."),

    ("Chemistry","TRUE_CONTROL","The pH of a neutral aqueous solution at 25°C is 7.0.",
     0,"easy","Wikipedia: pH",
     "https://en.wikipedia.org/wiki/PH","encyclopedia",
     "At 25°C, pure water has pH=7.00. Factual."),

    ("Chemistry","TRUE_CONTROL","Carbon-12 is the most abundant stable isotope of carbon.",
     0,"easy","Wikipedia: Carbon-12",
     "https://en.wikipedia.org/wiki/Carbon-12","encyclopedia",
     "¹²C constitutes ~98.9% of natural carbon. Factual."),

    ("Chemistry","TRUE_CONTROL","The Avogadro constant is approximately 6.022×10²³ per mole.",
     0,"easy","NIST CODATA 2018",
     "https://physics.nist.gov/cgi-bin/cuu/Value?na","standards_body",
     "Exact: 6.02214076×10²³ mol⁻¹. Factual."),

    ("Chemistry","TRUE_CONTROL","Sodium chloride (NaCl) dissociates completely into Na⁺ and Cl⁻ ions when dissolved in water.",
     0,"easy","Wikipedia: Sodium chloride",
     "https://en.wikipedia.org/wiki/Sodium_chloride","encyclopedia",
     "NaCl is a strong electrolyte that fully dissociates in aqueous solution. Factual."),

    ("Chemistry","NUMERICAL_PRECISION","Water has a molar mass of approximately 36 g/mol.",
     1,"easy","Wikipedia: Water",
     "https://en.wikipedia.org/wiki/Water","encyclopedia",
     "Correct molar mass is ~18 g/mol (2×1 + 16). Claim doubles the value."),

    ("Chemistry","NUMERICAL_PRECISION","The Avogadro constant is approximately 6.022×10²⁰ per mole.",
     1,"medium","NIST CODATA 2018",
     "https://physics.nist.gov/cgi-bin/cuu/Value?na","standards_body",
     "Correct value is 6.022×10²³. Claim is wrong by three orders of magnitude."),

    ("Chemistry","NUMERICAL_PRECISION","The boiling point of water at standard pressure is 212°C.",
     1,"easy","Wikipedia: Boiling point",
     "https://en.wikipedia.org/wiki/Boiling_point","encyclopedia",
     "Water boils at 100°C (212°F). Claim confuses Fahrenheit with Celsius."),

    ("Chemistry","NUMERICAL_PRECISION","The atomic number of gold is 97.",
     1,"easy","Wikipedia: Gold",
     "https://en.wikipedia.org/wiki/Gold","encyclopedia",
     "Gold (Au) has atomic number 79, not 97 (which is Berkelium)."),

    ("Chemistry","NUMERICAL_PRECISION","Carbon dioxide has a molar mass of approximately 44 kg/mol.",
     1,"medium","Wikipedia: Carbon dioxide",
     "https://en.wikipedia.org/wiki/Carbon_dioxide","encyclopedia",
     "CO₂ molar mass is 44.01 g/mol, not kg/mol. Unit and magnitude both wrong."),

    ("Chemistry","UNIT_SCALE","The boiling point of water at standard pressure is 100 K.",
     1,"medium","Wikipedia: Boiling point",
     "https://en.wikipedia.org/wiki/Boiling_point","encyclopedia",
     "Water boils at 100°C = 373.15 K. Claim states 100 K (−173°C), which is cryogenic."),

    ("Chemistry","UNIT_SCALE","Standard atomic weight of iron is approximately 55.8 kg/mol.",
     1,"medium","Wikipedia: Iron",
     "https://en.wikipedia.org/wiki/Iron","encyclopedia",
     "Iron molar mass is 55.845 g/mol. Claim gives kg/mol — a factor of 1000 error."),

    ("Chemistry","UNIT_SCALE","The concentration of a 1 M NaCl solution contains 1 gram of NaCl per litre.",
     1,"easy","Wikipedia: Molar concentration",
     "https://en.wikipedia.org/wiki/Molar_concentration","encyclopedia",
     "1 M NaCl = 58.44 g/L (1 mole NaCl), not 1 gram."),

    ("Chemistry","UNIT_SCALE","The density of water is approximately 1 kg/cm³.",
     1,"easy","Wikipedia: Properties of water",
     "https://en.wikipedia.org/wiki/Properties_of_water","encyclopedia",
     "Density is 1 g/cm³ = 1 kg/L = 1000 kg/m³. Claim is off by 10³."),

    ("Chemistry","UNIT_SCALE","Hydrochloric acid with a pH of 1 has a hydrogen ion concentration of 1 mM.",
     1,"medium","Wikipedia: pH",
     "https://en.wikipedia.org/wiki/PH","encyclopedia",
     "pH=1 means [H⁺]=0.1 mol/L = 100 mM, not 1 mM."),

    ("Chemistry","NEGATION","Acids do not donate protons to bases in Brønsted–Lowry acid-base theory.",
     1,"easy","Wikipedia: Brønsted–Lowry acid–base theory",
     "https://en.wikipedia.org/wiki/Br%C3%B8nsted%E2%80%93Lowry_acid%E2%80%93base_theory","encyclopedia",
     "Brønsted–Lowry definition: acids ARE proton donors. Negation is false."),

    ("Chemistry","NEGATION","Covalent bonds are not formed by sharing electrons between atoms.",
     1,"easy","Wikipedia: Covalent bond",
     "https://en.wikipedia.org/wiki/Covalent_bond","encyclopedia",
     "Covalent bonds ARE formed by electron sharing. Negation is false."),

    ("Chemistry","NEGATION","Oxidation does not involve the loss of electrons.",
     1,"easy","Wikipedia: Redox",
     "https://en.wikipedia.org/wiki/Redox","encyclopedia",
     "Oxidation is defined as electron loss (OIL RIG). Negation is false."),

    ("Chemistry","NEGATION","Noble gases do not have a complete outer electron shell.",
     1,"easy","Wikipedia: Noble gas",
     "https://en.wikipedia.org/wiki/Noble_gas","encyclopedia",
     "Noble gases have complete valence shells, which is why they are chemically inert. Negation is false."),

    ("Chemistry","NEGATION","Exothermic reactions do not release heat to the surroundings.",
     1,"easy","Wikipedia: Exothermic process",
     "https://en.wikipedia.org/wiki/Exothermic_process","encyclopedia",
     "Exothermic reactions by definition release heat. Negation is false."),

    ("Chemistry","CAUSAL_INVERSION","Adding a catalyst to a reaction slows the forward reaction rate by increasing the activation energy.",
     1,"easy","Wikipedia: Catalysis",
     "https://en.wikipedia.org/wiki/Catalysis","encyclopedia",
     "Catalysts lower activation energy and speed reactions. Both causal direction and effect are inverted."),

    ("Chemistry","CAUSAL_INVERSION","Higher temperatures cause reaction rates to decrease because molecules have lower average kinetic energy.",
     1,"medium","Wikipedia: Arrhenius equation",
     "https://en.wikipedia.org/wiki/Arrhenius_equation","encyclopedia",
     "Higher temperature increases kinetic energy and reaction rate. Causal direction fully reversed."),

    ("Chemistry","CAUSAL_INVERSION","An increase in pressure causes gas solubility in liquids to decrease, according to Henry's law.",
     1,"medium","Wikipedia: Henry's law",
     "https://en.wikipedia.org/wiki/Henry%27s_law","encyclopedia",
     "Henry's law: higher pressure increases gas solubility. Direction reversed."),

    ("Chemistry","CAUSAL_INVERSION","Increasing the concentration of reactants causes the equilibrium constant K to increase.",
     1,"hard","Wikipedia: Chemical equilibrium",
     "https://en.wikipedia.org/wiki/Chemical_equilibrium","encyclopedia",
     "K is temperature-dependent, not concentration-dependent. Concentration changes shift equilibrium position."),

    ("Chemistry","CAUSAL_INVERSION","Exothermic dissolution causes the solution temperature to decrease because heat is released.",
     1,"medium","Wikipedia: Enthalpy of solution",
     "https://en.wikipedia.org/wiki/Enthalpy_of_solution","encyclopedia",
     "Exothermic dissolution releases heat INTO the solution, increasing temperature. Direction reversed."),

    ("Chemistry","OUTDATED_SCIENTIFIC_CLAIM","Phlogiston is the substance released during combustion.",
     1,"easy","Wikipedia: Phlogiston theory",
     "https://en.wikipedia.org/wiki/Phlogiston_theory","encyclopedia",
     "Phlogiston theory disproved by Lavoisier (1770s), replaced by oxygen theory of combustion."),

    ("Chemistry","OUTDATED_SCIENTIFIC_CLAIM","Chemical elements cannot be interconverted and all matter is made of four elements: earth, water, fire, and air.",
     1,"easy","Wikipedia: Classical element",
     "https://en.wikipedia.org/wiki/Classical_element","encyclopedia",
     "Classical four-element theory superseded by atomic theory. Elements can be interconverted via nuclear reactions."),

    ("Chemistry","OUTDATED_SCIENTIFIC_CLAIM","Benzene is composed of alternating single and double bonds that are fixed in position.",
     1,"medium","Wikipedia: Benzene",
     "https://en.wikipedia.org/wiki/Benzene","encyclopedia",
     "Kekulé's alternating bond model superseded by resonance/delocalization model (1930s–1950s)."),

    ("Chemistry","OUTDATED_SCIENTIFIC_CLAIM","Atoms are indivisible solid spheres, as proposed by Dalton's atomic model.",
     1,"easy","Wikipedia: Atomic theory",
     "https://en.wikipedia.org/wiki/Atomic_theory","encyclopedia",
     "Dalton's solid sphere model superseded; atoms have substructure (Thomson, Rutherford, Bohr, QM)."),

    ("Chemistry","OUTDATED_SCIENTIFIC_CLAIM","Valence electrons are arranged in shells at fixed distances from the nucleus, as described by the Bohr model for multi-electron atoms.",
     1,"medium","Wikipedia: Bohr model",
     "https://en.wikipedia.org/wiki/Bohr_model","encyclopedia",
     "Bohr model accurate for H only. Multi-electron atoms require quantum mechanical orbital model."),

    ("Chemistry","TRUE_CORE_FALSE_ELABORATION","The Haber process synthesises ammonia from nitrogen and hydrogen gas and requires a temperature of exactly 300°C to proceed.",
     1,"medium","Wikipedia: Haber process",
     "https://en.wikipedia.org/wiki/Haber_process","encyclopedia",
     "Core true: Haber process makes NH₃ from N₂ + H₂. Fabrication: typical operating temperature is 400–500°C, not 300°C."),

    ("Chemistry","TRUE_CORE_FALSE_ELABORATION","DNA is composed of nucleotides, each containing a phosphate group, a sugar, and one of four nitrogenous bases; the sugar in DNA is ribose.",
     1,"medium","Wikipedia: DNA",
     "https://en.wikipedia.org/wiki/DNA","encyclopedia",
     "Core true: nucleotide structure. Fabrication: DNA sugar is deoxyribose; ribose is in RNA."),

    ("Chemistry","TRUE_CORE_FALSE_ELABORATION","Electroplating uses electrolysis to deposit a metal layer; the object to be plated is connected to the positive terminal of the power supply.",
     1,"medium","Wikipedia: Electroplating",
     "https://en.wikipedia.org/wiki/Electroplating","encyclopedia",
     "Core true: electrolysis deposits metal. Fabrication: the object is the CATHODE (negative terminal), not anode."),

    ("Chemistry","TRUE_CORE_FALSE_ELABORATION","Ozone (O₃) is formed in the stratosphere when UV radiation splits N₂ molecules.",
     1,"medium","Wikipedia: Ozone layer",
     "https://en.wikipedia.org/wiki/Ozone_layer","encyclopedia",
     "Core true: UV creates ozone. Fabrication: UV splits O₂ (not N₂) molecules; atomic O then reacts with O₂ to form O₃."),

    ("Chemistry","TRUE_CORE_FALSE_ELABORATION","Aspirin is an analgesic that works by inhibiting cyclooxygenase enzymes; it was first synthesised by Louis Pasteur in 1897.",
     1,"medium","Wikipedia: Aspirin",
     "https://en.wikipedia.org/wiki/Aspirin","encyclopedia",
     "Core true: COX inhibitor. Fabrication: synthesised by Felix Hoffmann (Bayer) in 1897, not Pasteur."),

    # ══════════════════════════════════════════════════════════════════════
    # BIOLOGY
    # ══════════════════════════════════════════════════════════════════════

    ("Biology","TRUE_CONTROL","Human DNA contains approximately 3 billion base pairs.",
     0,"easy","Wikipedia: Human genome",
     "https://en.wikipedia.org/wiki/Human_genome","encyclopedia",
     "Human haploid genome: ~3.2 billion bp. Factual."),

    ("Biology","TRUE_CONTROL","The mitochondrion is the primary site of ATP synthesis via oxidative phosphorylation in eukaryotic cells.",
     0,"easy","Wikipedia: Mitochondrion",
     "https://en.wikipedia.org/wiki/Mitochondrion","encyclopedia",
     "Mitochondria produce ATP via oxidative phosphorylation (electron transport chain + ATP synthase). Factual."),

    ("Biology","TRUE_CONTROL","DNA replication is semi-conservative: each daughter strand retains one original parental strand.",
     0,"medium","Wikipedia: DNA replication",
     "https://en.wikipedia.org/wiki/DNA_replication","encyclopedia",
     "Meselson-Stahl (1958) established semi-conservative replication. Factual."),

    ("Biology","TRUE_CONTROL","The human body has 23 pairs of chromosomes, totalling 46.",
     0,"easy","Wikipedia: Human karyotype",
     "https://en.wikipedia.org/wiki/Human_karyotype","encyclopedia",
     "Diploid human cells have 46 chromosomes (23 pairs). Factual."),

    ("Biology","TRUE_CONTROL","Photosynthesis in plants occurs in the chloroplast and converts CO₂ and water into glucose using light energy.",
     0,"easy","Wikipedia: Photosynthesis",
     "https://en.wikipedia.org/wiki/Photosynthesis","encyclopedia",
     "6CO₂+6H₂O+light→C₆H₁₂O₆+6O₂ in chloroplasts. Factual."),

    ("Biology","NUMERICAL_PRECISION","Human DNA contains approximately 300 million base pairs.",
     1,"medium","Wikipedia: Human genome",
     "https://en.wikipedia.org/wiki/Human_genome","encyclopedia",
     "Correct: ~3 billion (3×10⁹) bp. Claim states 300 million (3×10⁸) — off by factor of 10."),

    ("Biology","NUMERICAL_PRECISION","The human body has 46 pairs of chromosomes, totalling 92.",
     1,"easy","Wikipedia: Human karyotype",
     "https://en.wikipedia.org/wiki/Human_karyotype","encyclopedia",
     "Humans have 23 pairs (46 total). Claim doubles both values."),

    ("Biology","NUMERICAL_PRECISION","A typical human cell contains approximately 20 to 25 total genes.",
     1,"medium","Wikipedia: Human genome",
     "https://en.wikipedia.org/wiki/Human_genome","encyclopedia",
     "Human genome encodes ~20,000–25,000 protein-coding genes. Claim omits three orders of magnitude."),

    ("Biology","NUMERICAL_PRECISION","DNA replication copies the genome at approximately 10 nucleotides per second in humans.",
     1,"hard","Wikipedia: DNA replication",
     "https://en.wikipedia.org/wiki/DNA_replication","encyclopedia",
     "Human replication forks proceed at ~50 bp/s per fork; ~1000 nt/s for E. coli. 10 nt/s is too slow."),

    ("Biology","NUMERICAL_PRECISION","Cellular respiration of one molecule of glucose yields approximately 2 ATP molecules in total.",
     1,"medium","Wikipedia: Cellular respiration",
     "https://en.wikipedia.org/wiki/Cellular_respiration","encyclopedia",
     "Complete oxidative phosphorylation yields ~30–32 ATP per glucose. Claim only counts glycolysis net yield."),

    ("Biology","UNIT_SCALE","A typical human cell nucleus has a diameter of approximately 6 mm.",
     1,"medium","Wikipedia: Cell nucleus",
     "https://en.wikipedia.org/wiki/Cell_nucleus","encyclopedia",
     "Nucleus diameter ~6 μm (micrometres), not mm. Claim is 10³ too large."),

    ("Biology","UNIT_SCALE","The average human red blood cell has a diameter of approximately 8 mm.",
     1,"medium","Wikipedia: Red blood cell",
     "https://en.wikipedia.org/wiki/Red_blood_cell","encyclopedia",
     "RBC diameter ~8 μm, not 8 mm. Claim is 10³ too large."),

    ("Biology","UNIT_SCALE","DNA nucleotides are spaced approximately 3.4 cm apart along the double helix.",
     1,"hard","Wikipedia: DNA",
     "https://en.wikipedia.org/wiki/DNA","encyclopedia",
     "Base-pair spacing is 3.4 Å (0.34 nm), not cm. Claim is ~10⁸ too large."),

    ("Biology","UNIT_SCALE","A ribosome has a diameter of approximately 25 micrometres.",
     1,"hard","Wikipedia: Ribosome",
     "https://en.wikipedia.org/wiki/Ribosome","encyclopedia",
     "Ribosome diameter is ~25–30 nm (nanometres), not micrometres. Claim is 1000× too large."),

    ("Biology","UNIT_SCALE","The human genome contains approximately 3 billion kilobases of DNA.",
     1,"hard","Wikipedia: Human genome",
     "https://en.wikipedia.org/wiki/Human_genome","encyclopedia",
     "Genome is ~3 billion BASE PAIRS, not kilobases. In kilobases: ~3 million kb. Claim inflates by 10³."),

    ("Biology","NEGATION","Mitochondria do not contain their own DNA.",
     1,"easy","Wikipedia: Mitochondrial DNA",
     "https://en.wikipedia.org/wiki/Mitochondrial_DNA","encyclopedia",
     "Mitochondria DO have their own circular DNA (mtDNA). Negation is false."),

    ("Biology","NEGATION","Enzymes are permanently consumed and destroyed during the biochemical reactions they catalyse.",
     1,"medium","Wikipedia: Enzyme",
     "https://en.wikipedia.org/wiki/Enzyme","encyclopedia",
     "Enzymes act as catalysts and emerge unchanged/regenerated; claim incorrectly asserts they are consumed."),

    ("Biology","NEGATION","CRISPR-Cas9 does not use RNA to guide DNA editing.",
     1,"medium","Wikipedia: CRISPR",
     "https://en.wikipedia.org/wiki/CRISPR","encyclopedia",
     "CRISPR-Cas9 DOES use guide RNA (sgRNA) to locate target DNA sequences. Negation is false."),

    ("Biology","NEGATION","DNA polymerase does not require a primer to initiate synthesis of a new DNA strand.",
     1,"medium","Wikipedia: DNA polymerase",
     "https://en.wikipedia.org/wiki/DNA_polymerase","encyclopedia",
     "DNA polymerase requires a primer (free 3'-OH group) to begin synthesis. Negation is false."),

    ("Biology","NEGATION","Telomeres do not shorten with each cell division in somatic cells.",
     1,"medium","Wikipedia: Telomere",
     "https://en.wikipedia.org/wiki/Telomere","encyclopedia",
     "Telomeres DO shorten with successive cell divisions (Hayflick limit). Negation is false."),

    ("Biology","CAUSAL_INVERSION","mRNA is transcribed from a protein template during gene expression.",
     1,"easy","Wikipedia: Transcription",
     "https://en.wikipedia.org/wiki/Transcription_(biology)","encyclopedia",
     "mRNA is transcribed from DNA, not protein. Causal/directional source reversed."),

    ("Biology","CAUSAL_INVERSION","Natural selection acts on genotype directly, causing changes in phenotype distribution.",
     1,"medium","Wikipedia: Natural selection",
     "https://en.wikipedia.org/wiki/Natural_selection","encyclopedia",
     "Selection acts on phenotype; differential survival changes allele/genotype frequencies. Direction reversed."),

    ("Biology","CAUSAL_INVERSION","Increased oxygen availability causes the haemoglobin saturation curve to shift left, releasing less oxygen to tissues.",
     1,"hard","Wikipedia: Oxygen–haemoglobin dissociation curve",
     "https://en.wikipedia.org/wiki/Oxygen%E2%80%93haemoglobin_dissociation_curve","encyclopedia",
     "Rightward shift (Bohr effect) releases more O₂. Increased pO₂ increases saturation, not decreases release."),

    ("Biology","CAUSAL_INVERSION","Insulin secretion causes blood glucose levels to rise in healthy individuals.",
     1,"easy","Wikipedia: Insulin",
     "https://en.wikipedia.org/wiki/Insulin","encyclopedia",
     "Insulin promotes glucose uptake, LOWERING blood glucose. Causal direction reversed."),

    ("Biology","CAUSAL_INVERSION","Apoptosis is triggered by cell survival signals and prevents programmed cell death.",
     1,"medium","Wikipedia: Apoptosis",
     "https://en.wikipedia.org/wiki/Apoptosis","encyclopedia",
     "Apoptosis IS programmed cell death, triggered by stress/damage signals, not survival signals."),

    ("Biology","OUTDATED_SCIENTIFIC_CLAIM","Acquired characteristics are inherited by offspring, as proposed by Lamarck.",
     1,"easy","Wikipedia: Lamarckism",
     "https://en.wikipedia.org/wiki/Lamarckism","encyclopedia",
     "Lamarckian inheritance disproved. Neo-Darwinian synthesis: only genetic mutations are heritable (excluding epigenetics debate)."),

    ("Biology","OUTDATED_SCIENTIFIC_CLAIM","The spontaneous generation of life from non-living matter occurs regularly in nature.",
     1,"easy","Wikipedia: Spontaneous generation",
     "https://en.wikipedia.org/wiki/Spontaneous_generation","encyclopedia",
     "Disproved by Pasteur (1859–1861). Biogenesis: life arises from pre-existing life."),

    ("Biology","OUTDATED_SCIENTIFIC_CLAIM","Genes are located on proteins rather than nucleic acids.",
     1,"medium","Wikipedia: Avery–MacLeod–McCarty experiment",
     "https://en.wikipedia.org/wiki/Avery%E2%80%93MacLeod%E2%80%93McCarty_experiment","encyclopedia",
     "Pre-1944 belief. Avery, MacLeod, McCarty (1944) and Hershey-Chase (1952) proved DNA is the genetic material."),

    ("Biology","OUTDATED_SCIENTIFIC_CLAIM","Junk DNA has no biological function and constitutes the majority of the non-coding human genome.",
     1,"hard","Wikipedia: Noncoding DNA",
     "https://en.wikipedia.org/wiki/Noncoding_DNA","encyclopedia",
     "ENCODE project (2012) found ~80% of genome has biochemical activity. 'Junk DNA' concept is outdated."),

    ("Biology","OUTDATED_SCIENTIFIC_CLAIM","Neurons in the adult brain cannot regenerate or form new connections.",
     1,"medium","Wikipedia: Neurogenesis",
     "https://en.wikipedia.org/wiki/Adult_neurogenesis","encyclopedia",
     "Adult neurogenesis occurs in hippocampus; synaptic plasticity allows new connection formation throughout life."),

    ("Biology","TRUE_CORE_FALSE_ELABORATION","The polymerase chain reaction (PCR), developed by Kary Mullis in 1983, amplifies DNA using RNA polymerase.",
     1,"medium","Wikipedia: Polymerase chain reaction",
     "https://en.wikipedia.org/wiki/Polymerase_chain_reaction","encyclopedia",
     "Core true: PCR developed by Mullis ~1983. Fabrication: PCR uses heat-stable DNA polymerase (Taq), not RNA polymerase."),

    ("Biology","TRUE_CORE_FALSE_ELABORATION","The double helix structure of DNA was proposed by Watson and Crick in 1953, based on X-ray data from their own laboratory.",
     1,"medium","Wikipedia: Nucleic acid double helix",
     "https://en.wikipedia.org/wiki/Nucleic_acid_double_helix","encyclopedia",
     "Core true: Watson-Crick 1953. Fabrication: key X-ray data (Photo 51) came from Rosalind Franklin at King's College."),

    ("Biology","TRUE_CORE_FALSE_ELABORATION","The HIV virus causes AIDS by infecting and destroying CD4+ T-cells, and it was first identified in the 1970s.",
     1,"medium","Wikipedia: HIV/AIDS",
     "https://en.wikipedia.org/wiki/HIV/AIDS","encyclopedia",
     "Core true: HIV destroys CD4+ T-cells. Fabrication: HIV was identified in 1983 by Montagnier/Barré-Sinoussi and Gallo."),

    ("Biology","TRUE_CORE_FALSE_ELABORATION","Antibiotics kill bacteria by disrupting cell wall synthesis; they are equally effective against viral infections.",
     1,"easy","Wikipedia: Antibiotic",
     "https://en.wikipedia.org/wiki/Antibiotic","encyclopedia",
     "Core true: some antibiotics disrupt cell walls. Fabrication: antibiotics have no effect on viral infections."),

    ("Biology","TRUE_CORE_FALSE_ELABORATION","Restriction enzymes, used in molecular cloning, cut DNA at random positions along the strand.",
     1,"medium","Wikipedia: Restriction enzyme",
     "https://en.wikipedia.org/wiki/Restriction_enzyme","encyclopedia",
     "Core true: restriction enzymes cut DNA. Fabrication: they cut at specific recognition sequences, not random positions."),

    # ══════════════════════════════════════════════════════════════════════
    # MEDICINE
    # ══════════════════════════════════════════════════════════════════════

    ("Medicine","TRUE_CONTROL","Penicillin, the first widely used antibiotic, was discovered by Alexander Fleming in 1928.",
     0,"easy","Wikipedia: Penicillin",
     "https://en.wikipedia.org/wiki/Penicillin","encyclopedia",
     "Fleming observed penicillin's antibacterial effects in 1928. Factual."),

    ("Medicine","TRUE_CONTROL","Type 1 diabetes is caused by autoimmune destruction of insulin-producing beta cells in the pancreatic islets of Langerhans.",
     0,"medium","Wikipedia: Diabetes mellitus type 1",
     "https://en.wikipedia.org/wiki/Diabetes_mellitus_type_1","encyclopedia",
     "T1DM: autoimmune destruction of β-cells → no insulin production. Factual."),

    ("Medicine","TRUE_CONTROL","Hypertension is defined as a sustained systolic blood pressure of ≥140 mmHg or diastolic ≥90 mmHg.",
     0,"medium","Wikipedia: Hypertension",
     "https://en.wikipedia.org/wiki/Hypertension","encyclopedia",
     "WHO/JNC-7 definition: ≥140/90 mmHg. Factual."),

    ("Medicine","TRUE_CONTROL","The ABO blood group system involves antigens on red blood cell surfaces and corresponding antibodies in plasma.",
     0,"easy","Wikipedia: ABO blood group system",
     "https://en.wikipedia.org/wiki/ABO_blood_group_system","encyclopedia",
     "ABO system: surface antigens + serum antibodies. Discovered by Landsteiner 1901. Factual."),

    ("Medicine","TRUE_CONTROL","Statins reduce cardiovascular risk primarily by inhibiting HMG-CoA reductase, the rate-limiting enzyme in cholesterol synthesis.",
     0,"medium","Wikipedia: Statin",
     "https://en.wikipedia.org/wiki/Statin","encyclopedia",
     "Statins competitively inhibit HMG-CoA reductase, reducing LDL cholesterol. Factual."),

    ("Medicine","NUMERICAL_PRECISION","Normal adult resting heart rate is approximately 100–120 beats per minute.",
     1,"easy","Wikipedia: Heart rate",
     "https://en.wikipedia.org/wiki/Heart_rate","encyclopedia",
     "Normal resting HR: 60–100 bpm. Claim describes tachycardia range."),

    ("Medicine","NUMERICAL_PRECISION","The normal fasting blood glucose level in a healthy adult is approximately 7.0–8.0 mmol/L.",
     1,"medium","Wikipedia: Blood sugar level",
     "https://en.wikipedia.org/wiki/Blood_sugar_level","encyclopedia",
     "Normal fasting glucose: 3.9–5.5 mmol/L (70–100 mg/dL). Claim describes impaired fasting glucose range."),

    ("Medicine","NUMERICAL_PRECISION","Hypertension threshold is systolic blood pressure ≥100 mmHg.",
     1,"easy","Wikipedia: Hypertension",
     "https://en.wikipedia.org/wiki/Hypertension","encyclopedia",
     "Hypertension threshold is ≥140 mmHg systolic. ≥100 is within normal range."),

    ("Medicine","NUMERICAL_PRECISION","The human body temperature considered normal is approximately 38.5°C.",
     1,"easy","Wikipedia: Human body temperature",
     "https://en.wikipedia.org/wiki/Human_body_temperature","encyclopedia",
     "Normal body temperature: ~37.0°C (98.6°F). 38.5°C is low-grade fever."),

    ("Medicine","NUMERICAL_PRECISION","Severe anaemia is defined as a haemoglobin level below 12 g/dL in adults.",
     1,"medium","Wikipedia: Anaemia",
     "https://en.wikipedia.org/wiki/Anemia","encyclopedia",
     "WHO defines severe anaemia as Hb < 8 g/dL (or <7 for some guidelines). 12 g/dL is normal for women."),

    ("Medicine","UNIT_SCALE","A standard adult dose of aspirin for analgesia is approximately 500 grams.",
     1,"easy","Wikipedia: Aspirin",
     "https://en.wikipedia.org/wiki/Aspirin","encyclopedia",
     "Standard dose is 500 mg (milligrams), not grams. Claim inflates by 1000×; 500 g would be lethal."),

    ("Medicine","UNIT_SCALE","Blood pressure is typically measured in kilopascals (kPa) in clinical practice worldwide.",
     1,"medium","Wikipedia: Blood pressure",
     "https://en.wikipedia.org/wiki/Blood_pressure","encyclopedia",
     "Clinical practice uses mmHg (millimetres of mercury) globally. kPa is technically valid but not standard in clinical use."),

    ("Medicine","UNIT_SCALE","Therapeutic serum lithium levels for bipolar disorder are approximately 0.8–1.2 micromoles per litre.",
     1,"hard","Wikipedia: Lithium pharmacology",
     "https://en.wikipedia.org/wiki/Lithium_pharmacology","encyclopedia",
     "Therapeutic range: 0.8–1.2 mmol/L (millimolar), not micromolar. Claim is 1000× too low."),

    ("Medicine","UNIT_SCALE","A standard adult oral dose of paracetamol is approximately 1 kilogram.",
     1,"easy","Wikipedia: Paracetamol",
     "https://en.wikipedia.org/wiki/Paracetamol","encyclopedia",
     "Standard adult dose is 500–1000 mg (milligrams), not kilograms. Claim is off by a factor of 1000."),

    ("Medicine","UNIT_SCALE","A normal adult produces approximately 1500 litres of urine per day.",
     1,"easy","Wikipedia: Urine",
     "https://en.wikipedia.org/wiki/Urine","encyclopedia",
     "Normal urine output: ~1–2 litres/day, not 1500 litres. Claim is 750–1500× too high."),

    ("Medicine","NEGATION","Metformin, a first-line treatment for type 2 diabetes, does not lower blood glucose levels.",
     1,"easy","Wikipedia: Metformin",
     "https://en.wikipedia.org/wiki/Metformin","encyclopedia",
     "Metformin DOES lower blood glucose (primarily by reducing hepatic glucose output). Negation is false."),

    ("Medicine","NEGATION","Beta-blockers do not reduce heart rate.",
     1,"easy","Wikipedia: Beta blocker",
     "https://en.wikipedia.org/wiki/Beta_blocker","encyclopedia",
     "Beta-blockers antagonise β-adrenergic receptors, reducing heart rate and contractility. Negation is false."),

    ("Medicine","NEGATION","Vaccines do not stimulate the adaptive immune system to produce antigen-specific memory cells.",
     1,"easy","Wikipedia: Vaccine",
     "https://en.wikipedia.org/wiki/Vaccine","encyclopedia",
     "Vaccines work precisely by triggering adaptive immunity and memory cell formation. Negation is false."),

    ("Medicine","NEGATION","Haemophilia A is not caused by a deficiency of clotting factor VIII.",
     1,"medium","Wikipedia: Haemophilia A",
     "https://en.wikipedia.org/wiki/Haemophilia_A","encyclopedia",
     "Haemophilia A IS caused by deficiency of factor VIII. Negation is false."),

    ("Medicine","NEGATION","Antigen-presenting cells do not activate T lymphocytes during the adaptive immune response.",
     1,"medium","Wikipedia: Antigen-presenting cell",
     "https://en.wikipedia.org/wiki/Antigen-presenting_cell","encyclopedia",
     "APCs present antigens to T-cells via MHC, activating adaptive immunity. Negation is false."),

    ("Medicine","CAUSAL_INVERSION","ACE inhibitors lower blood pressure by increasing angiotensin II production.",
     1,"medium","Wikipedia: ACE inhibitor",
     "https://en.wikipedia.org/wiki/ACE_inhibitor","encyclopedia",
     "ACE inhibitors BLOCK angiotensin II production (by inhibiting ACE). Causal direction reversed."),

    ("Medicine","CAUSAL_INVERSION","Corticosteroids cause inflammation by promoting pro-inflammatory cytokine release.",
     1,"medium","Wikipedia: Corticosteroid",
     "https://en.wikipedia.org/wiki/Corticosteroid","encyclopedia",
     "Corticosteroids are anti-inflammatory; they suppress cytokine production. Effect inverted."),

    ("Medicine","CAUSAL_INVERSION","Smoking causes a decrease in the risk of developing lung cancer.",
     1,"easy","Wikipedia: Lung cancer",
     "https://en.wikipedia.org/wiki/Lung_cancer","encyclopedia",
     "Smoking is the single largest risk factor for lung cancer (~85% of cases). Direction reversed."),

    ("Medicine","CAUSAL_INVERSION","Opioids reduce pain perception by blocking μ-opioid receptors in the CNS.",
     1,"medium","Wikipedia: Opioid receptor",
     "https://en.wikipedia.org/wiki/Opioid_receptor","encyclopedia",
     "Opioids ACTIVATE (agonise) μ-opioid receptors, reducing pain. Blocking would antagonise them (naloxone)."),

    ("Medicine","CAUSAL_INVERSION","Diuretics treat oedema by causing the kidneys to retain more water and sodium.",
     1,"easy","Wikipedia: Diuretic",
     "https://en.wikipedia.org/wiki/Diuretic","encyclopedia",
     "Diuretics promote excretion of water and sodium. Causal direction reversed."),

    ("Medicine","OUTDATED_SCIENTIFIC_CLAIM","Bloodletting is an effective treatment for fever and infectious diseases.",
     1,"easy","Wikipedia: Bloodletting",
     "https://en.wikipedia.org/wiki/Bloodletting","encyclopedia",
     "Bloodletting was discredited in the 19th century. Known to cause harm; not evidence-based therapy."),

    ("Medicine","OUTDATED_SCIENTIFIC_CLAIM","Stomach ulcers are caused primarily by stress and excess stomach acid, with no role for infection.",
     1,"medium","Wikipedia: Peptic ulcer disease",
     "https://en.wikipedia.org/wiki/Peptic_ulcer_disease","encyclopedia",
     "H. pylori infection is the primary cause of most peptic ulcers; established by Marshall & Warren (Nobel 2005)."),

    ("Medicine","OUTDATED_SCIENTIFIC_CLAIM","The human heart is controlled by the brain via direct neural connections only, with no intrinsic pacemaker.",
     1,"medium","Wikipedia: Cardiac pacemaker",
     "https://en.wikipedia.org/wiki/Cardiac_pacemaker","encyclopedia",
     "The SA node acts as intrinsic pacemaker; heart can beat autonomously. Outdated Galenic model."),

    ("Medicine","OUTDATED_SCIENTIFIC_CLAIM","Mental illness is caused by demonic possession or supernatural forces.",
     1,"easy","Wikipedia: History of psychiatry",
     "https://en.wikipedia.org/wiki/History_of_psychiatry","encyclopedia",
     "Pre-scientific understanding. Modern psychiatry: neurobiological, genetic, and environmental causes."),

    ("Medicine","OUTDATED_SCIENTIFIC_CLAIM","The appendix is a completely vestigial organ with no function in the human body.",
     1,"medium","Wikipedia: Appendix (anatomy)",
     "https://en.wikipedia.org/wiki/Appendix_(anatomy)","encyclopedia",
     "Appendix has immune function (lymphoid tissue, gut microbiome reservoir). 'Completely vestigial' is outdated."),

    ("Medicine","TRUE_CORE_FALSE_ELABORATION","Insulin, a hormone that regulates blood glucose, is produced by the alpha cells of the islets of Langerhans in the pancreas.",
     1,"medium","Wikipedia: Insulin",
     "https://en.wikipedia.org/wiki/Insulin","encyclopedia",
     "Core true: insulin regulates blood glucose. Fabrication: produced by BETA cells (not alpha cells; alpha cells make glucagon)."),

    ("Medicine","TRUE_CORE_FALSE_ELABORATION","Penicillin, discovered by Alexander Fleming in 1928, is an antibiotic that destroys viruses by inhibiting viral reverse transcriptase.",
     1,"medium","Wikipedia: Penicillin",
     "https://en.wikipedia.org/wiki/Penicillin","encyclopedia",
     "Core true: Fleming discovered penicillin in 1928 as an antibiotic. Fabrication: Penicillin targets bacterial peptidoglycan cell walls, not viral reverse transcriptase."),

    ("Medicine","TRUE_CORE_FALSE_ELABORATION","The MMR vaccine protects against measles, mumps, and rubella, and it was demonstrated to cause autism in a 1998 Lancet study that has since been retracted.",
     1,"medium","Wikipedia: MMR vaccine controversy",
     "https://en.wikipedia.org/wiki/MMR_vaccine_controversy","encyclopedia",
     "Core true: MMR protects against those diseases; 1998 Wakefield study was retracted. Fabrication: no causal link to autism was actually demonstrated even in the original paper."),

    ("Medicine","TRUE_CORE_FALSE_ELABORATION","mRNA COVID-19 vaccines cause the body to produce spike protein antibodies and permanently alter the recipient's DNA.",
     1,"easy","Wikipedia: COVID-19 vaccine",
     "https://en.wikipedia.org/wiki/COVID-19_vaccine","encyclopedia",
     "Core true: mRNA vaccines induce spike protein antibody response. Fabrication: mRNA does not enter the nucleus or alter DNA."),

    ("Medicine","TRUE_CORE_FALSE_ELABORATION","Statin medications lower LDL cholesterol by inhibiting HMG-CoA reductase, and they directly reverse advanced coronary calcification within 24 hours.",
     1,"medium","Wikipedia: Statin",
     "https://en.wikipedia.org/wiki/Statin","encyclopedia",
     "Core true: Statins inhibit HMG-CoA reductase to reduce LDL. Fabrication: Statins do not reverse calcified plaques within 24 hours."),

    # ══════════════════════════════════════════════════════════════════════
    # MATHEMATICS
    # ══════════════════════════════════════════════════════════════════════

    ("Mathematics","TRUE_CONTROL","The sum of the interior angles of any triangle in Euclidean geometry is exactly 180 degrees.",
     0,"easy","Wikipedia: Triangle",
     "https://en.wikipedia.org/wiki/Triangle","encyclopedia",
     "Fundamental Euclidean geometry theorem. Factual."),

    ("Mathematics","TRUE_CONTROL","The number π is irrational and cannot be expressed as a ratio of two integers.",
     0,"easy","Wikipedia: Pi",
     "https://en.wikipedia.org/wiki/Pi","encyclopedia",
     "Proved irrational by Lambert (1761). Factual."),

    ("Mathematics","TRUE_CONTROL","Euler's identity states that e^(iπ) + 1 = 0.",
     0,"easy","Wikipedia: Euler's identity",
     "https://en.wikipedia.org/wiki/Euler%27s_identity","encyclopedia",
     "Exact mathematical identity. Factual."),

    ("Mathematics","TRUE_CONTROL","There are infinitely many prime numbers, as proved by Euclid.",
     0,"easy","Wikipedia: Euclid's theorem",
     "https://en.wikipedia.org/wiki/Euclid%27s_theorem","encyclopedia",
     "Euclid's theorem (c. 300 BCE): infinitely many primes. Factual."),

    ("Mathematics","TRUE_CONTROL","The Pythagorean theorem states that in a right triangle a² + b² = c², where c is the hypotenuse.",
     0,"easy","Wikipedia: Pythagorean theorem",
     "https://en.wikipedia.org/wiki/Pythagorean_theorem","encyclopedia",
     "Fundamental theorem of Euclidean geometry. Factual."),

    ("Mathematics","NUMERICAL_PRECISION","The value of π is exactly 22/7.",
     1,"easy","Wikipedia: Pi",
     "https://en.wikipedia.org/wiki/Pi","encyclopedia",
     "22/7 ≈ 3.142857... is an approximation; π is irrational. The claim 'exactly' is false."),

    ("Mathematics","NUMERICAL_PRECISION","Euler's number e is approximately 2.178.",
     1,"easy","Wikipedia: E (mathematical constant)",
     "https://en.wikipedia.org/wiki/E_(mathematical_constant)","encyclopedia",
     "e ≈ 2.71828... Claim states 2.178 — transposed digits."),

    ("Mathematics","NUMERICAL_PRECISION","The square root of 2 is exactly 1.4.",
     1,"easy","Wikipedia: Square root of 2",
     "https://en.wikipedia.org/wiki/Square_root_of_2","encyclopedia",
     "√2 ≈ 1.41421356... The claim truncates and states 'exactly', which is false for an irrational."),

    ("Mathematics","NUMERICAL_PRECISION","The sum of interior angles of a regular hexagon is 540 degrees.",
     1,"easy","Wikipedia: Hexagon",
     "https://en.wikipedia.org/wiki/Hexagon","encyclopedia",
     "Regular hexagon: (6-2)×180 = 720 degrees. Pentagon has 540°. Claim confuses hexagon with pentagon."),

    ("Mathematics","NUMERICAL_PRECISION","There are 365 prime numbers less than 1000.",
     1,"medium","Wikipedia: Prime-counting function",
     "https://en.wikipedia.org/wiki/Prime-counting_function","encyclopedia",
     "π(1000) = 168 primes less than 1000. Claim is incorrect by ~197."),

    ("Mathematics","UNIT_SCALE","An angle of π radians is equivalent to 90 degrees.",
     1,"easy","Wikipedia: Radian",
     "https://en.wikipedia.org/wiki/Radian","encyclopedia",
     "π radians = 180 degrees. π/2 radians = 90 degrees. Claim is off by factor of 2."),

    ("Mathematics","UNIT_SCALE","The area of a circle with radius 1 metre is π square centimetres.",
     1,"medium","Wikipedia: Circle",
     "https://en.wikipedia.org/wiki/Circle","encyclopedia",
     "Area = π r² = π m² = π × 10,000 cm². Claim gives the right formula but wrong unit."),

    ("Mathematics","UNIT_SCALE","A probability of 1 in 100 is equivalent to 100%.",
     1,"easy","Wikipedia: Probability",
     "https://en.wikipedia.org/wiki/Probability","encyclopedia",
     "1/100 = 0.01 = 1%, not 100%. Claim inverts the relationship."),

    ("Mathematics","UNIT_SCALE","A megabyte contains 1000 bytes.",
     1,"easy","Wikipedia: Megabyte",
     "https://en.wikipedia.org/wiki/Megabyte","encyclopedia",
     "1 MB = 10⁶ bytes (SI) or 2²⁰ bytes (IEC). Claim states 1000 bytes (= 1 kilobyte)."),

    ("Mathematics","UNIT_SCALE","The natural logarithm of 1 is equal to 1.",
     1,"easy","Wikipedia: Natural logarithm",
     "https://en.wikipedia.org/wiki/Natural_logarithm","encyclopedia",
     "ln(1) = 0, not 1. ln(e) = 1. Claim confuses ln(1) with ln(e)."),

    ("Mathematics","NEGATION","The empty set is not a subset of every set.",
     1,"easy","Wikipedia: Empty set",
     "https://en.wikipedia.org/wiki/Empty_set","encyclopedia",
     "The empty set IS a subset of every set by vacuous truth. Negation is false."),

    ("Mathematics","NEGATION","Gödel's incompleteness theorems do not apply to Peano arithmetic.",
     1,"medium","Wikipedia: Gödel's incompleteness theorems",
     "https://en.wikipedia.org/wiki/G%C3%B6del%27s_incompleteness_theorems","encyclopedia",
     "Gödel's theorems explicitly apply to Peano arithmetic and any sufficiently expressive formal system. Negation is false."),

    ("Mathematics","NEGATION","A continuous function defined on a closed bounded interval does not necessarily achieve its maximum value.",
     1,"medium","Wikipedia: Extreme value theorem",
     "https://en.wikipedia.org/wiki/Extreme_value_theorem","encyclopedia",
     "Extreme value theorem guarantees max and min on [a,b]. Negation is false."),

    ("Mathematics","NEGATION","The derivative of a constant function is not zero.",
     1,"easy","Wikipedia: Derivative",
     "https://en.wikipedia.org/wiki/Derivative","encyclopedia",
     "Derivative of any constant is 0. Negation is false."),

    ("Mathematics","NEGATION","Not all natural numbers are either even or odd.",
     1,"easy","Wikipedia: Parity (mathematics)",
     "https://en.wikipedia.org/wiki/Parity_(mathematics)","encyclopedia",
     "Every natural number is either even or odd (by definition of integer parity). Negation is false."),

    ("Mathematics","CAUSAL_INVERSION","A larger sample size causes statistical tests to become less powerful.",
     1,"medium","Wikipedia: Statistical power",
     "https://en.wikipedia.org/wiki/Statistical_power","encyclopedia",
     "Larger sample size INCREASES statistical power (probability of detecting a true effect). Direction reversed."),

    ("Mathematics","CAUSAL_INVERSION","Increasing the significance threshold α from 0.01 to 0.05 makes a test more conservative.",
     1,"medium","Wikipedia: Statistical significance",
     "https://en.wikipedia.org/wiki/Statistical_significance","encyclopedia",
     "Raising α from 0.01 to 0.05 makes the test LESS conservative (more likely to reject null). Direction reversed."),

    ("Mathematics","CAUSAL_INVERSION","Adding more features to a linear regression model always reduces the training R² value.",
     1,"medium","Wikipedia: Coefficient of determination",
     "https://en.wikipedia.org/wiki/Coefficient_of_determination","encyclopedia",
     "Adding features never reduces training R² (it can only stay same or increase). Direction reversed."),

    ("Mathematics","CAUSAL_INVERSION","A higher variance in a dataset causes the standard deviation to decrease.",
     1,"easy","Wikipedia: Standard deviation",
     "https://en.wikipedia.org/wiki/Standard_deviation","encyclopedia",
     "SD = √variance. Higher variance → higher SD. Causal direction reversed."),

    ("Mathematics","CAUSAL_INVERSION","Multiplying two numbers each greater than 1 produces a product smaller than either factor.",
     1,"easy","Wikipedia: Multiplication",
     "https://en.wikipedia.org/wiki/Multiplication","encyclopedia",
     "For a,b > 1: a×b > a and a×b > b. Claim reverses the inequality."),

    ("Mathematics","OUTDATED_SCIENTIFIC_CLAIM","The parallel postulate is a necessary consequence of the other four Euclidean postulates.",
     1,"medium","Wikipedia: Parallel postulate",
     "https://en.wikipedia.org/wiki/Parallel_postulate","encyclopedia",
     "19th-century mathematics proved the parallel postulate is independent; non-Euclidean geometries are consistent without it."),

    ("Mathematics","OUTDATED_SCIENTIFIC_CLAIM","All continuous functions are differentiable everywhere.",
     1,"medium","Wikipedia: Weierstrass function",
     "https://en.wikipedia.org/wiki/Weierstrass_function","encyclopedia",
     "Weierstrass (1872) constructed a continuous, nowhere-differentiable function. The claim is a pre-modern assumption."),

    ("Mathematics","OUTDATED_SCIENTIFIC_CLAIM","The set of real numbers has the same cardinality as the set of natural numbers.",
     1,"medium","Wikipedia: Cantor's diagonal argument",
     "https://en.wikipedia.org/wiki/Cantor%27s_diagonal_argument","encyclopedia",
     "Cantor (1891) proved |ℝ| > |ℕ|; reals are uncountable. The claim reflects pre-Cantorian intuition."),

    ("Mathematics","OUTDATED_SCIENTIFIC_CLAIM","Fermat's Last Theorem remained unproved and may be unprovable.",
     1,"medium","Wikipedia: Fermat's Last Theorem",
     "https://en.wikipedia.org/wiki/Fermat%27s_Last_Theorem","encyclopedia",
     "Andrew Wiles proved Fermat's Last Theorem in 1995. 'May be unprovable' is outdated."),

    ("Mathematics","OUTDATED_SCIENTIFIC_CLAIM","The four-colour theorem cannot be proved by computer-assisted verification.",
     1,"hard","Wikipedia: Four color theorem",
     "https://en.wikipedia.org/wiki/Four_color_theorem","encyclopedia",
     "Four color theorem was proved in 1976 by Appel and Haken using computer-assisted case analysis."),

    ("Mathematics","TRUE_CORE_FALSE_ELABORATION","Bayes' theorem provides a way to update probabilities based on new evidence; it states that P(A|B) = P(B|A).",
     1,"medium","Wikipedia: Bayes' theorem",
     "https://en.wikipedia.org/wiki/Bayes%27_theorem","encyclopedia",
     "Core true: Bayes updates probabilities. Fabrication: P(A|B) = P(B|A)×P(A)/P(B), not just P(B|A)."),

    ("Mathematics","TRUE_CORE_FALSE_ELABORATION","The central limit theorem states that the sample mean of any distribution converges to a normal distribution as n increases, regardless of whether the population has finite variance.",
     1,"hard","Wikipedia: Central limit theorem",
     "https://en.wikipedia.org/wiki/Central_limit_theorem","encyclopedia",
     "Core true: CLT gives normal convergence. Fabrication: CLT requires finite variance; heavy-tailed distributions (Cauchy) do not satisfy CLT."),

    ("Mathematics","TRUE_CORE_FALSE_ELABORATION","Euclid's algorithm finds the greatest common divisor of two integers by repeatedly subtracting the larger from the smaller, converging in O(n) steps.",
     1,"medium","Wikipedia: Euclidean algorithm",
     "https://en.wikipedia.org/wiki/Euclidean_algorithm","encyclopedia",
     "Core true: Euclid's algorithm finds GCD. Fabrication: modern form uses modular division (not subtraction); complexity is O(log min(a,b))."),

    ("Mathematics","TRUE_CORE_FALSE_ELABORATION","P vs NP is an unsolved problem in computer science asking whether every problem whose solution can be verified in polynomial time can also be solved in polynomial time; most experts believe P = NP.",
     1,"medium","Wikipedia: P versus NP problem",
     "https://en.wikipedia.org/wiki/P_versus_NP_problem","encyclopedia",
     "Core true: correct problem description. Fabrication: consensus among experts is P ≠ NP, not P = NP."),

    ("Mathematics","TRUE_CORE_FALSE_ELABORATION","The Riemann hypothesis, one of the Millennium Prize Problems, has been proved true for the first 10 trillion zeros of the Riemann zeta function.",
     1,"hard","Wikipedia: Riemann hypothesis",
     "https://en.wikipedia.org/wiki/Riemann_hypothesis","encyclopedia",
     "Core true: it is a Millennium Prize Problem. Partial fabrication: zeros have been numerically verified (>10¹³) but this does NOT constitute a proof; the hypothesis remains unproved."),
]
# fmt: on


def build_dataset() -> list[dict]:
    records = []
    seen_ids: set[str] = set()
    idx = 0
    for row in CLAIMS:
        domain, category, claim, gt, difficulty, source, url, src_type, provenance = row
        idx += 1
        record_id = f"8A_{domain[:3].upper()}_{category[:3].upper()}_{idx:04d}"
        assert record_id not in seen_ids, f"Duplicate ID: {record_id}"
        seen_ids.add(record_id)
        records.append({
            "id": record_id,
            "domain": domain,
            "category": category,
            "claim": claim,
            "ground_truth": gt,
            "ground_truth_label": "factual" if gt == 0 else "hallucinated",
            "ground_truth_source": source,
            "source_url": url,
            "source_type": src_type,
            "difficulty": difficulty,
            "provenance": provenance,
            "annotation_method": "investigator_authored_from_cited_reference",
            "hallucisense_used_for_gt": False,
        })
    return records


def main():
    records = build_dataset()
    out_path = OUT_DIR / "dataset_8a.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()

    # Verification
    domains = {}
    cats = {}
    labels = {0: 0, 1: 0}
    for r in records:
        domains[r["domain"]] = domains.get(r["domain"], 0) + 1
        cats[r["category"]] = cats.get(r["category"], 0) + 1
        labels[r["ground_truth"]] += 1

    manifest = {
        "dataset_name": "Phase8A_Scientific_Adversarial_Benchmark",
        "version": "1.0.0",
        "creation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sha256": sha,
        "total_records": len(records),
        "domain_distribution": domains,
        "category_distribution": cats,
        "label_distribution": {"factual_0": labels[0], "hallucinated_1": labels[1]},
        "annotation_method": "investigator_authored_from_cited_reference",
        "hallucisense_used_for_gt": False,
        "p1_used_for_gt": False,
        "source_types": sorted(set(r["source_type"] for r in records)),
        "generation_script": "backend/evaluation/phase8a/build_dataset_8a.py",
        "seed": "N/A — deterministic dataset, no randomness",
    }

    (OUT_DIR / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT_DIR / "dataset_hashes.json").write_text(json.dumps({"dataset_8a.jsonl": sha}, indent=2))

    print(f"Dataset 8A: {len(records)} records written → {out_path}")
    print(f"  SHA-256: {sha}")
    print(f"  Domains: {domains}")
    print(f"  Categories: {cats}")
    print(f"  Labels: factual={labels[0]}, hallucinated={labels[1]}")


if __name__ == "__main__":
    main()

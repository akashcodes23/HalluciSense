"""Phase 52 — Controlled 50/50 Balanced Forensic Dataset Generator.

Constructs 300 perfectly balanced examples (150 Factual vs 150 Hallucinated):
- 150 Factual (y=0):
  * A_clearly_factual: 40
  * M_paraphrase: 40
  * H_numerical_correctness: 35
  * G_multi_claim_consistency: 35
- 150 Hallucinated (y=1):
  * B_clearly_false: 20
  * C_direct_contradiction: 20
  * D_unsupported_claim: 20
  * E_ambiguous_claim: 15
  * F_multi_claim_contradiction: 15
  * I_numerical_error: 20
  * J_entity_swap: 15
  * K_temporal_mutation: 15
  * L_negation: 10

Outputs:
- backend/reports/phase52/forensic_50_50_dataset.json
- backend/reports/phase52/PHASE52_DIAGNOSTIC_DATASET.md
"""

import json
from pathlib import Path

OUT_DIR = Path("backend/reports/phase52")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 150 Factual Examples (y=0)
FACTUAL_EXAMPLES = [
    # A_clearly_factual (40 items)
    {"id": "F_A01", "category": "A_clearly_factual", "text": "The capital of France is Paris.", "query": "Capital of France", "label": 0},
    {"id": "F_A02", "category": "A_clearly_factual", "text": "Water freezes at 0 degrees Celsius under standard atmospheric pressure.", "query": "Water freezing point", "label": 0},
    {"id": "F_A03", "category": "A_clearly_factual", "text": "The chemical formula for water is H2O.", "query": "Chemical formula water", "label": 0},
    {"id": "F_A04", "category": "A_clearly_factual", "text": "Jupiter is the largest planet in our solar system.", "query": "Largest planet", "label": 0},
    {"id": "F_A05", "category": "A_clearly_factual", "text": "Albert Einstein developed the theory of general relativity.", "query": "Who developed general relativity", "label": 0},
    {"id": "F_A06", "category": "A_clearly_factual", "text": "The speed of light in a vacuum is approximately 299,792,458 meters per second.", "query": "Speed of light", "label": 0},
    {"id": "F_A07", "category": "A_clearly_factual", "text": "Oxygen has atomic number 8.", "query": "Atomic number oxygen", "label": 0},
    {"id": "F_A08", "category": "A_clearly_factual", "text": "Photosynthesis converts sunlight, carbon dioxide, and water into glucose and oxygen.", "query": "Photosynthesis", "label": 0},
    {"id": "F_A09", "category": "A_clearly_factual", "text": "The Pacific Ocean is the largest ocean on Earth.", "query": "Largest ocean", "label": 0},
    {"id": "F_A10", "category": "A_clearly_factual", "text": "William Shakespeare wrote Hamlet.", "query": "Who wrote Hamlet", "label": 0},
    {"id": "F_A11", "category": "A_clearly_factual", "text": "The human body typically has 206 bones in adulthood.", "query": "Human bones count", "label": 0},
    {"id": "F_A12", "category": "A_clearly_factual", "text": "Tokyo is the capital of Japan.", "query": "Capital of Japan", "label": 0},
    {"id": "F_A13", "category": "A_clearly_factual", "text": "DNA consists of two strands forming a double helix.", "query": "DNA structure", "label": 0},
    {"id": "F_A14", "category": "A_clearly_factual", "text": "The Mona Lisa was painted by Leonardo da Vinci.", "query": "Who painted Mona Lisa", "label": 0},
    {"id": "F_A15", "category": "A_clearly_factual", "text": "Mount Everest is the highest mountain above sea level.", "query": "Highest mountain", "label": 0},
    {"id": "F_A16", "category": "A_clearly_factual", "text": "Helium is a noble gas with atomic number 2.", "query": "Helium atomic number", "label": 0},
    {"id": "F_A17", "category": "A_clearly_factual", "text": "The Amazon rainforest is predominantly located in Brazil.", "query": "Amazon rainforest location", "label": 0},
    {"id": "F_A18", "category": "A_clearly_factual", "text": "The Sahara is the largest hot desert in the world.", "query": "Largest hot desert", "label": 0},
    {"id": "F_A19", "category": "A_clearly_factual", "text": "Gold is a chemical element with symbol Au.", "query": "Gold symbol", "label": 0},
    {"id": "F_A20", "category": "A_clearly_factual", "text": "The Great Barrier Reef is located off the coast of Australia.", "query": "Great Barrier Reef", "label": 0},
    {"id": "F_A21", "category": "A_clearly_factual", "text": "The Moon orbits around Earth.", "query": "Moon orbit", "label": 0},
    {"id": "F_A22", "category": "A_clearly_factual", "text": "Mars is the fourth planet from the Sun.", "query": "Mars solar system", "label": 0},
    {"id": "F_A23", "category": "A_clearly_factual", "text": "Carbon has atomic number 6.", "query": "Carbon atomic number", "label": 0},
    {"id": "F_A24", "category": "A_clearly_factual", "text": "Rome is the capital of Italy.", "query": "Capital of Italy", "label": 0},
    {"id": "F_A25", "category": "A_clearly_factual", "text": "Madrid is the capital of Spain.", "query": "Capital of Spain", "label": 0},
    {"id": "F_A26", "category": "A_clearly_factual", "text": "The human heart pumps blood throughout the circulatory system.", "query": "Heart function", "label": 0},
    {"id": "F_A27", "category": "A_clearly_factual", "text": "Nitrogen makes up roughly 78 percent of Earth's atmosphere.", "query": "Earth atmosphere nitrogen", "label": 0},
    {"id": "F_A28", "category": "A_clearly_factual", "text": "Isaac Newton published the Philosophiæ Naturalis Principia Mathematica.", "query": "Principia Mathematica author", "label": 0},
    {"id": "F_A29", "category": "A_clearly_factual", "text": "The Atlantic Ocean is the second-largest ocean on Earth.", "query": "Second largest ocean", "label": 0},
    {"id": "F_A30", "category": "A_clearly_factual", "text": "Beethoven composed nine complete symphonies.", "query": "Beethoven symphonies", "label": 0},
    {"id": "F_A31", "category": "A_clearly_factual", "text": "The Nile is a major river in northeast Africa.", "query": "Nile River", "label": 0},
    {"id": "F_A32", "category": "A_clearly_factual", "text": "Mercury is the closest planet to the Sun.", "query": "Closest planet to Sun", "label": 0},
    {"id": "F_A33", "category": "A_clearly_factual", "text": "The Declaration of Independence was adopted in 1776.", "query": "Declaration of Independence year", "label": 0},
    {"id": "F_A34", "category": "A_clearly_factual", "text": "Neil Armstrong was the first person to walk on the Moon in 1969.", "query": "First person on moon", "label": 0},
    {"id": "F_A35", "category": "A_clearly_factual", "text": "Sound requires a material medium to propagate.", "query": "Sound propagation medium", "label": 0},
    {"id": "F_A36", "category": "A_clearly_factual", "text": "Venus has a thick atmosphere containing mostly carbon dioxide.", "query": "Venus atmosphere", "label": 0},
    {"id": "F_A37", "category": "A_clearly_factual", "text": "Alexander Fleming discovered penicillin in 1928.", "query": "Penicillin discovery", "label": 0},
    {"id": "F_A38", "category": "A_clearly_factual", "text": "Ottawa is the capital city of Canada.", "query": "Capital of Canada", "label": 0},
    {"id": "F_A39", "category": "A_clearly_factual", "text": "Canberra is the capital city of Australia.", "query": "Capital of Australia", "label": 0},
    {"id": "F_A40", "category": "A_clearly_factual", "text": "Diamond is a crystalline allotrope of carbon.", "query": "Diamond carbon", "label": 0},

    # M_paraphrase (40 items)
    {"id": "F_M01", "category": "M_paraphrase", "text": "The principal city and government seat of France is Paris.", "query": "Capital of France", "label": 0},
    {"id": "F_M02", "category": "M_paraphrase", "text": "At zero degrees on the Celsius scale, pure liquid water turns to solid ice.", "query": "Water freezing point", "label": 0},
    {"id": "F_M03", "category": "M_paraphrase", "text": "A water molecule consists of two hydrogen atoms bonded to one oxygen atom.", "query": "Water formula", "label": 0},
    {"id": "F_M04", "category": "M_paraphrase", "text": "Among all the planets orbiting our Sun, Jupiter has the greatest volume and mass.", "query": "Largest planet", "label": 0},
    {"id": "F_M05", "category": "M_paraphrase", "text": "General relativity was formulated by the theoretical physicist Albert Einstein.", "query": "General relativity", "label": 0},
    {"id": "F_M06", "category": "M_paraphrase", "text": "In a vacuum, photons propagate at a constant velocity of nearly 300,000 kilometers per second.", "query": "Speed of light", "label": 0},
    {"id": "F_M07", "category": "M_paraphrase", "text": "The atomic nucleus of an oxygen atom contains exactly eight protons.", "query": "Oxygen protons", "label": 0},
    {"id": "F_M08", "category": "M_paraphrase", "text": "Plants utilize solar energy to convert carbon dioxide and water into biochemical sugars.", "query": "Photosynthesis", "label": 0},
    {"id": "F_M09", "category": "M_paraphrase", "text": "Of all marine bodies on Earth, the Pacific encompasses the largest surface area.", "query": "Pacific Ocean", "label": 0},
    {"id": "F_M10", "category": "M_paraphrase", "text": "The tragedy of Prince Hamlet of Denmark was authored by William Shakespeare.", "query": "Hamlet author", "label": 0},
    {"id": "F_M11", "category": "M_paraphrase", "text": "The adult human skeletal framework comprises 206 individual bones.", "query": "Human skeleton bones", "label": 0},
    {"id": "F_M12", "category": "M_paraphrase", "text": "The national capital and most populous metropolis of Japan is Tokyo.", "query": "Capital of Japan", "label": 0},
    {"id": "F_M13", "category": "M_paraphrase", "text": "Deoxyribonucleic acid is structured as a double-stranded helical polymer.", "query": "DNA structure", "label": 0},
    {"id": "F_M14", "category": "M_paraphrase", "text": "Leonardo da Vinci is the Renaissance master who created the Mona Lisa portrait.", "query": "Mona Lisa creator", "label": 0},
    {"id": "F_M15", "category": "M_paraphrase", "text": "The highest terrestrial elevation above mean sea level is the summit of Mount Everest.", "query": "Mount Everest height", "label": 0},
    {"id": "F_M16", "category": "M_paraphrase", "text": "With two protons in its nucleus, helium is classified as an unreactive noble gas.", "query": "Helium noble gas", "label": 0},
    {"id": "F_M17", "category": "M_paraphrase", "text": "The vast majority of the Amazon basin lies within the borders of Brazil.", "query": "Amazon rainforest Brazil", "label": 0},
    {"id": "F_M18", "category": "M_paraphrase", "text": "The Sahara represents the most extensive non-polar arid region on our planet.", "query": "Sahara desert", "label": 0},
    {"id": "F_M19", "category": "M_paraphrase", "text": "Represented by the elemental symbol Au, gold is a dense transition metal.", "query": "Gold elemental symbol", "label": 0},
    {"id": "F_M20", "category": "M_paraphrase", "text": "Situated in the Coral Sea off Queensland, the Great Barrier Reef is the largest coral system.", "query": "Great Barrier Reef Australia", "label": 0},
    {"id": "F_M21", "category": "M_paraphrase", "text": "Earth is accompanied by a single natural planetary satellite known as the Moon.", "query": "Moon Earth", "label": 0},
    {"id": "F_M22", "category": "M_paraphrase", "text": "Orbiting as the fourth celestial body from the Sun, Mars possesses an iron-rich surface.", "query": "Mars red planet", "label": 0},
    {"id": "F_M23", "category": "M_paraphrase", "text": "Carbon atoms carry precisely six positive elementary charges in their atomic nuclei.", "query": "Carbon protons", "label": 0},
    {"id": "F_M24", "category": "M_paraphrase", "text": "The historic metropolitan capital of Italy is Rome.", "query": "Capital of Italy", "label": 0},
    {"id": "F_M25", "category": "M_paraphrase", "text": "The central administrative hub and capital of Spain is Madrid.", "query": "Capital of Spain", "label": 0},
    {"id": "F_M26", "category": "M_paraphrase", "text": "Blood circulation throughout human tissues is driven by rhythmic cardiac contractions.", "query": "Heart pumping", "label": 0},
    {"id": "F_M27", "category": "M_paraphrase", "text": "Molecular nitrogen gas constitutes roughly four-fifths of the dry air in our atmosphere.", "query": "Nitrogen atmosphere", "label": 0},
    {"id": "F_M28", "category": "M_paraphrase", "text": "Sir Isaac Newton authored the foundational mathematical treatise on classical mechanics.", "query": "Isaac Newton mechanics", "label": 0},
    {"id": "F_M29", "category": "M_paraphrase", "text": "Ranked second in global surface area, the Atlantic Ocean separates the Americas from Afro-Eurasia.", "query": "Atlantic Ocean", "label": 0},
    {"id": "F_M30", "category": "M_paraphrase", "text": "Ludwig van Beethoven finalized nine numbered symphonic masterpieces during his lifetime.", "query": "Beethoven symphonies", "label": 0},
    {"id": "F_M31", "category": "M_paraphrase", "text": "Flowing north into the Mediterranean, the Nile traverses eastern Africa.", "query": "Nile River Africa", "label": 0},
    {"id": "F_M32", "category": "M_paraphrase", "text": "Positioned nearest to the central solar star, Mercury has the smallest orbital radius.", "query": "Mercury orbit", "label": 0},
    {"id": "F_M33", "category": "M_paraphrase", "text": "In the year 1776, American colonies ratified their formal Declaration of Independence.", "query": "Declaration Independence 1776", "label": 0},
    {"id": "F_M34", "category": "M_paraphrase", "text": "Apollo 11 commander Neil Armstrong stepped onto lunar soil in July 1969.", "query": "Apollo 11 Neil Armstrong", "label": 0},
    {"id": "F_M35", "category": "M_paraphrase", "text": "Acoustic pressure waves are incapable of traversing a total vacuum.", "query": "Sound in vacuum", "label": 0},
    {"id": "F_M36", "category": "M_paraphrase", "text": "Dense carbon dioxide clouds generate extreme greenhouse temperatures on Venus.", "query": "Venus carbon dioxide", "label": 0},
    {"id": "F_M37", "category": "M_paraphrase", "text": "The antimicrobial substance penicillin was discovered by Alexander Fleming in 1928.", "query": "Penicillin discovery Fleming", "label": 0},
    {"id": "F_M38", "category": "M_paraphrase", "text": "Canada is federally governed from its national capital in Ottawa.", "query": "Ottawa Canada", "label": 0},
    {"id": "F_M39", "category": "M_paraphrase", "text": "Australia maintains its federal parliament in the capital city of Canberra.", "query": "Canberra Australia", "label": 0},
    {"id": "F_M40", "category": "M_paraphrase", "text": "Under high pressure, pure carbon forms extremely rigid tetrahedral diamond lattices.", "query": "Diamond carbon lattice", "label": 0},

    # H_numerical_correctness (35 items)
    {"id": "F_H01", "category": "H_numerical_correctness", "text": "12 multiplied by 8 equals 96.", "query": "12*8", "label": 0},
    {"id": "F_H02", "category": "H_numerical_correctness", "text": "15 plus 27 equals 42.", "query": "15+27", "label": 0},
    {"id": "F_H03", "category": "H_numerical_correctness", "text": "100 divided by 4 equals 25.", "query": "100/4", "label": 0},
    {"id": "F_H04", "category": "H_numerical_correctness", "text": "50 minus 18 equals 32.", "query": "50-18", "label": 0},
    {"id": "F_H05", "category": "H_numerical_correctness", "text": "7 multiplied by 9 equals 63.", "query": "7*9", "label": 0},
    {"id": "F_H06", "category": "H_numerical_correctness", "text": "144 divided by 12 equals 12.", "query": "144/12", "label": 0},
    {"id": "F_H07", "category": "H_numerical_correctness", "text": "25 plus 75 equals 100.", "query": "25+75", "label": 0},
    {"id": "F_H08", "category": "H_numerical_correctness", "text": "9 multiplied by 9 equals 81.", "query": "9*9", "label": 0},
    {"id": "F_H09", "category": "H_numerical_correctness", "text": "80 divided by 8 equals 10.", "query": "80/8", "label": 0},
    {"id": "F_H10", "category": "H_numerical_correctness", "text": "64 minus 28 equals 36.", "query": "64-28", "label": 0},
    {"id": "F_H11", "category": "H_numerical_correctness", "text": "11 multiplied by 11 equals 121.", "query": "11*11", "label": 0},
    {"id": "F_H12", "category": "H_numerical_correctness", "text": "200 minus 45 equals 155.", "query": "200-45", "label": 0},
    {"id": "F_H13", "category": "H_numerical_correctness", "text": "6 multiplied by 7 equals 42.", "query": "6*7", "label": 0},
    {"id": "F_H14", "category": "H_numerical_correctness", "text": "81 divided by 9 equals 9.", "query": "81/9", "label": 0},
    {"id": "F_H15", "category": "H_numerical_correctness", "text": "30 plus 70 equals 100.", "query": "30+70", "label": 0},
    {"id": "F_H16", "category": "H_numerical_correctness", "text": "16 multiplied by 4 equals 64.", "query": "16*4", "label": 0},
    {"id": "F_H17", "category": "H_numerical_correctness", "text": "500 divided by 5 equals 100.", "query": "500/5", "label": 0},
    {"id": "F_H18", "category": "H_numerical_correctness", "text": "13 plus 19 equals 32.", "query": "13+19", "label": 0},
    {"id": "F_H19", "category": "H_numerical_correctness", "text": "48 divided by 6 equals 8.", "query": "48/6", "label": 0},
    {"id": "F_H20", "category": "H_numerical_correctness", "text": "8 multiplied by 5 equals 40.", "query": "8*5", "label": 0},
    {"id": "F_H21", "category": "H_numerical_correctness", "text": "90 minus 35 equals 55.", "query": "90-35", "label": 0},
    {"id": "F_H22", "category": "H_numerical_correctness", "text": "14 multiplied by 3 equals 42.", "query": "14*3", "label": 0},
    {"id": "F_H23", "category": "H_numerical_correctness", "text": "72 divided by 8 equals 9.", "query": "72/8", "label": 0},
    {"id": "F_H24", "category": "H_numerical_correctness", "text": "60 plus 40 equals 100.", "query": "60+40", "label": 0},
    {"id": "F_H25", "category": "H_numerical_correctness", "text": "15 multiplied by 5 equals 75.", "query": "15*5", "label": 0},
    {"id": "F_H26", "category": "H_numerical_correctness", "text": "120 divided by 10 equals 12.", "query": "120/10", "label": 0},
    {"id": "F_H27", "category": "H_numerical_correctness", "text": "85 minus 25 equals 60.", "query": "85-25", "label": 0},
    {"id": "F_H28", "category": "H_numerical_correctness", "text": "7 multiplied by 7 equals 49.", "query": "7*7", "label": 0},
    {"id": "F_H29", "category": "H_numerical_correctness", "text": "36 divided by 4 equals 9.", "query": "36/4", "label": 0},
    {"id": "F_H30", "category": "H_numerical_correctness", "text": "18 plus 22 equals 40.", "query": "18+22", "label": 0},
    {"id": "F_H31", "category": "H_numerical_correctness", "text": "21 multiplied by 3 equals 63.", "query": "21*3", "label": 0},
    {"id": "F_H32", "category": "H_numerical_correctness", "text": "64 divided by 8 equals 8.", "query": "64/8", "label": 0},
    {"id": "F_H33", "category": "H_numerical_correctness", "text": "45 minus 15 equals 30.", "query": "45-15", "label": 0},
    {"id": "F_H34", "category": "H_numerical_correctness", "text": "8 multiplied by 9 equals 72.", "query": "8*9", "label": 0},
    {"id": "F_H35", "category": "H_numerical_correctness", "text": "150 divided by 3 equals 50.", "query": "150/3", "label": 0},

    # G_multi_claim_consistency (35 items)
    {"id": "F_G01", "category": "G_multi_claim_consistency", "text": "Paris is the capital of France. Berlin is the capital of Germany.", "query": "Capitals France Germany", "label": 0},
    {"id": "F_G02", "category": "G_multi_claim_consistency", "text": "Water freezes at 0 degrees Celsius. Water boils at 100 degrees Celsius under standard pressure.", "query": "Water phase transitions", "label": 0},
    {"id": "F_G03", "category": "G_multi_claim_consistency", "text": "Jupiter is the largest planet in our solar system. Saturn is famous for its prominent ring system.", "query": "Gas giants", "label": 0},
    {"id": "F_G04", "category": "G_multi_claim_consistency", "text": "Mount Everest is Earth's highest mountain above sea level. K2 is the second-highest mountain.", "query": "Highest mountains", "label": 0},
    {"id": "F_G05", "category": "G_multi_claim_consistency", "text": "The Nile is a major north-flowing river in Africa. The Amazon is the largest river by discharge.", "query": "World rivers", "label": 0},
    {"id": "F_G06", "category": "G_multi_claim_consistency", "text": "Oxygen has atomic number 8. Carbon has atomic number 6.", "query": "Atomic numbers", "label": 0},
    {"id": "F_G07", "category": "G_multi_claim_consistency", "text": "William Shakespeare wrote Hamlet. He also wrote Macbeth.", "query": "Shakespeare tragedies", "label": 0},
    {"id": "F_G08", "category": "G_multi_claim_consistency", "text": "Tokyo is the capital of Japan. Rome is the capital of Italy.", "query": "Capitals", "label": 0},
    {"id": "F_G09", "category": "G_multi_claim_consistency", "text": "The Pacific is the largest ocean. The Atlantic is the second-largest ocean.", "query": "Oceans", "label": 0},
    {"id": "F_G10", "category": "G_multi_claim_consistency", "text": "Albert Einstein developed general relativity. Isaac Newton formulated the laws of universal gravitation.", "query": "Physicists", "label": 0},
    {"id": "F_G11", "category": "G_multi_claim_consistency", "text": "DNA stores genetic information. RNA plays a key role in protein synthesis.", "query": "Nucleic acids", "label": 0},
    {"id": "F_G12", "category": "G_multi_claim_consistency", "text": "Helium is a noble gas. Neon is also a noble gas.", "query": "Noble gases", "label": 0},
    {"id": "F_G13", "category": "G_multi_claim_consistency", "text": "The Sahara is the largest hot desert. Antarctica is classified as a polar desert.", "query": "Deserts", "label": 0},
    {"id": "F_G14", "category": "G_multi_claim_consistency", "text": "Leonardo da Vinci painted the Mona Lisa. Michelangelo sculpted David.", "query": "Renaissance art", "label": 0},
    {"id": "F_G15", "category": "G_multi_claim_consistency", "text": "Gold has symbol Au. Silver has symbol Ag.", "query": "Chemical symbols", "label": 0},
    {"id": "F_G16", "category": "G_multi_claim_consistency", "text": "The human heart pumps blood. The lungs facilitate gas exchange.", "query": "Human organs", "label": 0},
    {"id": "F_G17", "category": "G_multi_claim_consistency", "text": "Mercury is closest to the Sun. Neptune is the eighth planet from the Sun.", "query": "Planets Sun distance", "label": 0},
    {"id": "F_G18", "category": "G_multi_claim_consistency", "text": "Photosynthesis produces glucose. Cellular respiration breaks down glucose to release energy.", "query": "Bioenergetics", "label": 0},
    {"id": "F_G19", "category": "G_multi_claim_consistency", "text": "Madrid is the capital of Spain. Lisbon is the capital of Portugal.", "query": "Iberian capitals", "label": 0},
    {"id": "F_G20", "category": "G_multi_claim_consistency", "text": "Iron is a magnetic transition metal. Copper is an excellent conductor of electricity.", "query": "Metals properties", "label": 0},
    {"id": "F_G21", "category": "G_multi_claim_consistency", "text": "Venus is the second planet from the Sun. Earth is the third planet from the Sun.", "query": "Inner planets", "label": 0},
    {"id": "F_G22", "category": "G_multi_claim_consistency", "text": "Hydrogen is the lightest element. Helium is the second-lightest element.", "query": "Lightest elements", "label": 0},
    {"id": "F_G23", "category": "G_multi_claim_consistency", "text": "The Arctic Ocean is located around the North Pole. Antarctica is a continent at the South Pole.", "query": "Poles Earth", "label": 0},
    {"id": "F_G24", "category": "G_multi_claim_consistency", "text": "Canada is north of the United States. Mexico is south of the United States.", "query": "North America geography", "label": 0},
    {"id": "F_G25", "category": "G_multi_claim_consistency", "text": "Red blood cells transport oxygen. White blood cells defend against infections.", "query": "Blood cells", "label": 0},
    {"id": "F_G26", "category": "G_multi_claim_consistency", "text": "The Eiffel Tower is in Paris. The Colosseum is in Rome.", "query": "Monuments Europe", "label": 0},
    {"id": "F_G27", "category": "G_multi_claim_consistency", "text": "Diamond is extremely hard. Graphite is soft and slippery.", "query": "Carbon allotropes", "label": 0},
    {"id": "F_G28", "category": "G_multi_claim_consistency", "text": "The Declaration of Independence was signed in 1776. The US Constitution was written in 1787.", "query": "US history founding", "label": 0},
    {"id": "F_G29", "category": "G_multi_claim_consistency", "text": "Cows are herbivores. Lions are carnivores.", "query": "Animal diets", "label": 0},
    {"id": "F_G30", "category": "G_multi_claim_consistency", "text": "Electrons have negative charge. Protons have positive charge.", "query": "Subatomic particles charge", "label": 0},
    {"id": "F_G31", "category": "G_multi_claim_consistency", "text": "The Mediterranean Sea connects to the Atlantic Ocean. The Red Sea connects to the Indian Ocean.", "query": "Seas connections", "label": 0},
    {"id": "F_G32", "category": "G_multi_claim_consistency", "text": "Sound travels through air. Light travels through both air and vacuum.", "query": "Waves propagation", "label": 0},
    {"id": "F_G33", "category": "G_multi_claim_consistency", "text": "Athens is the capital of Greece. Cairo is the capital of Egypt.", "query": "Mediterranean capitals", "label": 0},
    {"id": "F_G34", "category": "G_multi_claim_consistency", "text": "Sodium reacts vigorously with water. Chlorine is a halogen gas.", "query": "Sodium chlorine", "label": 0},
    {"id": "F_G35", "category": "G_multi_claim_consistency", "text": "The Sun generates energy via nuclear fusion. Nuclear power plants utilize nuclear fission.", "query": "Nuclear processes", "label": 0},
]

# 150 Hallucinated Examples (y=1)
HALLUCINATED_EXAMPLES = [
    # B_clearly_false (20 items)
    {"id": "H_B01", "category": "B_clearly_false", "text": "The capital of France is Berlin.", "query": "Capital of France", "label": 1},
    {"id": "H_B02", "category": "B_clearly_false", "text": "Water freezes at 100 degrees Celsius under standard atmospheric pressure.", "query": "Water freezing point", "label": 1},
    {"id": "H_B03", "category": "B_clearly_false", "text": "The chemical formula for water is CO2.", "query": "Formula for water", "label": 1},
    {"id": "H_B04", "category": "B_clearly_false", "text": "Mars is the largest planet in our solar system.", "query": "Largest planet", "label": 1},
    {"id": "H_B05", "category": "B_clearly_false", "text": "Isaac Newton developed the theory of general relativity.", "query": "General relativity", "label": 1},
    {"id": "H_B06", "category": "B_clearly_false", "text": "The speed of light in a vacuum is 500 meters per second.", "query": "Speed of light", "label": 1},
    {"id": "H_B07", "category": "B_clearly_false", "text": "Oxygen has atomic number 79.", "query": "Oxygen atomic number", "label": 1},
    {"id": "H_B08", "category": "B_clearly_false", "text": "Photosynthesis occurs exclusively in animal cells.", "query": "Photosynthesis animal", "label": 1},
    {"id": "H_B09", "category": "B_clearly_false", "text": "The Atlantic Ocean is the smallest ocean on Earth.", "query": "Smallest ocean", "label": 1},
    {"id": "H_B10", "category": "B_clearly_false", "text": "Charles Dickens wrote Hamlet.", "query": "Who wrote Hamlet", "label": 1},
    {"id": "H_B11", "category": "B_clearly_false", "text": "The human body contains over 5,000 bones in adulthood.", "query": "Human bones", "label": 1},
    {"id": "H_B12", "category": "B_clearly_false", "text": "Beijing is the capital of Japan.", "query": "Capital of Japan", "label": 1},
    {"id": "H_B13", "category": "B_clearly_false", "text": "DNA is composed of a single straight protein strand with no nucleotides.", "query": "DNA composition", "label": 1},
    {"id": "H_B14", "category": "B_clearly_false", "text": "Pablo Picasso painted the Mona Lisa in 1985.", "query": "Mona Lisa Picasso", "label": 1},
    {"id": "H_B15", "category": "B_clearly_false", "text": "Mount Fuji is the highest mountain on Earth.", "query": "Highest mountain Fuji", "label": 1},
    {"id": "H_B16", "category": "B_clearly_false", "text": "Helium is an alkali metal that burns vigorously in air.", "query": "Helium alkali metal", "label": 1},
    {"id": "H_B17", "category": "B_clearly_false", "text": "The Amazon rainforest is located in central Germany.", "query": "Amazon Germany", "label": 1},
    {"id": "H_B18", "category": "B_clearly_false", "text": "The Sahara is an Arctic tundra covered in perpetual glacier ice.", "query": "Sahara ice tundra", "label": 1},
    {"id": "H_B19", "category": "B_clearly_false", "text": "Gold has the chemical symbol Fe and rusts easily.", "query": "Gold symbol Fe", "label": 1},
    {"id": "H_B20", "category": "B_clearly_false", "text": "The Great Barrier Reef is a volcanic mountain range in Switzerland.", "query": "Great Barrier Reef Switzerland", "label": 1},

    # C_direct_contradiction (20 items)
    {"id": "H_C01", "category": "C_direct_contradiction", "text": "The earth is entirely flat and has no curvature.", "query": "Earth shape", "label": 1},
    {"id": "H_C02", "category": "C_direct_contradiction", "text": "Humans do not require oxygen to survive.", "query": "Humans oxygen need", "label": 1},
    {"id": "H_C03", "category": "C_direct_contradiction", "text": "The sun orbits around the Earth once every 24 hours.", "query": "Sun orbit Earth", "label": 1},
    {"id": "H_C04", "category": "C_direct_contradiction", "text": "Sound travels faster in a complete vacuum than through solid steel.", "query": "Sound in vacuum", "label": 1},
    {"id": "H_C05", "category": "C_direct_contradiction", "text": "Absolute zero temperature is hotter than the core of the Sun.", "query": "Absolute zero temperature", "label": 1},
    {"id": "H_C06", "category": "C_direct_contradiction", "text": "Mammals are cold-blooded creatures that lay shelled eggs exclusively.", "query": "Mammals warm blooded", "label": 1},
    {"id": "H_C07", "category": "C_direct_contradiction", "text": "Light cannot travel through empty space.", "query": "Light in vacuum", "label": 1},
    {"id": "H_C08", "category": "C_direct_contradiction", "text": "Pure water is a strong acidic substance with pH 1.0.", "query": "Water pH", "label": 1},
    {"id": "H_C09", "category": "C_direct_contradiction", "text": "Gravity repels masses away from each other proportionally to distance.", "query": "Gravity repulsion", "label": 1},
    {"id": "H_C10", "category": "C_direct_contradiction", "text": "Diamonds are made entirely of pure liquid nitrogen.", "query": "Diamond composition", "label": 1},
    {"id": "H_C11", "category": "C_direct_contradiction", "text": "The Pacific ocean contains no liquid water.", "query": "Pacific Ocean water", "label": 1},
    {"id": "H_C12", "category": "C_direct_contradiction", "text": "Electrons have a strong positive electrical charge.", "query": "Electron charge", "label": 1},
    {"id": "H_C13", "category": "C_direct_contradiction", "text": "Plants produce carbon dioxide and consume pure methane during photosynthesis.", "query": "Photosynthesis methane", "label": 1},
    {"id": "H_C14", "category": "C_direct_contradiction", "text": "The Moon has a larger mass and volume than the Sun.", "query": "Moon Sun mass", "label": 1},
    {"id": "H_C15", "category": "C_direct_contradiction", "text": "The human brain contains zero neurons.", "query": "Brain neurons", "label": 1},
    {"id": "H_C16", "category": "C_direct_contradiction", "text": "Antarctica is the hottest tropical rainforest on Earth.", "query": "Antarctica climate", "label": 1},
    {"id": "H_C17", "category": "C_direct_contradiction", "text": "Iron is a gas at room temperature.", "query": "Iron state", "label": 1},
    {"id": "H_C18", "category": "C_direct_contradiction", "text": "All living organisms are completely devoid of cells.", "query": "Cells living organisms", "label": 1},
    {"id": "H_C19", "category": "C_direct_contradiction", "text": "The speed of sound exceeds the speed of light in a vacuum.", "query": "Speed sound vs light", "label": 1},
    {"id": "H_C20", "category": "C_direct_contradiction", "text": "The Eiffel Tower was carved out of solid marble in Ancient Egypt.", "query": "Eiffel Tower Egypt", "label": 1},

    # D_unsupported_claim (20 items)
    {"id": "H_D01", "category": "D_unsupported_claim", "text": "Napoleon Bonaparte secretly traveled to Australia in 1812 to establish an underground palace.", "query": "Napoleon Australia", "label": 1},
    {"id": "H_D02", "category": "D_unsupported_claim", "text": "Ancient Romans invented quantum computers powered by steam in 50 BC.", "query": "Romans quantum computers", "label": 1},
    {"id": "H_D03", "category": "D_unsupported_claim", "text": "Eating purple cabbage allows humans to communicate telepathically across galaxies.", "query": "Purple cabbage telepathy", "label": 1},
    {"id": "H_D04", "category": "D_unsupported_claim", "text": "Shakespeare owned a domesticated penguin named Bartholomew in Stratford-upon-Avon.", "query": "Shakespeare pet penguin", "label": 1},
    {"id": "H_D05", "category": "D_unsupported_claim", "text": "The core of the planet Saturn is composed of solid milk chocolate.", "query": "Saturn chocolate core", "label": 1},
    {"id": "H_D06", "category": "D_unsupported_claim", "text": "Cleopatra invented the electric guitar during her reign in Alexandria.", "query": "Cleopatra electric guitar", "label": 1},
    {"id": "H_D07", "category": "D_unsupported_claim", "text": "A secret civilization of giant badger monks operates beneath the Antarctic ice sheet.", "query": "Antarctic badger civilization", "label": 1},
    {"id": "H_D08", "category": "D_unsupported_claim", "text": "George Washington invented the internet in 1789 to send encrypted messages to Thomas Jefferson.", "query": "George Washington internet", "label": 1},
    {"id": "H_D09", "category": "D_unsupported_claim", "text": "Pluto is made entirely of compressed titanium and emits classical violin music.", "query": "Pluto titanium violin", "label": 1},
    {"id": "H_D10", "category": "D_unsupported_claim", "text": "The Statue of Liberty was originally built as a wireless charging tower for steam locomotives.", "query": "Statue of Liberty steam", "label": 1},
    {"id": "H_D11", "category": "D_unsupported_claim", "text": "Drinking seawater gives humans the biological ability to breathe underwater indefinitely.", "query": "Drinking seawater breathe", "label": 1},
    {"id": "H_D12", "category": "D_unsupported_claim", "text": "Beethoven composed his 9th symphony while scuba diving in the Pacific Ocean.", "query": "Beethoven scuba diving", "label": 1},
    {"id": "H_D13", "category": "D_unsupported_claim", "text": "Sunlight contains micro-crystals of ruby that give birds the power of flight.", "query": "Ruby crystals birds flight", "label": 1},
    {"id": "H_D14", "category": "D_unsupported_claim", "text": "The Great Pyramids were originally painted neon pink with fluorescent dyes.", "query": "Great Pyramids neon pink", "label": 1},
    {"id": "H_D15", "category": "D_unsupported_claim", "text": "Galileo used an optical laser pointer to communicate with Martian settlers.", "query": "Galileo laser pointer", "label": 1},
    {"id": "H_D16", "category": "D_unsupported_claim", "text": "The Amazon river flows backwards every leap year due to lunar gravitational anomalies.", "query": "Amazon river backwards leap year", "label": 1},
    {"id": "H_D17", "category": "D_unsupported_claim", "text": "Julius Caesar was an accomplished jazz saxophonist in 44 BC.", "query": "Julius Caesar jazz saxophone", "label": 1},
    {"id": "H_D18", "category": "D_unsupported_claim", "text": "Clouds are solid fiberglass formations suspended by magnetic levitation.", "query": "Clouds fiberglass", "label": 1},
    {"id": "H_D19", "category": "D_unsupported_claim", "text": "Alexander the Great discovered nuclear fission using bronze pottery.", "query": "Alexander the Great nuclear", "label": 1},
    {"id": "H_D20", "category": "D_unsupported_claim", "text": "Koalas can run at speeds exceeding 120 miles per hour when hunting gazelles.", "query": "Koala running speed 120", "label": 1},

    # E_ambiguous_claim (15 items)
    {"id": "H_E01", "category": "E_ambiguous_claim", "text": "It has been said that something happened somewhere in Europe many centuries ago that changed everything.", "query": "European history", "label": 1},
    {"id": "H_E02", "category": "E_ambiguous_claim", "text": "Certain mysterious cosmic energy pulses might be responsible for all unknown phenomena on Earth.", "query": "Cosmic energy", "label": 1},
    {"id": "H_E03", "category": "E_ambiguous_claim", "text": "Some ancient philosopher proved that reality is just an illusion created by vibrational frequencies.", "query": "Philosophy vibration reality", "label": 1},
    {"id": "H_E04", "category": "E_ambiguous_claim", "text": "Scientists may have discovered a secret element that defies all known laws of physics.", "query": "Secret element physics", "label": 1},
    {"id": "H_E05", "category": "E_ambiguous_claim", "text": "An obscure historical figure allegedly achieved immortality through natural alchemy.", "query": "Alchemy immortality", "label": 1},
    {"id": "H_E06", "category": "E_ambiguous_claim", "text": "There are rumors that the weather in certain remote islands is controlled by ancient mechanisms.", "query": "Ancient weather control", "label": 1},
    {"id": "H_E07", "category": "E_ambiguous_claim", "text": "Certain frequencies of sound are capable of manipulating human thoughts instantaneously.", "query": "Sound thoughts manipulation", "label": 1},
    {"id": "H_E08", "category": "E_ambiguous_claim", "text": "A famous king in the Middle Ages was actually two different people acting as one person.", "query": "Medieval kings double", "label": 1},
    {"id": "H_E09", "category": "E_ambiguous_claim", "text": "Some studies suggest that trees have a hidden collective consciousness with secret plans.", "query": "Tree consciousness plans", "label": 1},
    {"id": "H_E10", "category": "E_ambiguous_claim", "text": "Mysterious forces at the center of the Earth regulate the rotation of the Milky Way galaxy.", "query": "Earth center Milky Way", "label": 1},
    {"id": "H_E11", "category": "E_ambiguous_claim", "text": "Ancient texts suggest that humans once flew using specialized vocal harmonies.", "query": "Vocal harmonies flight", "label": 1},
    {"id": "H_E12", "category": "E_ambiguous_claim", "text": "Certain minerals can produce unlimited electrical power if exposed to starlight.", "query": "Starlight mineral power", "label": 1},
    {"id": "H_E13", "category": "E_ambiguous_claim", "text": "An unidentified historical empire conquered all continents before disappearing without a trace.", "query": "Lost empire continents", "label": 1},
    {"id": "H_E14", "category": "E_ambiguous_claim", "text": "The moon might be hollow and occupied by forgotten historical expeditions.", "query": "Hollow moon expeditions", "label": 1},
    {"id": "H_E15", "category": "E_ambiguous_claim", "text": "Some obscure mathematical equations have the physical power to alter gravitational fields.", "query": "Math equations gravity", "label": 1},

    # F_multi_claim_contradiction (15 items)
    {"id": "H_F01", "category": "F_multi_claim_contradiction", "text": "Paris is the capital of France. Berlin is the capital of France.", "query": "Capitals France", "label": 1},
    {"id": "H_F02", "category": "F_multi_claim_contradiction", "text": "Water freezes at 0 degrees Celsius. Water only freezes at 100 degrees Celsius.", "query": "Water freezing point", "label": 1},
    {"id": "H_F03", "category": "F_multi_claim_contradiction", "text": "The sun rises in the east. The sun never rises in the east and only rises in the west.", "query": "Sun rising", "label": 1},
    {"id": "H_F04", "category": "F_multi_claim_contradiction", "text": "Jupiter is a gas giant. Jupiter has a solid iron surface with no atmosphere.", "query": "Jupiter structure", "label": 1},
    {"id": "H_F05", "category": "F_multi_claim_contradiction", "text": "Albert Einstein was born in Germany. Albert Einstein was born on the moon in 2050.", "query": "Einstein birth", "label": 1},
    {"id": "H_F06", "category": "F_multi_claim_contradiction", "text": "Gold is a metal. Gold is a non-metallic organic liquid.", "query": "Gold metal liquid", "label": 1},
    {"id": "H_F07", "category": "F_multi_claim_contradiction", "text": "Mount Everest is in the Himalayas. Mount Everest is located entirely in Brazil.", "query": "Everest location", "label": 1},
    {"id": "H_F08", "category": "F_multi_claim_contradiction", "text": "Humans are mammals. Humans belong to the reptile class.", "query": "Humans mammals reptiles", "label": 1},
    {"id": "H_F09", "category": "F_multi_claim_contradiction", "text": "Tokyo is in Japan. Tokyo is located in central Canada.", "query": "Tokyo Japan Canada", "label": 1},
    {"id": "H_F10", "category": "F_multi_claim_contradiction", "text": "The Pacific Ocean is filled with saltwater. The Pacific Ocean contains purely freshwater.", "query": "Pacific saltwater freshwater", "label": 1},
    {"id": "H_F11", "category": "F_multi_claim_contradiction", "text": "Oxygen is necessary for aerobic respiration. Oxygen is completely toxic and never used in respiration.", "query": "Oxygen respiration", "label": 1},
    {"id": "H_F12", "category": "F_multi_claim_contradiction", "text": "Shakespeare was an English playwright. Shakespeare never wrote in English and lived in Tokyo.", "query": "Shakespeare language", "label": 1},
    {"id": "H_F13", "category": "F_multi_claim_contradiction", "text": "The speed of light is finite. The speed of light is infinite and instantaneous.", "query": "Speed of light finite infinite", "label": 1},
    {"id": "H_F14", "category": "F_multi_claim_contradiction", "text": "DNA has a double helix structure. DNA is completely unstructured and contains no strands.", "query": "DNA structure unstructured", "label": 1},
    {"id": "H_F15", "category": "F_multi_claim_contradiction", "text": "The Nile River is in Africa. The Nile River flows exclusively through Antarctica.", "query": "Nile River Antarctica", "label": 1},

    # I_numerical_error (20 items)
    {"id": "H_I01", "category": "I_numerical_error", "text": "12 multiplied by 8 equals 95.", "query": "12*8", "label": 1},
    {"id": "H_I02", "category": "I_numerical_error", "text": "15 plus 27 equals 49.", "query": "15+27", "label": 1},
    {"id": "H_I03", "category": "I_numerical_error", "text": "100 divided by 4 equals 26.", "query": "100/4", "label": 1},
    {"id": "H_I04", "category": "I_numerical_error", "text": "50 minus 18 equals 35.", "query": "50-18", "label": 1},
    {"id": "H_I05", "category": "I_numerical_error", "text": "7 multiplied by 9 equals 65.", "query": "7*9", "label": 1},
    {"id": "H_I06", "category": "I_numerical_error", "text": "144 divided by 12 equals 14.", "query": "144/12", "label": 1},
    {"id": "H_I07", "category": "I_numerical_error", "text": "25 plus 75 equals 110.", "query": "25+75", "label": 1},
    {"id": "H_I08", "category": "I_numerical_error", "text": "9 multiplied by 9 equals 83.", "query": "9*9", "label": 1},
    {"id": "H_I09", "category": "I_numerical_error", "text": "80 divided by 8 equals 12.", "query": "80/8", "label": 1},
    {"id": "H_I10", "category": "I_numerical_error", "text": "64 minus 28 equals 40.", "query": "64-28", "label": 1},
    {"id": "H_I11", "category": "I_numerical_error", "text": "11 multiplied by 11 equals 125.", "query": "11*11", "label": 1},
    {"id": "H_I12", "category": "I_numerical_error", "text": "200 minus 45 equals 160.", "query": "200-45", "label": 1},
    {"id": "H_I13", "category": "I_numerical_error", "text": "6 multiplied by 7 equals 45.", "query": "6*7", "label": 1},
    {"id": "H_I14", "category": "I_numerical_error", "text": "81 divided by 9 equals 8.", "query": "81/9", "label": 1},
    {"id": "H_I15", "category": "I_numerical_error", "text": "30 plus 70 equals 90.", "query": "30+70", "label": 1},
    {"id": "H_I16", "category": "I_numerical_error", "text": "16 multiplied by 4 equals 60.", "query": "16*4", "label": 1},
    {"id": "H_I17", "category": "I_numerical_error", "text": "500 divided by 5 equals 105.", "query": "500/5", "label": 1},
    {"id": "H_I18", "category": "I_numerical_error", "text": "13 plus 19 equals 35.", "query": "13+19", "label": 1},
    {"id": "H_I19", "category": "I_numerical_error", "text": "48 divided by 6 equals 9.", "query": "48/6", "label": 1},
    {"id": "H_I20", "category": "I_numerical_error", "text": "8 multiplied by 5 equals 45.", "query": "8*5", "label": 1},

    # J_entity_swap (15 items)
    {"id": "H_J01", "category": "J_entity_swap", "text": "Leonardo da Vinci wrote the tragedy Hamlet.", "query": "Who wrote Hamlet", "label": 1},
    {"id": "H_J02", "category": "J_entity_swap", "text": "William Shakespeare painted the Mona Lisa.", "query": "Who painted Mona Lisa", "label": 1},
    {"id": "H_J03", "category": "J_entity_swap", "text": "Albert Einstein developed the telescope and discovered Jupiter's moons in 1610.", "query": "Jupiter moons discovery", "label": 1},
    {"id": "H_J04", "category": "J_entity_swap", "text": "Galileo Galilei formulated the theory of general relativity in 1915.", "query": "General relativity", "label": 1},
    {"id": "H_J05", "category": "J_entity_swap", "text": "Madrid is the capital of Italy.", "query": "Capital of Italy", "label": 1},
    {"id": "H_J06", "category": "J_entity_swap", "text": "Rome is the capital of Spain.", "query": "Capital of Spain", "label": 1},
    {"id": "H_J07", "category": "J_entity_swap", "text": "The Amazon River flows through Egypt into the Mediterranean Sea.", "query": "Amazon River Egypt", "label": 1},
    {"id": "H_J08", "category": "J_entity_swap", "text": "The Nile River originates in the Andes and discharges in Brazil.", "query": "Nile River Andes", "label": 1},
    {"id": "H_J09", "category": "J_entity_swap", "text": "Neil Armstrong discovered penicillin in 1928.", "query": "Penicillin discovery", "label": 1},
    {"id": "H_J10", "category": "J_entity_swap", "text": "Alexander Fleming was the first person to walk on the Moon.", "query": "First person on moon", "label": 1},
    {"id": "H_J11", "category": "J_entity_swap", "text": "Ludwig van Beethoven invented the telephone.", "query": "Telephone invention", "label": 1},
    {"id": "H_J12", "category": "J_entity_swap", "text": "Alexander Graham Bell composed the Moonlight Sonata.", "query": "Moonlight Sonata composer", "label": 1},
    {"id": "H_J13", "category": "J_entity_swap", "text": "Isaac Newton sculpted the statue of David.", "query": "Statue of David", "label": 1},
    {"id": "H_J14", "category": "J_entity_swap", "text": "Michelangelo published the Principia Mathematica establishing gravity.", "query": "Principia Mathematica", "label": 1},
    {"id": "H_J15", "category": "J_entity_swap", "text": "Tokyo is the capital of South Korea.", "query": "Capital of South Korea", "label": 1},

    # K_temporal_mutation (15 items)
    {"id": "H_K01", "category": "K_temporal_mutation", "text": "World War II ended in 1776.", "query": "When did World War II end", "label": 1},
    {"id": "H_K02", "category": "K_temporal_mutation", "text": "The United States Declaration of Independence was signed in 1945.", "query": "Declaration Independence year", "label": 1},
    {"id": "H_K03", "category": "K_temporal_mutation", "text": "The Apollo 11 moon landing occurred in 1492.", "query": "Apollo 11 moon landing year", "label": 1},
    {"id": "H_K04", "category": "K_temporal_mutation", "text": "Christopher Columbus reached the Americas in 1969.", "query": "Columbus voyage year", "label": 1},
    {"id": "H_K05", "category": "K_temporal_mutation", "text": "The French Revolution began in 2010.", "query": "French Revolution year", "label": 1},
    {"id": "H_K06", "category": "K_temporal_mutation", "text": "The fall of the Western Roman Empire occurred in 1989.", "query": "Fall of Rome year", "label": 1},
    {"id": "H_K07", "category": "K_temporal_mutation", "text": "The Berlin Wall fell in 476 AD.", "query": "Fall of Berlin Wall year", "label": 1},
    {"id": "H_K08", "category": "K_temporal_mutation", "text": "The Titanic sank in the year 2020.", "query": "Titanic sinking year", "label": 1},
    {"id": "H_K09", "category": "K_temporal_mutation", "text": "The first iPhone was released by Apple in 1805.", "query": "iPhone release year", "label": 1},
    {"id": "H_K10", "category": "K_temporal_mutation", "text": "The battle of Waterloo was fought in 1999.", "query": "Battle of Waterloo year", "label": 1},
    {"id": "H_K11", "category": "K_temporal_mutation", "text": "The Magna Carta was signed in 1914.", "query": "Magna Carta year", "label": 1},
    {"id": "H_K12", "category": "K_temporal_mutation", "text": "World War I began in 1215.", "query": "World War I start year", "label": 1},
    {"id": "H_K13", "category": "K_temporal_mutation", "text": "The Chernobyl disaster occurred in 1650.", "query": "Chernobyl disaster year", "label": 1},
    {"id": "H_K14", "category": "K_temporal_mutation", "text": "The Industrial Revolution started in 300 BC.", "query": "Industrial Revolution year", "label": 1},
    {"id": "H_K15", "category": "K_temporal_mutation", "text": "Albert Einstein won the Nobel Prize in Physics in 1450.", "query": "Einstein Nobel Prize year", "label": 1},

    # L_negation (10 items)
    {"id": "H_L01", "category": "L_negation", "text": "Paris is not the capital of France.", "query": "Capital of France", "label": 1},
    {"id": "H_L02", "category": "L_negation", "text": "Water does not freeze at 0 degrees Celsius.", "query": "Water freezing point", "label": 1},
    {"id": "H_L03", "category": "L_negation", "text": "Jupiter is not the largest planet in our solar system.", "query": "Largest planet", "label": 1},
    {"id": "H_L04", "category": "L_negation", "text": "The Earth does not orbit around the Sun.", "query": "Earth orbit Sun", "label": 1},
    {"id": "H_L05", "category": "L_negation", "text": "Oxygen is not required for human respiration.", "query": "Oxygen human respiration", "label": 1},
    {"id": "H_L06", "category": "L_negation", "text": "Albert Einstein did not contribute to the theory of relativity.", "query": "Einstein relativity", "label": 1},
    {"id": "H_L07", "category": "L_negation", "text": "The chemical formula for water is not H2O.", "query": "Water chemical formula", "label": 1},
    {"id": "H_L08", "category": "L_negation", "text": "Mount Everest is not the highest mountain above sea level.", "query": "Mount Everest highest", "label": 1},
    {"id": "H_L09", "category": "L_negation", "text": "Tokyo is not located in Japan.", "query": "Tokyo Japan", "label": 1},
    {"id": "H_L10", "category": "L_negation", "text": "William Shakespeare was not a playwright.", "query": "Shakespeare playwright", "label": 1},
]


def build_phase52_dataset():
    all_examples = FACTUAL_EXAMPLES + HALLUCINATED_EXAMPLES

    cat_counts = {}
    for ex in all_examples:
        c = ex["category"]
        cat_counts[c] = cat_counts.get(c, 0) + 1

    n_fact = sum(1 for x in all_examples if x["label"] == 0)
    n_hall = sum(1 for x in all_examples if x["label"] == 1)

    payload = {
        "dataset_version": "phase52_balanced_50_50_v1",
        "total_examples": len(all_examples),
        "total_factual": n_fact,
        "total_hallucinated": n_hall,
        "class_balance_ratio": f"{n_fact}/{n_hall} (1.000)",
        "category_counts": cat_counts,
        "examples": all_examples,
    }

    out_json = OUT_DIR / "forensic_50_50_dataset.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    md_content = f"""# PHASE 52 — CONTROLLED 50/50 BALANCED FORENSIC DATASET
**Stratified Diagnostic Dataset Specification ($N=300$)**
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-02
**Status**: `FROZEN 50/50 FORENSIC DATASET`

---

## 1. Class Balance & Stratification Summary

- **Total Samples ($N$)**: **300**
- **Factual / Non-Hallucinated ($y=0$)**: **150 (50.0%)**
- **Hallucinated / Contradictory ($y=1$)**: **150 (50.0%)**
- **Exact Class Ratio**: **1.000 : 1.000** (Zero class imbalance skew)

### Stratification Matrix:

| Class | Category Name | Count | Percentage |
| :--- | :--- | :--- | :--- |
| **Factual ($y=0$)** | `A_clearly_factual` | 40 | 13.33% |
| **Factual ($y=0$)** | `M_paraphrase` | 40 | 13.33% |
| **Factual ($y=0$)** | `H_numerical_correctness` | 35 | 11.67% |
| **Factual ($y=0$)** | `G_multi_claim_consistency` | 35 | 11.67% |
| *Subtotal Factual* | *All 4 Factual Categories* | **150** | **50.00%** |
| **Hallucinated ($y=1$)** | `B_clearly_false` | 20 | 6.67% |
| **Hallucinated ($y=1$)** | `C_direct_contradiction` | 20 | 6.67% |
| **Hallucinated ($y=1$)** | `D_unsupported_claim` | 20 | 6.67% |
| **Hallucinated ($y=1$)** | `E_ambiguous_claim` | 15 | 5.00% |
| **Hallucinated ($y=1$)** | `F_multi_claim_contradiction`| 15 | 5.00% |
| **Hallucinated ($y=1$)** | `I_numerical_error` | 20 | 6.67% |
| **Hallucinated ($y=1$)** | `J_entity_swap` | 15 | 5.00% |
| **Hallucinated ($y=1$)** | `K_temporal_mutation` | 15 | 5.00% |
| **Hallucinated ($y=1$)** | `L_negation` | 10 | 3.33% |
| *Subtotal Hallucinated* | *All 9 Hallucinated Categories*| **150** | **50.00%** |
| **TOTAL** | **All 13 Stratified Categories** | **300** | **100.00%** |

---

## 2. Artifact Path
- `backend/reports/phase52/forensic_50_50_dataset.json`
"""

    out_md = OUT_DIR / "PHASE52_DIAGNOSTIC_DATASET.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Generated Phase 52 50/50 dataset: {len(all_examples)} examples ({n_fact} Factual, {n_hall} Hallucinated).")
    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")

if __name__ == "__main__":
    build_phase52_dataset()

"""Phase 51 — Stratified Diagnostic Dataset Generator.

Constructs 280 balanced, domain-diverse examples (20 per category across 14 categories):
A. Clearly factual (label=0)
B. Clearly false (label=1)
C. Direct contradiction (label=1)
D. Unsupported claim (label=1)
E. Ambiguous claim (label=1)
F. Multi-claim contradiction (label=1)
G. Multi-claim consistency (label=0)
H. Numerical correctness (label=0)
I. Numerical error (label=1)
J. Entity swap (label=1)
K. Temporal mutation (label=1)
L. Negation (label=1)
M. Paraphrase (label=0)
N. Unsupported causal claim (label=1)

Outputs:
- backend/reports/phase51/diagnostic_dataset.json
- backend/reports/phase51/PHASE51_DIAGNOSTIC_DATASET.md
"""

import json
from pathlib import Path
from typing import List, Dict, Any

OUT_DIR = Path("backend/reports/phase51")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES_DATA: Dict[str, List[Dict[str, Any]]] = {
    # -------------------------------------------------------------
    # Category A: Clearly factual (20 examples, label=0)
    # -------------------------------------------------------------
    "A_clearly_factual": [
        {"id": "A_01", "text": "The capital of France is Paris.", "query": "What is the capital of France?", "label": 0},
        {"id": "A_02", "text": "Water freezes at 0 degrees Celsius under standard atmospheric pressure.", "query": "At what temperature does water freeze?", "label": 0},
        {"id": "A_03", "text": "The chemical formula for water is H2O.", "query": "What is the chemical formula for water?", "label": 0},
        {"id": "A_04", "text": "Jupiter is the largest planet in our solar system.", "query": "Which is the largest planet?", "label": 0},
        {"id": "A_05", "text": "Albert Einstein developed the theory of general relativity.", "query": "Who developed general relativity?", "label": 0},
        {"id": "A_06", "text": "The speed of light in a vacuum is approximately 299,792,458 meters per second.", "query": "What is the speed of light?", "label": 0},
        {"id": "A_07", "text": "Oxygen has atomic number 8.", "query": "What is the atomic number of oxygen?", "label": 0},
        {"id": "A_08", "text": "Photosynthesis converts sunlight, carbon dioxide, and water into glucose and oxygen.", "query": "What is photosynthesis?", "label": 0},
        {"id": "A_09", "text": "The Pacific Ocean is the largest ocean on Earth.", "query": "What is the largest ocean?", "label": 0},
        {"id": "A_10", "text": "William Shakespeare wrote Hamlet.", "query": "Who wrote Hamlet?", "label": 0},
        {"id": "A_11", "text": "The human body typically has 206 bones in adulthood.", "query": "How many bones in the human body?", "label": 0},
        {"id": "A_12", "text": "Tokyo is the capital of Japan.", "query": "What is the capital of Japan?", "label": 0},
        {"id": "A_13", "text": "DNA consists of two strands forming a double helix.", "query": "What is the structure of DNA?", "label": 0},
        {"id": "A_14", "text": "The Mona Lisa was painted by Leonardo da Vinci.", "query": "Who painted the Mona Lisa?", "label": 0},
        {"id": "A_15", "text": "Mount Everest is the highest mountain above sea level.", "query": "What is the highest mountain?", "label": 0},
        {"id": "A_16", "text": "Helium is a noble gas with atomic number 2.", "query": "What is helium?", "label": 0},
        {"id": "A_17", "text": "The Amazon rainforest is predominantly located in Brazil.", "query": "Where is the Amazon rainforest?", "label": 0},
        {"id": "A_18", "text": "The Sahara is the largest hot desert in the world.", "query": "What is the largest hot desert?", "label": 0},
        {"id": "A_19", "text": "Gold is a chemical element with symbol Au.", "query": "What is the symbol for gold?", "label": 0},
        {"id": "A_20", "text": "The Great Barrier Reef is located off the coast of Australia.", "query": "Where is the Great Barrier Reef?", "label": 0},
    ],

    # -------------------------------------------------------------
    # Category B: Clearly false (20 examples, label=1)
    # -------------------------------------------------------------
    "B_clearly_false": [
        {"id": "B_01", "text": "The capital of France is Berlin.", "query": "What is the capital of France?", "label": 1},
        {"id": "B_02", "text": "Water freezes at 100 degrees Celsius under standard atmospheric pressure.", "query": "At what temperature does water freeze?", "label": 1},
        {"id": "B_03", "text": "The chemical formula for water is CO2.", "query": "What is the formula for water?", "label": 1},
        {"id": "B_04", "text": "Mars is the largest planet in our solar system.", "query": "Which is the largest planet?", "label": 1},
        {"id": "B_05", "text": "Isaac Newton developed the theory of general relativity.", "query": "Who developed general relativity?", "label": 1},
        {"id": "B_06", "text": "The speed of light in a vacuum is 500 meters per second.", "query": "What is the speed of light?", "label": 1},
        {"id": "B_07", "text": "Oxygen has atomic number 79.", "query": "What is the atomic number of oxygen?", "label": 1},
        {"id": "B_08", "text": "Photosynthesis occurs exclusively in animal cells.", "query": "What is photosynthesis?", "label": 1},
        {"id": "B_09", "text": "The Atlantic Ocean is the smallest ocean on Earth.", "query": "What is the largest ocean?", "label": 1},
        {"id": "B_10", "text": "Charles Dickens wrote Hamlet.", "query": "Who wrote Hamlet?", "label": 1},
        {"id": "B_11", "text": "The human body contains over 5,000 bones in adulthood.", "query": "How many bones in the human body?", "label": 1},
        {"id": "B_12", "text": "Beijing is the capital of Japan.", "query": "What is the capital of Japan?", "label": 1},
        {"id": "B_13", "text": "DNA is composed of a single straight protein strand with no nucleotides.", "query": "What is DNA?", "label": 1},
        {"id": "B_14", "text": "Pablo Picasso painted the Mona Lisa in 1985.", "query": "Who painted the Mona Lisa?", "label": 1},
        {"id": "B_15", "text": "Mount Fuji is the highest mountain on Earth.", "query": "What is the highest mountain?", "label": 1},
        {"id": "B_16", "text": "Helium is an alkali metal that burns vigorously in air.", "query": "What is helium?", "label": 1},
        {"id": "B_17", "text": "The Amazon rainforest is located in central Germany.", "query": "Where is the Amazon?", "label": 1},
        {"id": "B_18", "text": "The Sahara is an Arctic tundra covered in perpetual glacier ice.", "query": "What is the Sahara?", "label": 1},
        {"id": "B_19", "text": "Gold has the chemical symbol Fe and rusts easily.", "query": "What is gold?", "label": 1},
        {"id": "B_20", "text": "The Great Barrier Reef is a volcanic mountain range in Switzerland.", "query": "Where is the Great Barrier Reef?", "label": 1},
    ],

    # -------------------------------------------------------------
    # Category C: Direct contradiction (20 examples, label=1)
    # -------------------------------------------------------------
    "C_direct_contradiction": [
        {"id": "C_01", "text": "The earth is entirely flat and has no curvature.", "query": "Is the earth round?", "label": 1},
        {"id": "C_02", "text": "Humans do not require oxygen to survive.", "query": "Do humans need oxygen?", "label": 1},
        {"id": "C_03", "text": "The sun orbits around the Earth once every 24 hours.", "query": "Does the earth orbit the sun?", "label": 1},
        {"id": "C_04", "text": "Sound travels faster in a complete vacuum than through solid steel.", "query": "How does sound travel?", "label": 1},
        {"id": "C_05", "text": "Absolute zero temperature is hotter than the core of the Sun.", "query": "What is absolute zero?", "label": 1},
        {"id": "C_06", "text": "Mammals are cold-blooded creatures that lay shelled eggs exclusively.", "query": "Are mammals warm-blooded?", "label": 1},
        {"id": "C_07", "text": "Light cannot travel through empty space.", "query": "Can light travel in vacuum?", "label": 1},
        {"id": "C_08", "text": "Pure water is a strong acidic substance with pH 1.0.", "query": "What is the pH of water?", "label": 1},
        {"id": "C_09", "text": "Gravity repels masses away from each other proportionally to distance.", "query": "What is gravity?", "label": 1},
        {"id": "C_10", "text": "Diamonds are made entirely of pure liquid nitrogen.", "query": "What are diamonds made of?", "label": 1},
        {"id": "C_11", "text": "The Pacific ocean contains no liquid water.", "query": "Pacific ocean", "label": 1},
        {"id": "C_12", "text": "Electrons have a strong positive electrical charge.", "query": "What charge do electrons have?", "label": 1},
        {"id": "C_13", "text": "Plants produce carbon dioxide and consume pure methane during photosynthesis.", "query": "Photosynthesis", "label": 1},
        {"id": "C_14", "text": "The Moon has a larger mass and volume than the Sun.", "query": "Is the moon bigger than the sun?", "label": 1},
        {"id": "C_15", "text": "The human brain contains zero neurons.", "query": "Human brain", "label": 1},
        {"id": "C_16", "text": "Antarctica is the hottest tropical rainforest on Earth.", "query": "What is Antarctica?", "label": 1},
        {"id": "C_17", "text": "Iron is a gas at room temperature.", "query": "What state is iron?", "label": 1},
        {"id": "C_18", "text": "All living organisms are completely devoid of cells.", "query": "Cell theory", "label": 1},
        {"id": "C_19", "text": "The speed of sound exceeds the speed of light in a vacuum.", "query": "Speed of sound vs light", "label": 1},
        {"id": "C_20", "text": "The Eiffel Tower was carved out of solid marble in Ancient Egypt.", "query": "Eiffel tower origin", "label": 1},
    ],

    # -------------------------------------------------------------
    # Category D: Unsupported claim (20 examples, label=1)
    # -------------------------------------------------------------
    "D_unsupported_claim": [
        {"id": "D_01", "text": "Napoleon Bonaparte secretly traveled to Australia in 1812 to establish an underground palace.", "query": "Napoleon travels", "label": 1},
        {"id": "D_02", "text": "Ancient Romans invented quantum computers powered by steam in 50 BC.", "query": "Roman inventions", "label": 1},
        {"id": "D_03", "text": "Eating purple cabbage allows humans to communicate telepathically across galaxies.", "query": "Purple cabbage effects", "label": 1},
        {"id": "D_04", "text": "Shakespeare owned a domesticated penguin named Bartholomew in Stratford-upon-Avon.", "query": "Shakespeare pets", "label": 1},
        {"id": "D_05", "text": "The core of the planet Saturn is composed of solid milk chocolate.", "query": "Saturn core composition", "label": 1},
        {"id": "D_06", "text": "Cleopatra invented the electric guitar during her reign in Alexandria.", "query": "Cleopatra inventions", "label": 1},
        {"id": "D_07", "text": "A secret civilization of giant badger monks operates beneath the Antarctic ice sheet.", "query": "Antarctic discoveries", "label": 1},
        {"id": "D_08", "text": "George Washington invented the internet in 1789 to send encrypted messages to Thomas Jefferson.", "query": "George Washington internet", "label": 1},
        {"id": "D_09", "text": "Pluto is made entirely of compressed titanium and emits classical violin music.", "query": "Pluto characteristics", "label": 1},
        {"id": "D_10", "text": "The Statue of Liberty was originally built as a wireless charging tower for steam locomotives.", "query": "Statue of Liberty purpose", "label": 1},
        {"id": "D_11", "text": "Drinking seawater gives humans the biological ability to breathe underwater indefinitely.", "query": "Drinking seawater", "label": 1},
        {"id": "D_12", "text": "Beethoven composed his 9th symphony while scuba diving in the Pacific Ocean.", "query": "Beethoven 9th symphony", "label": 1},
        {"id": "D_13", "text": "Sunlight contains micro-crystals of ruby that give birds the power of flight.", "query": "How birds fly", "label": 1},
        {"id": "D_14", "text": "The Great Pyramids were originally painted neon pink with fluorescent dyes.", "query": "Great Pyramids color", "label": 1},
        {"id": "D_15", "text": "Galileo used an optical laser pointer to communicate with Martian settlers.", "query": "Galileo laser", "label": 1},
        {"id": "D_16", "text": "The Amazon river flows backwards every leap year due to lunar gravitational anomalies.", "query": "Amazon river flow", "label": 1},
        {"id": "D_17", "text": "Julius Caesar was an accomplished jazz saxophonist in 44 BC.", "query": "Julius Caesar music", "label": 1},
        {"id": "D_18", "text": "Clouds are solid fiberglass formations suspended by magnetic levitation.", "query": "What are clouds?", "label": 1},
        {"id": "D_19", "text": "Alexander the Great discovered nuclear fission using bronze pottery.", "query": "Alexander the Great physics", "label": 1},
        {"id": "D_20", "text": "Koalas can run at speeds exceeding 120 miles per hour when hunting gazelles.", "query": "Koala running speed", "label": 1},
    ],

    # -------------------------------------------------------------
    # Category E: Ambiguous claim (20 examples, label=1)
    # -------------------------------------------------------------
    "E_ambiguous_claim": [
        {"id": "E_01", "text": "It has been said that something happened somewhere in Europe many centuries ago that changed everything.", "query": "European history", "label": 1},
        {"id": "E_02", "text": "Certain mysterious cosmic energy pulses might be responsible for all unknown phenomena on Earth.", "query": "Cosmic energy", "label": 1},
        {"id": "E_03", "text": "Some ancient philosopher proved that reality is just an illusion created by vibrational frequencies.", "query": "Philosophy", "label": 1},
        {"id": "E_04", "text": "Scientists may have discovered a secret element that defies all known laws of physics.", "query": "New elements", "label": 1},
        {"id": "E_05", "text": "An obscure historical figure allegedly achieved immortality through natural alchemy.", "query": "Alchemy history", "label": 1},
        {"id": "E_06", "text": "There are rumors that the weather in certain remote islands is controlled by ancient mechanisms.", "query": "Weather control", "label": 1},
        {"id": "E_07", "text": "Certain frequencies of sound are capable of manipulating human thoughts instantaneously.", "query": "Sound frequency thoughts", "label": 1},
        {"id": "E_08", "text": "A famous king in the Middle Ages was actually two different people acting as one person.", "query": "Medieval kings", "label": 1},
        {"id": "E_09", "text": "Some studies suggest that trees have a hidden collective consciousness with secret plans.", "query": "Tree communication", "label": 1},
        {"id": "E_10", "text": "Mysterious forces at the center of the Earth regulate the rotation of the Milky Way galaxy.", "query": "Earth center", "label": 1},
        {"id": "E_11", "text": "Ancient texts suggest that humans once flew using specialized vocal harmonies.", "query": "Ancient texts flight", "label": 1},
        {"id": "E_12", "text": "Certain minerals can produce unlimited electrical power if exposed to starlight.", "query": "Mineral power", "label": 1},
        {"id": "E_13", "text": "An unidentified historical empire conquered all continents before disappearing without a trace.", "query": "Lost empires", "label": 1},
        {"id": "E_14", "text": "The moon might be hollow and occupied by forgotten historical expeditions.", "query": "Hollow moon", "label": 1},
        {"id": "E_15", "text": "Some obscure mathematical equations have the physical power to alter gravitational fields.", "query": "Math and gravity", "label": 1},
        {"id": "E_16", "text": "It is widely speculated by unnamed researchers that time travel was solved in 1920.", "query": "Time travel research", "label": 1},
        {"id": "E_17", "text": "Certain animals possess an innate sixth sense enabling them to predict historical stock markets.", "query": "Animal senses", "label": 1},
        {"id": "E_18", "text": "A hidden library containing all lost knowledge is buried beneath a random desert dune.", "query": "Lost libraries", "label": 1},
        {"id": "E_19", "text": "Water memory enables ordinary tap water to remember human emotions across centuries.", "query": "Water memory", "label": 1},
        {"id": "E_20", "text": "The pyramids were designed to emit invisible cosmic rays to preserve pharaonic spirits.", "query": "Pyramid rays", "label": 1},
    ],

    # -------------------------------------------------------------
    # Category F: Multi-claim contradiction (20 examples, label=1)
    # -------------------------------------------------------------
    "F_multi_claim_contradiction": [
        {"id": "F_01", "text": "Paris is the capital of France. Berlin is the capital of France.", "query": "Capitals", "label": 1},
        {"id": "F_02", "text": "Water freezes at 0 degrees Celsius. Water only freezes at 100 degrees Celsius.", "query": "Water freezing point", "label": 1},
        {"id": "F_03", "text": "The sun rises in the east. The sun never rises in the east and only rises in the west.", "query": "Sun rising", "label": 1},
        {"id": "F_04", "text": "Jupiter is a gas giant. Jupiter has a solid iron surface with no atmosphere.", "query": "Jupiter planet", "label": 1},
        {"id": "F_05", "text": "Albert Einstein was born in Germany. Albert Einstein was born on the moon in 2050.", "query": "Einstein birthplace", "label": 1},
        {"id": "F_06", "text": "Gold is a metal. Gold is a non-metallic organic liquid.", "query": "Gold properties", "label": 1},
        {"id": "F_07", "text": "Mount Everest is in the Himalayas. Mount Everest is located entirely in Brazil.", "query": "Mount Everest location", "label": 1},
        {"id": "F_08", "text": "Humans are mammals. Humans belong to the reptile class.", "query": "Human taxonomy", "label": 1},
        {"id": "F_09", "text": "Tokyo is in Japan. Tokyo is located in central Canada.", "query": "Tokyo location", "label": 1},
        {"id": "F_10", "text": "The Pacific Ocean is filled with saltwater. The Pacific Ocean contains purely freshwater.", "query": "Pacific Ocean water", "label": 1},
        {"id": "F_11", "text": "Oxygen is necessary for aerobic respiration. Oxygen is completely toxic and never used in respiration.", "query": "Oxygen respiration", "label": 1},
        {"id": "F_12", "text": "Shakespeare was an English playwright. Shakespeare never wrote in English and lived in Tokyo.", "query": "Shakespeare language", "label": 1},
        {"id": "F_13", "text": "The speed of light is finite. The speed of light is infinite and instantaneous.", "query": "Speed of light", "label": 1},
        {"id": "F_14", "text": "DNA has a double helix structure. DNA is completely unstructured and contains no strands.", "query": "DNA structure", "label": 1},
        {"id": "F_15", "text": "The Nile River is in Africa. The Nile River flows exclusively through Antarctica.", "query": "Nile River location", "label": 1},
        {"id": "F_16", "text": "Diamond is an allotrope of carbon. Diamond contains zero carbon atoms.", "query": "Diamond composition", "label": 1},
        {"id": "F_17", "text": "Helium is lighter than air. Helium is heavier than solid lead.", "query": "Helium density", "label": 1},
        {"id": "F_18", "text": "Leonardo da Vinci painted the Mona Lisa. Leonardo da Vinci was a painter who never produced any artwork.", "query": "Leonardo artwork", "label": 1},
        {"id": "F_19", "text": "The Amazon is a river. The Amazon is a dry subterranean cave with zero water.", "query": "Amazon river", "label": 1},
        {"id": "F_20", "text": "Earth has one natural moon. Earth has fifteen natural gas moons.", "query": "Earth moons", "label": 1},
    ],

    # -------------------------------------------------------------
    # Category G: Multi-claim consistency (20 examples, label=0)
    # -------------------------------------------------------------
    "G_multi_claim_consistency": [
        {"id": "G_01", "text": "Paris is the capital of France. Berlin is the capital of Germany.", "query": "Capitals", "label": 0},
        {"id": "G_02", "text": "Water freezes at 0 degrees Celsius. Water boils at 100 degrees Celsius under standard pressure.", "query": "Water phase changes", "label": 0},
        {"id": "G_03", "text": "Jupiter is the largest planet in our solar system. Saturn is famous for its prominent ring system.", "query": "Gas giants", "label": 0},
        {"id": "G_04", "text": "Mount Everest is Earth's highest mountain above sea level. K2 is the second-highest mountain.", "query": "Highest mountains", "label": 0},
        {"id": "G_05", "text": "The Nile is a major north-flowing river in Africa. The Amazon is the largest river by discharge.", "query": "World rivers", "label": 0},
        {"id": "G_06", "text": "Oxygen has atomic number 8. Carbon has atomic number 6.", "query": "Atomic numbers", "label": 0},
        {"id": "G_07", "text": "William Shakespeare wrote Hamlet. He also wrote Macbeth.", "query": "Shakespeare plays", "label": 0},
        {"id": "G_08", "text": "Tokyo is the capital of Japan. Rome is the capital of Italy.", "query": "World capitals", "label": 0},
        {"id": "G_09", "text": "The Pacific is the largest ocean. The Atlantic is the second-largest ocean.", "query": "Oceans", "label": 0},
        {"id": "G_10", "text": "Albert Einstein developed general relativity. Isaac Newton formulated the laws of universal gravitation.", "query": "Physicists", "label": 0},
        {"id": "G_11", "text": "DNA stores genetic information. RNA plays a key role in protein synthesis.", "query": "Nucleic acids", "label": 0},
        {"id": "G_12", "text": "Helium is a noble gas. Neon is also a noble gas.", "query": "Noble gases", "label": 0},
        {"id": "G_13", "text": "The Sahara is the largest hot desert. Antarctica is classified as a polar desert.", "query": "Deserts", "label": 0},
        {"id": "G_14", "text": "Leonardo da Vinci painted the Mona Lisa. Michelangelo sculpted David.", "query": "Renaissance artists", "label": 0},
        {"id": "G_15", "text": "Gold has symbol Au. Silver has symbol Ag.", "query": "Chemical symbols", "label": 0},
        {"id": "G_16", "text": "The human heart pumps blood. The lungs facilitate gas exchange.", "query": "Human organs", "label": 0},
        {"id": "G_17", "text": "Mercury is closest to the Sun. Neptune is the eighth planet from the Sun.", "query": "Solar system planets", "label": 0},
        {"id": "G_18", "text": "Photosynthesis produces glucose. Cellular respiration breaks down glucose to release energy.", "query": "Biological processes", "label": 0},
        {"id": "G_19", "text": "Madrid is the capital of Spain. Lisbon is the capital of Portugal.", "query": "Iberian capitals", "label": 0},
        {"id": "G_20", "text": "Iron is a magnetic transition metal. Copper is an excellent conductor of electricity.", "query": "Metals", "label": 0},
    ],

    # -------------------------------------------------------------
    # Category H: Numerical correctness (20 examples, label=0)
    # -------------------------------------------------------------
    "H_numerical_correctness": [
        {"id": "H_01", "text": "12 multiplied by 8 equals 96.", "query": "12*8", "label": 0},
        {"id": "H_02", "text": "15 plus 27 equals 42.", "query": "15+27", "label": 0},
        {"id": "H_03", "text": "100 divided by 4 equals 25.", "query": "100/4", "label": 0},
        {"id": "H_04", "text": "50 minus 18 equals 32.", "query": "50-18", "label": 0},
        {"id": "H_05", "text": "7 multiplied by 9 equals 63.", "query": "7*9", "label": 0},
        {"id": "H_06", "text": "144 divided by 12 equals 12.", "query": "144/12", "label": 0},
        {"id": "H_07", "text": "25 plus 75 equals 100.", "query": "25+75", "label": 0},
        {"id": "H_08", "text": "9 multiplied by 9 equals 81.", "query": "9*9", "label": 0},
        {"id": "H_09", "text": "80 divided by 8 equals 10.", "query": "80/8", "label": 0},
        {"id": "H_10", "text": "64 minus 28 equals 36.", "query": "64-28", "label": 0},
        {"id": "H_11", "text": "11 multiplied by 11 equals 121.", "query": "11*11", "label": 0},
        {"id": "H_12", "text": "200 minus 45 equals 155.", "query": "200-45", "label": 0},
        {"id": "H_13", "text": "6 multiplied by 7 equals 42.", "query": "6*7", "label": 0},
        {"id": "H_14", "text": "81 divided by 9 equals 9.", "query": "81/9", "label": 0},
        {"id": "H_15", "text": "30 plus 70 equals 100.", "query": "30+70", "label": 0},
        {"id": "H_16", "text": "16 multiplied by 4 equals 64.", "query": "16*4", "label": 0},
        {"id": "H_17", "text": "500 divided by 5 equals 100.", "query": "500/5", "label": 0},
        {"id": "H_18", "text": "13 plus 19 equals 32.", "query": "13+19", "label": 0},
        {"id": "H_19", "text": "48 divided by 6 equals 8.", "query": "48/6", "label": 0},
        {"id": "H_20", "text": "8 multiplied by 5 equals 40.", "query": "8*5", "label": 0},
    ],

    # -------------------------------------------------------------
    # Category I: Numerical error (20 examples, label=1)
    # -------------------------------------------------------------
    "I_numerical_error": [
        {"id": "I_01", "text": "12 multiplied by 8 equals 95.", "query": "12*8", "label": 1},
        {"id": "I_02", "text": "15 plus 27 equals 49.", "query": "15+27", "label": 1},
        {"id": "I_03", "text": "100 divided by 4 equals 26.", "query": "100/4", "label": 1},
        {"id": "I_04", "text": "50 minus 18 equals 35.", "query": "50-18", "label": 1},
        {"id": "I_05", "text": "7 multiplied by 9 equals 65.", "query": "7*9", "label": 1},
        {"id": "I_06", "text": "144 divided by 12 equals 14.", "query": "144/12", "label": 1},
        {"id": "I_07", "text": "25 plus 75 equals 110.", "query": "25+75", "label": 1},
        {"id": "I_08", "text": "9 multiplied by 9 equals 83.", "query": "9*9", "label": 1},
        {"id": "I_09", "text": "80 divided by 8 equals 12.", "query": "80/8", "label": 1},
        {"id": "I_10", "text": "64 minus 28 equals 40.", "query": "64-28", "label": 1},
        {"id": "I_11", "text": "11 multiplied by 11 equals 125.", "query": "11*11", "label": 1},
        {"id": "I_12", "text": "200 minus 45 equals 160.", "query": "200-45", "label": 1},
        {"id": "I_13", "text": "6 multiplied by 7 equals 45.", "query": "6*7", "label": 1},
        {"id": "I_14", "text": "81 divided by 9 equals 8.", "query": "81/9", "label": 1},
        {"id": "I_15", "text": "30 plus 70 equals 90.", "query": "30+70", "label": 1},
        {"id": "I_16", "text": "16 multiplied by 4 equals 60.", "query": "16*4", "label": 1},
        {"id": "I_17", "text": "500 divided by 5 equals 105.", "query": "500/5", "label": 1},
        {"id": "I_18", "text": "13 plus 19 equals 35.", "query": "13+19", "label": 1},
        {"id": "I_19", "text": "48 divided by 6 equals 9.", "query": "48/6", "label": 1},
        {"id": "I_20", "text": "8 multiplied by 5 equals 45.", "query": "8*5", "label": 1},
    ],

    # -------------------------------------------------------------
    # Category J: Entity swap (20 examples, label=1)
    # -------------------------------------------------------------
    "J_entity_swap": [
        {"id": "J_01", "text": "Leonardo da Vinci wrote the tragedy Hamlet.", "query": "Who wrote Hamlet?", "label": 1},
        {"id": "J_02", "text": "William Shakespeare painted the Mona Lisa.", "query": "Who painted Mona Lisa?", "label": 1},
        {"id": "J_03", "text": "Albert Einstein developed the telescope and discovered Jupiter's moons in 1610.", "query": "Jupiter moons discovery", "label": 1},
        {"id": "J_04", "text": "Galileo Galilei formulated the theory of general relativity in 1915.", "query": "General relativity", "label": 1},
        {"id": "J_05", "text": "Madrid is the capital of Italy.", "query": "Capital of Italy", "label": 1},
        {"id": "J_06", "text": "Rome is the capital of Spain.", "query": "Capital of Spain", "label": 1},
        {"id": "J_07", "text": "The Amazon River flows through Egypt into the Mediterranean Sea.", "query": "Amazon River", "label": 1},
        {"id": "J_08", "text": "The Nile River originates in the Andes and discharges in Brazil.", "query": "Nile River", "label": 1},
        {"id": "J_09", "text": "Neil Armstrong discovered penicillin in 1928.", "query": "Discovery of penicillin", "label": 1},
        {"id": "J_10", "text": "Alexander Fleming was the first person to walk on the Moon.", "query": "First person on Moon", "label": 1},
        {"id": "J_11", "text": "Ludwig van Beethoven invented the telephone.", "query": "Telephone invention", "label": 1},
        {"id": "J_12", "text": "Alexander Graham Bell composed the Moonlight Sonata.", "query": "Moonlight Sonata", "label": 1},
        {"id": "J_13", "text": "Isaac Newton sculpted the statue of David.", "query": "Statue of David", "label": 1},
        {"id": "J_14", "text": "Michelangelo published the Principia Mathematica establishing gravity.", "query": "Principia Mathematica", "label": 1},
        {"id": "J_15", "text": "Tokyo is the capital of South Korea.", "query": "Capital of South Korea", "label": 1},
        {"id": "J_16", "text": "Seoul is the capital of Japan.", "query": "Capital of Japan", "label": 1},
        {"id": "J_17", "text": "Charles Darwin founded Microsoft in 1975.", "query": "Microsoft founder", "label": 1},
        {"id": "J_18", "text": "Bill Gates published On the Origin of Species in 1859.", "query": "Origin of Species", "label": 1},
        {"id": "J_19", "text": "Mars is known as the Blue Planet due to its vast liquid oceans.", "query": "Blue Planet", "label": 1},
        {"id": "J_20", "text": "Earth is known as the Red Planet due to its iron oxide surface dust.", "query": "Red Planet", "label": 1},
    ],

    # -------------------------------------------------------------
    # Category K: Temporal mutation (20 examples, label=1)
    # -------------------------------------------------------------
    "K_temporal_mutation": [
        {"id": "K_01", "text": "World War II ended in 1776.", "query": "When did World War II end?", "label": 1},
        {"id": "K_02", "text": "The United States Declaration of Independence was signed in 1945.", "query": "Declaration of Independence year", "label": 1},
        {"id": "K_03", "text": "The Apollo 11 moon landing occurred in 1492.", "query": "Apollo 11 year", "label": 1},
        {"id": "K_04", "text": "Christopher Columbus reached the Americas in 1969.", "query": "Columbus voyage year", "label": 1},
        {"id": "K_05", "text": "The French Revolution began in 2010.", "query": "French Revolution year", "label": 1},
        {"id": "K_06", "text": "The fall of the Western Roman Empire occurred in 1989.", "query": "Fall of Rome", "label": 1},
        {"id": "K_07", "text": "The Berlin Wall fell in 476 AD.", "query": "Fall of Berlin Wall", "label": 1},
        {"id": "K_08", "text": "The Titanic sank in the year 2020.", "query": "Titanic sinking year", "label": 1},
        {"id": "K_09", "text": "The first iPhone was released by Apple in 1805.", "query": "iPhone release year", "label": 1},
        {"id": "K_10", "text": "The battle of Waterloo was fought in 1999.", "query": "Battle of Waterloo", "label": 1},
        {"id": "K_11", "text": "The Magna Carta was signed in 1914.", "query": "Magna Carta year", "label": 1},
        {"id": "K_12", "text": "World War I began in 1215.", "query": "World War I start year", "label": 1},
        {"id": "K_13", "text": "The Chernobyl disaster occurred in 1650.", "query": "Chernobyl year", "label": 1},
        {"id": "K_14", "text": "The Industrial Revolution started in 300 BC.", "query": "Industrial Revolution", "label": 1},
        {"id": "K_15", "text": "Albert Einstein won the Nobel Prize in Physics in 1450.", "query": "Einstein Nobel Prize", "label": 1},
        {"id": "K_16", "text": "The first heavier-than-air airplane flight by the Wright brothers happened in 1600.", "query": "Wright brothers flight", "label": 1},
        {"id": "K_17", "text": "The United Nations was established in 1700.", "query": "UN founding year", "label": 1},
        {"id": "K_18", "text": "The American Civil War was fought between 1970 and 1975.", "query": "American Civil War years", "label": 1},
        {"id": "K_19", "text": "Johannes Gutenberg introduced the printing press to Europe in 1995.", "query": "Printing press year", "label": 1},
        {"id": "K_20", "text": "The Hubble Space Telescope was launched in 1890.", "query": "Hubble launch year", "label": 1},
    ],

    # -------------------------------------------------------------
    # Category L: Negation (20 examples, label=1)
    # -------------------------------------------------------------
    "L_negation": [
        {"id": "L_01", "text": "Paris is not the capital of France.", "query": "Capital of France", "label": 1},
        {"id": "L_02", "text": "Water does not freeze at 0 degrees Celsius.", "query": "Water freezing point", "label": 1},
        {"id": "L_03", "text": "Jupiter is not the largest planet in our solar system.", "query": "Largest planet", "label": 1},
        {"id": "L_04", "text": "The Earth does not orbit around the Sun.", "query": "Earth orbit", "label": 1},
        {"id": "L_05", "text": "Oxygen is not required for human respiration.", "query": "Oxygen need", "label": 1},
        {"id": "L_06", "text": "Albert Einstein did not contribute to the theory of relativity.", "query": "Einstein relativity", "label": 1},
        {"id": "L_07", "text": "The chemical formula for water is not H2O.", "query": "Formula for water", "label": 1},
        {"id": "L_08", "text": "Mount Everest is not the highest mountain above sea level.", "query": "Mount Everest height", "label": 1},
        {"id": "L_09", "text": "Tokyo is not located in Japan.", "query": "Tokyo location", "label": 1},
        {"id": "L_10", "text": "William Shakespeare was not a playwright.", "query": "Shakespeare profession", "label": 1},
        {"id": "L_11", "text": "The Pacific Ocean is not the largest ocean on Earth.", "query": "Pacific Ocean size", "label": 1},
        {"id": "L_12", "text": "Diamonds are not composed of carbon atoms.", "query": "Diamond composition", "label": 1},
        {"id": "L_13", "text": "The heart does not pump blood through the human body.", "query": "Heart function", "label": 1},
        {"id": "L_14", "text": "Light does not travel at a finite speed in a vacuum.", "query": "Speed of light", "label": 1},
        {"id": "L_15", "text": "Gold is not a chemical element.", "query": "Gold element", "label": 1},
        {"id": "L_16", "text": "The Mona Lisa was not created during the Renaissance.", "query": "Mona Lisa period", "label": 1},
        {"id": "L_17", "text": "DNA is not involved in genetic inheritance.", "query": "DNA function", "label": 1},
        {"id": "L_18", "text": "Helium is not an inert gas.", "query": "Helium reactivity", "label": 1},
        {"id": "L_19", "text": "The Amazon is not located in South America.", "query": "Amazon location", "label": 1},
        {"id": "L_20", "text": "The speed of sound is not affected by the medium it travels through.", "query": "Speed of sound medium", "label": 1},
    ],

    # -------------------------------------------------------------
    # Category M: Paraphrase (20 examples, label=0)
    # -------------------------------------------------------------
    "M_paraphrase": [
        {"id": "M_01", "text": "The principal city and government seat of France is Paris.", "query": "Capital of France", "label": 0},
        {"id": "M_02", "text": "At zero degrees on the Celsius scale, pure liquid water turns to solid ice.", "query": "Water freezing point", "label": 0},
        {"id": "M_03", "text": "A water molecule consists of two hydrogen atoms bonded to one oxygen atom.", "query": "Water formula", "label": 0},
        {"id": "M_04", "text": "Among all the planets orbiting our Sun, Jupiter has the greatest volume and mass.", "query": "Largest planet", "label": 0},
        {"id": "M_05", "text": "General relativity was formulated by the theoretical physicist Albert Einstein.", "query": "Who developed relativity?", "label": 0},
        {"id": "M_06", "text": "In a vacuum, photons propagate at a constant velocity of nearly 300,000 kilometers per second.", "query": "Speed of light", "label": 0},
        {"id": "M_07", "text": "The atomic nucleus of an oxygen atom contains exactly eight protons.", "query": "Oxygen atomic number", "label": 0},
        {"id": "M_08", "text": "Plants utilize solar energy to convert carbon dioxide and water into biochemical sugars.", "query": "Photosynthesis", "label": 0},
        {"id": "M_09", "text": "Of all marine bodies on Earth, the Pacific encompasses the largest surface area.", "query": "Pacific Ocean", "label": 0},
        {"id": "M_10", "text": "The tragedy of Prince Hamlet of Denmark was authored by William Shakespeare.", "query": "Hamlet author", "label": 0},
        {"id": "M_11", "text": "The adult human skeletal framework comprises 206 individual bones.", "query": "Human skeleton", "label": 0},
        {"id": "M_12", "text": "The national capital and most populous metropolis of Japan is Tokyo.", "query": "Japan capital", "label": 0},
        {"id": "M_13", "text": "Deoxyribonucleic acid is structured as a double-stranded helical polymer.", "query": "DNA structure", "label": 0},
        {"id": "M_14", "text": "Leonardo da Vinci is the Renaissance master who created the Mona Lisa portrait.", "query": "Mona Lisa artist", "label": 0},
        {"id": "M_15", "text": "The highest terrestrial elevation above mean sea level is the summit of Mount Everest.", "query": "Mount Everest", "label": 0},
        {"id": "M_16", "text": "With two protons in its nucleus, helium is classified as an unreactive noble gas.", "query": "Helium gas", "label": 0},
        {"id": "M_17", "text": "The vast majority of the Amazon basin lies within the borders of Brazil.", "query": "Amazon basin", "label": 0},
        {"id": "M_18", "text": "The Sahara represents the most extensive non-polar arid region on our planet.", "query": "Sahara desert", "label": 0},
        {"id": "M_19", "text": "Represented by the elemental symbol Au, gold is a dense transition metal.", "query": "Gold element", "label": 0},
        {"id": "M_20", "text": "Situated in the Coral Sea off Queensland, the Great Barrier Reef is the largest coral system.", "query": "Great Barrier Reef", "label": 0},
    ],

    # -------------------------------------------------------------
    # Category N: Unsupported causal claim (20 examples, label=1)
    # -------------------------------------------------------------
    "N_unsupported_causal": [
        {"id": "N_01", "text": "The French Revolution occurred because King Louis XVI was secretly a clone created by Martian time travelers.", "query": "French Revolution cause", "label": 1},
        {"id": "N_02", "text": "Dinosaurs went extinct because ancient humans hunted them with laser rifles.", "query": "Dinosaur extinction", "label": 1},
        {"id": "N_03", "text": "The Roman Empire collapsed because emperors drank excessive amounts of caffeinated soda.", "query": "Fall of Rome cause", "label": 1},
        {"id": "N_04", "text": "The Titanic sank because a giant subterranean sea monster attacked the hull.", "query": "Why did Titanic sink?", "label": 1},
        {"id": "N_05", "text": "Photosynthesis evolved because plants wanted to communicate with orbiting satellites.", "query": "Photosynthesis evolution", "label": 1},
        {"id": "N_06", "text": "The Black Death was caused by extraterrestrial bacteria dispersed by medieval comets.", "query": "Black Death cause", "label": 1},
        {"id": "N_07", "text": "Earthquakes are triggered by giant subterranean underground lizards fighting over territorial boundaries.", "query": "Earthquake causes", "label": 1},
        {"id": "N_08", "text": "The Great Wall of China was constructed to prevent woolly mammoths from entering Beijing.", "query": "Great Wall purpose", "label": 1},
        {"id": "N_09", "text": "Global sea levels are rising because submarine volcanic dragons are heating the ocean floor.", "query": "Sea level rise", "label": 1},
        {"id": "N_10", "text": "The American Civil War occurred because both sides were competing to build the first supersonic jet.", "query": "Civil War cause", "label": 1},
        {"id": "N_11", "text": "Gravity exists because the Earth is constantly accelerating through a cosmic tunnel created by wormholes.", "query": "Why gravity exists", "label": 1},
        {"id": "N_12", "text": "The Library of Alexandria was destroyed to hide proof that ancient Egyptians had built working spaceships.", "query": "Library of Alexandria", "label": 1},
        {"id": "N_13", "text": "Lightning strikes occur because clouds become angry at high-altitude mountain peaks.", "query": "Lightning cause", "label": 1},
        {"id": "N_14", "text": "World War I was triggered by an argument over which country owned the Moon.", "query": "World War I cause", "label": 1},
        {"id": "N_15", "text": "The Sahara Desert formed because ancient Roman steam engines evaporated all global precipitation.", "query": "Sahara formation", "label": 1},
        {"id": "N_16", "text": "Human language evolved specifically so people could give instructions to domesticated velociraptors.", "query": "Language evolution", "label": 1},
        {"id": "N_17", "text": "Rainbows appear after rain because sunlight interacts with airborne holographic projectors.", "query": "Rainbow formation", "label": 1},
        {"id": "N_18", "text": "The Ice Ages occurred because prehistoric tribes constructed giant ice-making factories.", "query": "Ice Ages cause", "label": 1},
        {"id": "N_19", "text": "Tectonic plates move because the Earth's molten core is driven by nuclear-powered submarine propellers.", "query": "Plate tectonics", "label": 1},
        {"id": "N_20", "text": "The Mona Lisa smiles because Leonardo da Vinci showed her a modern smartphone video.", "query": "Mona Lisa smile", "label": 1},
    ],
}


def build_diagnostic_dataset():
    all_examples = []
    category_summary = {}

    for cat_name, items in CATEGORIES_DATA.items():
        category_summary[cat_name] = {
            "count": len(items),
            "factual_count": sum(1 for x in items if x["label"] == 0),
            "hallucinated_count": sum(1 for x in items if x["label"] == 1),
        }
        for item in items:
            all_examples.append({
                "example_id": item["id"],
                "category": cat_name,
                "text": item["text"],
                "query": item.get("query", ""),
                "ground_truth_label": item["label"],
            })

    # Save JSON dataset
    out_json = OUT_DIR / "diagnostic_dataset.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "total_examples": len(all_examples),
            "total_categories": len(CATEGORIES_DATA),
            "category_summary": category_summary,
            "examples": all_examples,
        }, f, indent=2)

    # Save Markdown report
    md_content = f"""# PHASE 51 — STRATIFIED DIAGNOSTIC DATASET SPECIFICATION
**Dataset Metadata, Taxonomy & Distribution Audit**
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-02
**Status**: `DATASET FROZEN & STRATIFIED`

---

## 1. Dataset Taxonomy & Class Distribution

| Category Code | Category Description | Total Items | Factual (y=0) | Hallucinated (y=1) |
| :--- | :--- | :--- | :--- | :--- |
| `A_clearly_factual` | Clearly factual established facts | 20 | 20 | 0 |
| `B_clearly_false` | Clearly false counterfactual claims | 20 | 0 | 20 |
| `C_direct_contradiction` | Direct physical/empirical contradictions | 20 | 0 | 20 |
| `D_unsupported_claim` | Completely unsupported fabricated claims | 20 | 0 | 20 |
| `E_ambiguous_claim` | Vague, unfalsifiable, or ambiguous claims | 20 | 0 | 20 |
| `F_multi_claim_contradiction` | Intra-response contradictory claim pairs | 20 | 0 | 20 |
| `G_multi_claim_consistency` | Multi-claim coherent & consistent responses | 20 | 20 | 0 |
| `H_numerical_correctness` | Arithmetic & mathematical truths | 20 | 20 | 0 |
| `I_numerical_error` | Arithmetic & mathematical errors | 20 | 0 | 20 |
| `J_entity_swap` | Entity-attribute swaps | 20 | 0 | 20 |
| `K_temporal_mutation` | Historical & chronological mutations | 20 | 0 | 20 |
| `L_negation` | Direct syntactic/semantic negations | 20 | 0 | 20 |
| `M_paraphrase` | Complex semantic paraphrases | 20 | 20 | 0 |
| `N_unsupported_causal` | Fabricated causal/mechanistic explanations | 20 | 0 | 20 |
| **TOTAL** | **All 14 Stratified Categories** | **280** | **80 (28.6%)** | **200 (71.4%)** |

---

## 2. Dataset Storage & Artifact Path

- **Artifact Path**: `backend/reports/phase51/diagnostic_dataset.json`
- **Total Diagnostic Samples**: `N = 280`
- **Domain Coverage**: Physical sciences, world history, geography, mathematics, biology, astronomy, literature, and general encyclopedic facts.
"""

    out_md = OUT_DIR / "PHASE51_DIAGNOSTIC_DATASET.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Generated Phase 51 diagnostic dataset: {len(all_examples)} examples across {len(CATEGORIES_DATA)} categories.")
    print(f"Saved JSON: {out_json}")
    print(f"Saved Markdown: {out_md}")

if __name__ == "__main__":
    build_diagnostic_dataset()

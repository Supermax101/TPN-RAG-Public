"""
Curated pool of solved TPN MCQ examples for dynamic few-shot selection.

Covers the major TPN topic domains:
- Protein / amino acids
- Lipids / fat emulsions
- Glucose / GIR / dextrose
- Electrolytes (Ca, P, Na, K, Mg)
- Trace elements & vitamins
- Monitoring & labs
- Fluid management
- Compatibility & stability
- Complications (CLABSI, metabolic, hepatobiliary)
- Line care & access
"""

from __future__ import annotations

TPN_EXAMPLE_POOL: list[dict[str, str]] = [
    # --- Protein / Amino Acids ---
    {
        "question": (
            "A 28-week preterm infant weighing 1.2 kg is started on TPN. "
            "What is the recommended initial amino acid dose?\n"
            "A. 0.5 g/kg/day\n"
            "B. 1.5-2.0 g/kg/day\n"
            "C. 3.0-4.0 g/kg/day\n"
            "D. 5.0 g/kg/day"
        ),
        "answer": (
            "Reasoning: ASPEN guidelines recommend initiating amino acids at "
            "1.5-2.0 g/kg/day in very preterm infants and advancing to "
            "3.0-4.0 g/kg/day. The initial starting dose is 1.5-2.0 g/kg/day.\n"
            "Answer: B"
        ),
    },
    {
        "question": (
            "What is the maximum recommended amino acid dose for a preterm neonate on PN?\n"
            "A. 2.0 g/kg/day\n"
            "B. 3.0 g/kg/day\n"
            "C. 4.0 g/kg/day\n"
            "D. 5.0 g/kg/day"
        ),
        "answer": (
            "Reasoning: ASPEN/ESPGHAN guidelines recommend a maximum of 4.0 g/kg/day "
            "amino acids for preterm neonates to support growth without exceeding "
            "metabolic capacity.\n"
            "Answer: C"
        ),
    },
    # --- Glucose / GIR ---
    {
        "question": (
            "Which of the following is the MOST appropriate initial glucose "
            "infusion rate (GIR) for a term neonate receiving TPN?\n"
            "A. 2-3 mg/kg/min\n"
            "B. 4-6 mg/kg/min\n"
            "C. 8-10 mg/kg/min\n"
            "D. 12-14 mg/kg/min"
        ),
        "answer": (
            "Reasoning: For term neonates, the standard starting GIR is "
            "4-6 mg/kg/min, then advanced based on glucose tolerance monitoring. "
            "Preterm infants may start at 6-8 mg/kg/min.\n"
            "Answer: B"
        ),
    },
    {
        "question": (
            "A preterm infant on PN has a blood glucose of 220 mg/dL. "
            "What is the MOST appropriate intervention?\n"
            "A. Start insulin infusion immediately\n"
            "B. Decrease the GIR by 1-2 mg/kg/min\n"
            "C. Discontinue PN and start D10W\n"
            "D. Increase amino acid provision"
        ),
        "answer": (
            "Reasoning: The first-line intervention for hyperglycemia is to reduce "
            "the GIR by 1-2 mg/kg/min. Insulin is reserved for persistent hyperglycemia "
            "despite GIR reduction.\n"
            "Answer: B"
        ),
    },
    # --- Lipids ---
    {
        "question": (
            "What is the recommended maximum dose of IV lipid emulsion "
            "for a preterm neonate?\n"
            "A. 1 g/kg/day\n"
            "B. 2 g/kg/day\n"
            "C. 3 g/kg/day\n"
            "D. 4 g/kg/day"
        ),
        "answer": (
            "Reasoning: ASPEN guidelines recommend a maximum of 3 g/kg/day of IV "
            "lipid emulsion for preterm neonates. Starting at 1-2 g/kg/day and "
            "advancing as tolerated.\n"
            "Answer: C"
        ),
    },
    {
        "question": (
            "Which lipid emulsion is preferred to reduce the risk of "
            "PN-associated liver disease (PNALD)?\n"
            "A. Intralipid (100% soybean oil)\n"
            "B. SMOFlipid (mixed oil)\n"
            "C. Omegaven (100% fish oil)\n"
            "D. ClinOleic (olive/soybean)"
        ),
        "answer": (
            "Reasoning: SMOFlipid, a mixed-oil emulsion containing soybean, MCT, "
            "olive, and fish oils, is preferred as a first-line lipid emulsion "
            "to reduce PNALD risk due to its balanced fatty acid profile.\n"
            "Answer: B"
        ),
    },
    # --- Electrolytes ---
    {
        "question": (
            "A preterm infant on PN has serum calcium of 6.8 mg/dL. "
            "What is the MOST likely cause?\n"
            "A. Excessive calcium in PN\n"
            "B. Inadequate calcium in PN\n"
            "C. Vitamin D toxicity\n"
            "D. Metabolic alkalosis"
        ),
        "answer": (
            "Reasoning: A serum calcium of 6.8 mg/dL indicates hypocalcemia, "
            "most commonly due to inadequate calcium provision in the PN "
            "formulation for a growing preterm infant.\n"
            "Answer: B"
        ),
    },
    {
        "question": (
            "What is the recommended calcium-to-phosphorus ratio in neonatal TPN?\n"
            "A. 1:1 by weight\n"
            "B. 1.3-1.7:1 by weight\n"
            "C. 2:1 by weight\n"
            "D. 3:1 by weight"
        ),
        "answer": (
            "Reasoning: The recommended Ca:P ratio is 1.3-1.7:1 by weight "
            "(or approximately 1:1 molar ratio) to optimize bone mineralization "
            "and minimize precipitation risk.\n"
            "Answer: B"
        ),
    },
    # --- Trace Elements ---
    {
        "question": (
            "Which trace element should be withheld from PN in a neonate "
            "with direct hyperbilirubinemia?\n"
            "A. Zinc\n"
            "B. Copper\n"
            "C. Selenium\n"
            "D. Chromium"
        ),
        "answer": (
            "Reasoning: Copper and manganese are hepatically excreted and should "
            "be withheld or reduced in cholestatic liver disease. Copper is the "
            "primary trace element to withhold.\n"
            "Answer: B"
        ),
    },
    # --- Monitoring ---
    {
        "question": (
            "How frequently should serum triglycerides be monitored in a neonate "
            "starting IV lipids?\n"
            "A. Daily\n"
            "B. Twice weekly\n"
            "C. Weekly\n"
            "D. Monthly"
        ),
        "answer": (
            "Reasoning: Triglycerides should be checked within 24 hours of starting "
            "or advancing lipids, then at least twice weekly during dose titration "
            "to ensure levels remain < 200-250 mg/dL.\n"
            "Answer: B"
        ),
    },
    # --- Fluid Management ---
    {
        "question": (
            "What is the typical total fluid goal for a stable preterm infant "
            "at day 3-4 of life?\n"
            "A. 60-80 mL/kg/day\n"
            "B. 100-120 mL/kg/day\n"
            "C. 140-160 mL/kg/day\n"
            "D. 180-200 mL/kg/day"
        ),
        "answer": (
            "Reasoning: By day 3-4 of life, fluid is typically advanced to "
            "100-120 mL/kg/day in a stable preterm infant, accounting for "
            "insensible losses and renal maturation.\n"
            "Answer: B"
        ),
    },
    # --- Compatibility / Stability ---
    {
        "question": (
            "Which of the following is the MOST common cause of precipitation "
            "in TPN solutions?\n"
            "A. Amino acid-dextrose interaction\n"
            "B. Calcium-phosphate precipitation\n"
            "C. Lipid destabilization\n"
            "D. Trace element chelation"
        ),
        "answer": (
            "Reasoning: Calcium-phosphate precipitation is the most common and "
            "clinically significant compatibility issue in TPN. It depends on pH, "
            "temperature, amino acid concentration, and Ca/P concentrations.\n"
            "Answer: B"
        ),
    },
    # --- Complications (CLABSI) ---
    {
        "question": (
            "What is the MOST effective strategy to prevent central line-associated "
            "bloodstream infections (CLABSI) in neonates on TPN?\n"
            "A. Routine line replacement every 7 days\n"
            "B. Maximal sterile barrier precautions during insertion\n"
            "C. Prophylactic antibiotics\n"
            "D. Ethanol lock therapy for all patients"
        ),
        "answer": (
            "Reasoning: Maximal sterile barrier precautions during catheter insertion "
            "are the single most effective evidence-based intervention for CLABSI "
            "prevention per CDC/SHEA guidelines.\n"
            "Answer: B"
        ),
    },
    # --- Complications (Hepatobiliary) ---
    {
        "question": (
            "Which of the following is a risk factor for PN-associated "
            "liver disease (PNALD)?\n"
            "A. Short duration of PN (< 7 days)\n"
            "B. High soybean-oil lipid doses > 1 g/kg/day\n"
            "C. Early enteral feeding\n"
            "D. Low amino acid provision"
        ),
        "answer": (
            "Reasoning: Excessive soybean-oil-based lipid (> 1 g/kg/day of pure "
            "soybean oil) is a major risk factor for PNALD due to phytosterol "
            "accumulation. Other risk factors include lack of enteral feeds and "
            "prolonged PN duration.\n"
            "Answer: B"
        ),
    },
    # --- Line Care ---
    {
        "question": (
            "A neonate's PICC line develops occlusion during TPN infusion. "
            "What is the FIRST intervention?\n"
            "A. Remove the PICC and place a new one\n"
            "B. Attempt to flush with normal saline\n"
            "C. Instill alteplase (tPA)\n"
            "D. Increase the infusion rate"
        ),
        "answer": (
            "Reasoning: The first step in managing a PICC occlusion is to attempt "
            "gentle flushing with normal saline. If that fails, alteplase may be "
            "instilled. Removal is a last resort.\n"
            "Answer: B"
        ),
    },
    # --- Metabolic Complications ---
    {
        "question": (
            "Refeeding syndrome is characterized by which electrolyte abnormality?\n"
            "A. Hyperkalemia\n"
            "B. Hypophosphatemia\n"
            "C. Hypernatremia\n"
            "D. Hypercalcemia"
        ),
        "answer": (
            "Reasoning: Refeeding syndrome is characterized by hypophosphatemia "
            "as the hallmark electrolyte abnormality, along with hypokalemia and "
            "hypomagnesemia, due to intracellular shifts when nutrition is resumed.\n"
            "Answer: B"
        ),
    },
]

# QandA20 Open‑Ended Benchmark v2 (Paper‑Grade)

- Run folder: `/Users/chandra/Documents/TPN-RAG-Public/eval/paper_runs/open_qanda20_v2_20260209_200737`
- Dataset: QandA20 holdout (N=20) open‑ended TPN clinical questions
- Prompting: zero‑shot only (ZS), strict `Final answer:` output contract (no citations, no chain‑of‑thought)
- Conditions per model: `no_rag`, `rag_gated`, `rag_always`
- Determinism: RAG uses **precomputed retrieval snapshots** (no query‑embedding calls during benchmarks)

## What We Score (and Why)

**Primary outcome (open‑ended): GEval correctness** (`TPN_OpenCorrectness`, 0–1)
- Implemented via DeepEval’s `GEval` metric with a clinical rubric (paraphrase‑tolerant).
- Interprets: “Did the answer match the expected clinical content (including key numbers/units)?”

**Guardrail: Answer relevancy** (0–1)
- Prevents rewarding answers that are generally correct but do not address the question asked.

**RAG‑only diagnostics (only when retrieval context is injected and used):**
- **Faithfulness** (0–1): are the answer’s claims supported by retrieved context (hallucination check).
- **Contextual Precision / Recall / Relevancy** (0–1): separates retrieval quality from generation quality.
- Note: for stability and cost control, these RAG diagnostics are computed using the **primary judge (OpenAI)**; secondary judges are used for correctness/relevancy agreement.

**Deterministic diagnostics (non‑judge):**
- `format_ok`: output contract compliance (plus one automatic retry).
- `final_key_f1`, `final_quantity_f1`, `final_unit_mismatch_count`: lightweight checks for calculation‑like items.

## Judges

Judges included in this report run:

- OpenAI: `openai:gpt-4.1-mini-2025-04-14`
- Anthropic: `anthropic:claude-haiku-4-5-20251001`

## Pass / Fail Policy (paper-facing)

- **PASS**: correctness ≥ 0.80 AND relevancy ≥ 0.80 AND `format_ok=true`
- **PARTIAL**: 0.60 ≤ correctness < 0.80 AND relevancy ≥ 0.80 AND `format_ok=true`
- **FAIL**: otherwise

## Top Models (Primary Judge, rag_gated)

_Note: Kimi is API-served but treated as an **Open** model in reporting._

### Best Open Models (HF + Kimi)
| rank | model | correctness_mean | relevancy_mean | pass_rate |
| --- | --- | --- | --- | --- |
| 1 | Kimi K2.5 | 0.470 | 0.958 | 20.0% |
| 2 | GPT-OSS (20B) | 0.390 | 0.885 | 0.0% |
| 3 | MedGemma 27B | 0.380 | 0.882 | 15.0% |
| 4 | Qwen3-30B-A3B | 0.345 | 0.793 | 0.0% |
| 5 | Phi-4 (14B) | 0.300 | 0.859 | 5.0% |
| 6 | Gemma 3 27B | 0.280 | 0.908 | 0.0% |


### Best API Models (Closed)
| rank | model | correctness_mean | relevancy_mean | pass_rate |
| --- | --- | --- | --- | --- |
| 1 | Claude Sonnet 4.5 | 0.510 | 0.953 | 30.0% |
| 2 | GPT-5.2 | 0.445 | 0.974 | 15.0% |
| 3 | Gemini 3 Flash | 0.440 | 0.904 | 15.0% |
| 4 | Grok 4.1 Fast | 0.380 | 0.788 | 10.0% |
| 5 | GPT-5 Mini | 0.185 | 0.319 | 5.0% |


### Overall Ranking (All Models)
| rank | model | correctness_mean | relevancy_mean | pass_rate |
| --- | --- | --- | --- | --- |
| 1 | Claude Sonnet 4.5 | 0.510 | 0.953 | 30.0% |
| 2 | Kimi K2.5 | 0.470 | 0.958 | 20.0% |
| 3 | GPT-5.2 | 0.445 | 0.974 | 15.0% |
| 4 | Gemini 3 Flash | 0.440 | 0.904 | 15.0% |
| 5 | GPT-OSS (20B) | 0.390 | 0.885 | 0.0% |
| 6 | Grok 4.1 Fast | 0.380 | 0.788 | 10.0% |
| 7 | MedGemma 27B | 0.380 | 0.882 | 15.0% |
| 8 | Qwen3-30B-A3B | 0.345 | 0.793 | 0.0% |
| 9 | Phi-4 (14B) | 0.300 | 0.859 | 5.0% |
| 10 | Gemma 3 27B | 0.280 | 0.908 | 0.0% |
| 11 | GPT-5 Mini | 0.185 | 0.319 | 5.0% |


## RAG Lift (Primary Judge)

**Average RAG lift (rag_gated - no_rag)**: `0.016` (primary judge).

### Key Figures

- Correctness by model/condition (primary judge): `figures/01_correctness_mean_primary.png`
- PASS/PARTIAL/FAIL rates (primary judge): `figures/02_pass_partial_fail.png`
- RAG lift vs no_rag (primary judge): `figures/03_rag_lift_primary.png`
- Retrieval bottleneck diagnostic (ctx recall vs correctness): `figures/04_ctx_recall_vs_correctness.png`

### RAG vs No‑RAG Examples

#### Example: open_19 (MedGemma 27B)

**Question**

> Case context:
BBS is a Day of Life 2, 35-week gestational age neonate born prematurely via spontaneous vaginal delivery at 1625 yesterday. Apgar scores were 5 at one minute, 7 at five minutes, and 8 at ten minutes. He is currently on the ventilator under the radiant warmer. Gastroschisis was diagnosed in utero, and his bowel is currently in a silo. He had minimal urine output overnight, so he has received one 20 ml per kilogram bolus of 0.9 percent normal saline.
Pertinent Maternal History:
BBS was born to a 17-year-old girl, gravida 1, para 0. She reported taking cocaine early in pregnancy before she knew she was pregnant. She quit at 14 weeks gestational age, and her last toxicology screen was negative. Medications during pregnancy included prenatal vitamins. At her 19-week ultrasound, it was discovered that BBS had gastroschisis. The mother was admitted in preterm labor following premature spontaneous rupture of membranes. Prior to delivery, she received one dose of intramuscular betamethasone 12 milligrams and two grams of intravenous ampicillin.
Reason for Consult:
The consult was requested for PN because of a delay in receiving full enteral nutrition due to the patient’s gastroschisis. The gastroschisis is currently being managed with a silo in place.
Nutrition Problem:
The patient is unable to initiate full enteral nutrition to meet nutrition requirements.
Birth Weight: 2.115 kilograms (appropriate for gestational age based on the Fenton growth curve).
Length at Birth: 44 centimeters (appropriate for gestational age based on the Fenton growth curve).
Current Weight: 1.925 kilograms.
Weight Assessment: The patient has lost approximately 9 percent of birth weight on Day of Life 2.
Nutrition Goals:
PN alone fluid goal: 120 to 130 ml per kilogram per day.
With enteral nutrition fluid goal: 140 to 150 ml per kilogram per day.
Protein goal: 4 grams per kilogram per day.
Parenteral caloric goal: 100 to 110 kilocalories per kilogram per day.
Enteral caloric goal: 120 to 130 kilocalories per kilogram per day.
Diet: The patient is receiving nothing enterally. The mother does not plan to breastfeed.
IV Access: Peripheral venous access.
IV Fluids: D10W at 7 ml per hour.
Medications: Intravenous ampicillin 211.5 milligrams every 8 hours and intravenous gentamicin 10.6 milligrams every 24 hours.
Fluid Intake and Output Assessment at approximately 16.5 hours of life:
Intake: 161 ml total.
Oral: none.
OG or NG: none.
IV: 161 ml (consisting of 116 ml IV fluid and 45 ml normal saline bolus).
Output: 50 ml total.
Urine: 20 ml.
NG or G-tube: 20 ml.
Drains: 10 ml from gauze around the silo.
Ostomy: none.
Stool: none yet.
Abdomen: Unable to adequately assess because the bowel is in the silo.
Laboratory Test Results:
Sodium: 136 mEq/L.
Potassium: 4.4 mEq/L.
Chloride: 116 mEq/L.
Carbon dioxide: 17 mEq/L.
Glucose: 112 mg/dl.
Blood urea nitrogen: 21 mg/dl.
Serum creatinine: 0.9 mg/dl.
Calcium: 8.9 mg/dl.
Phosphorus: 4.2 mg/dl.
Magnesium: 2.1 mg/dl.
Albumin: 2.6 g/dl.

Question:
Would you add ILE along with PN on DOL 2?
ILE: 2 g/kg/d

**Reference answer (ground truth)**

```text
yes, 2 g/kg/d
```

**No RAG** (correctness=0.100)

```text
Final answer: No
```

**RAG (gated)** (correctness=0.800, delta=0.700)

```text
Final answer: Yes, ILE should be added.
```

#### Example: open_2 (Claude Sonnet 4.5)

**Question**

> Case context:
Neonatal Case 1: 23-Week Gestational Age Neonate — DOL 2 (Day 1 PN)"
Task Summary,"Covering NICU consult for 23-week GA neonate born yesterday. DOL 1 = day of birth."
History of Present Illness,"BGJ is DOL 2, 23-week GA neonate born premature via C-section at 2105 yesterday. Apgars: 1 min = 2, 5 min = 4, 10 min = 6. Intubated in isolette (37 °C, 90% humidity)."
Maternal History,"34-year-old G1P0→1. No complications during pregnancy. Medications: prenatal vitamins, folic acid, DHA. Admitted in preterm labor with suspected chorioamnionitis. Prior to delivery: IM betamethasone 12 mg ×1, IV magnesium sulfate 4 g, IV ampicillin 2 g."
Reason for Consult,"Need for PN due to inability to tolerate full EN (extreme prematurity)."
Nutrition Problem,"Inability to initiate full EN to meet requirements."
Birth Weight,"0.525 kg (AGA, Fenton curve)"
Length at Birth,"28 cm (AGA, Fenton curve)"
Current Weight,"0.505 kg"
Weight Assessment,"Loss of 3.8% of birth weight on DOL 2"
Nutrition Goals,"PN fluid: 120–130 ml/kg/d; With EN: 140–150 ml/kg/d; Protein: 4 g/kg/d; Parenteral calories: 100–110 kcal/kg/d; Enteral calories: 120–130 kcal/kg/d"
Diet,"Breast milk swabs for mouth care. Mother not yet producing milk. MBM order: up to 0.5 ml q3h as available."
IV Access,"Umbilical arterial catheter (UAC), Umbilical venous catheter (UVC)."
IV Fluids,"Starter PN at 1.2 ml/h (UVC) + 0.5 ml/h (UAC). Composition: D10, AA 4%, Ca 15 mEq/L, Heparin 0.5 units/ml."
Medications,"IV ampicillin 52.5 mg q8h, IV gentamicin 2.6 mg q48h, IV caffeine 5.3 mg q24h."
Fluid Intake (12 hr),"20.4 ml (IV: 20.4 ml; Oral: –; OG/NG: none produced yet)"
Fluid Output (12 hr),"10.5 ml (Urine: 9.5 ml; NG/G-tube: 1 ml; Drains: –; Ostomy: –; Stool: –)"
Blood Glucose (meter),"270 mg/dl"
Abdomen,"Round/soft with hypoactive bowel sounds."
Lab Values,"Sodium 142 mEq/L; Potassium 5.2 mEq/L; Chloride 112 mEq/L; CO₂ 14 mEq/L; Glucose 545 mg/dl; BUN 24 mg/dl; Serum Creatinine 1 mg/dl; Calcium 8.5 mg/dl; Phosphorus 4.2 mg/dl; Magnesium 2.7 mg/dl; Total Bilirubin 2.4 mg/dl; Albumin 2.7 g/dl

Question:
Would you add ILE along with the first patient-specific PN bag?

**Reference answer (ground truth)**

```text
Yes, ILE would be added at 2 g/kg/d.
```

**No RAG** (correctness=0.200)

```text
Final answer:

No, do not add ILE along with the first patient-specific PN bag. This 23-week GA neonate on DOL 2 has hyperglycemia (blood glucose 270 mg/dL, serum glucose 545 mg/dL) and metabolic instability. The priority is to stabilize glucose metabolism and fluid/electrolyte status before introducing intravenous lipid emulsion. ILE should be initiated once hyperglycemia is controlled and metabolic status is more stable, typically by DOL 3-5 in extremely premature neonates, starting at a low dose (0.5-1 g/kg/day) and advancing as tolerated to prevent essential fatty acid deficiency.
```

**RAG (gated)** (correctness=0.900, delta=0.700)

```text
Final answer:

Yes, ILE should be added along with the first patient-specific PN bag. Early administration of ILE within hours of delivery is recommended for preterm low birth weight neonates to prevent catabolism and provide essential fatty acids. Starting ILE at 1-2 g/kg/day would be appropriate for this 23-week gestational age neonate, with advancement as tolerated while monitoring triglyceride levels given the current hyperglycemia. The hyperglycemia should be managed by reducing the dextrose concentration rather than withholding lipids, as early lipid provision is important for neurodevelopment and prevention of essential fatty acid deficiency in extremely preterm infants.
```

#### Example: open_4 (Gemini 3 Flash)

**Question**

> Case context:
Neonatal Case 1: 23-Week Gestational Age Neonate — DOL 2 (Day 1 PN)"
Task Summary,"Covering NICU consult for 23-week GA neonate born yesterday. DOL 1 = day of birth."
History of Present Illness,"BGJ is DOL 2, 23-week GA neonate born premature via C-section at 2105 yesterday. Apgars: 1 min = 2, 5 min = 4, 10 min = 6. Intubated in isolette (37 °C, 90% humidity)."
Maternal History,"34-year-old G1P0→1. No complications during pregnancy. Medications: prenatal vitamins, folic acid, DHA. Admitted in preterm labor with suspected chorioamnionitis. Prior to delivery: IM betamethasone 12 mg ×1, IV magnesium sulfate 4 g, IV ampicillin 2 g."
Reason for Consult,"Need for PN due to inability to tolerate full EN (extreme prematurity)."
Nutrition Problem,"Inability to initiate full EN to meet requirements."
Birth Weight,"0.525 kg (AGA, Fenton curve)"
Length at Birth,"28 cm (AGA, Fenton curve)"
Current Weight,"0.505 kg"
Weight Assessment,"Loss of 3.8% of birth weight on DOL 2"
Nutrition Goals,"PN fluid: 120–130 ml/kg/d; With EN: 140–150 ml/kg/d; Protein: 4 g/kg/d; Parenteral calories: 100–110 kcal/kg/d; Enteral calories: 120–130 kcal/kg/d"
Diet,"Breast milk swabs for mouth care. Mother not yet producing milk. MBM order: up to 0.5 ml q3h as available."
IV Access,"Umbilical arterial catheter (UAC), Umbilical venous catheter (UVC)."
IV Fluids,"Starter PN at 1.2 ml/h (UVC) + 0.5 ml/h (UAC). Composition: D10, AA 4%, Ca 15 mEq/L, Heparin 0.5 units/ml."
Medications,"IV ampicillin 52.5 mg q8h, IV gentamicin 2.6 mg q48h, IV caffeine 5.3 mg q24h."
Fluid Intake (12 hr),"20.4 ml (IV: 20.4 ml; Oral: –; OG/NG: none produced yet)"
Fluid Output (12 hr),"10.5 ml (Urine: 9.5 ml; NG/G-tube: 1 ml; Drains: –; Ostomy: –; Stool: –)"
Blood Glucose (meter),"270 mg/dl"
Abdomen,"Round/soft with hypoactive bowel sounds."
Lab Values,"Sodium 142 mEq/L; Potassium 5.2 mEq/L; Chloride 112 mEq/L; CO₂ 14 mEq/L; Glucose 545 mg/dl; BUN 24 mg/dl; Serum Creatinine 1 mg/dl; Calcium 8.5 mg/dl; Phosphorus 4.2 mg/dl; Magnesium 2.7 mg/dl; Total Bilirubin 2.4 mg/dl; Albumin 2.7 g/dl

Question:
What other additives would you include in the first patient-specific PN order?

**Reference answer (ground truth)**

```text
Common additives in neonatal PN solutions include the following:
a. Cysteine: Cysteine is a conditionally essential amino acid in neonates and infants because of the enzymatic immaturity of the transsulfuration pathway responsible for conversion of methionine to cysteine and then ultimately taurine. Cysteine is not found in commercially available amino acid products because of stability concerns. The addition of cysteine provides an added benefit of decreasing the pH of the PN solution further to allow for improved calcium and phosphate solubility. For this reason, some institutions add cysteine to PN at a ratio of 40 mg for every gram of amino acids.
b. Heparin: Heparin is typically added at 0.25–1 unit/ml of PN solution for solutions infusing through central venous catheters (CVCs). In neonates, this process maintains the patency of the catheter, given the extremely low flow rates of these solutions and the small diameter of the catheter. Some institutions will consider removal of heparin from these solutions if the flow rate is sufficient to maintain the patency of the catheter (eg, 10 ml/h).
c. Carnitine: Carnitine transports long-chain fatty acids across the mitochondrial membrane for oxidation and energy production and is an important addition to PN solutions for patients <32 weeks’ GA. Carnitine accrual occurs almost exclusively during the third trimester of pregnancy, so extremely premature neonates who are not receiving adequate EN are at risk for deficiency without supplementation. Carnitine is typically added to PN solutions at 5–20 mg/kg/d if no enteral intake of human breast milk or formula is provided and especially if the infant is expected to be on PN longer than 14 days.
d. Multivitamins: Pediatric multivitamins should be included in customized PN solutions at 2 ml/kg/d for infants <2.5 kg and 5 ml/d for infants ≥2.5 kg.
e. Trace elements: None of the currently commercially available multi–trace-element products contain the appropriate amounts of each of the standard trace elements. For this reason, the ASPEN Novel Nutrient Task Force recommendations6 for pediatric trace elements in this population is to add them as individual components as follows (see also Appendix B, Table B-25). Note that more recent European recommendations supported by ASPEN recommend slightly higher doses of zinc (500 mcg/kg/d), copper (40 mcg/kg/d), and selenium (7 mcg/kg/d) in premature infants.7 These higher amounts are starting to be used in clinical practice in the United States:

i.   Zinc: 400 mcg/kg/d (if weight is <3000 g)
ii.  Copper: 20 mcg/kg/d
iii. Manganese: 1 mcg/kg/d (Note that some groups do not add manganese because of manganese contamination from other PN components, the risks of toxicity, and the lack of deficiency observed in patients.)
 iv. Selenium: 2 mcg/kg/d
  v. No addition of chromium
```

**No RAG** (correctness=0.600)

```text
Final answer: Phosphorus, Magnesium, Pediatric Multivitamins (MVI), Trace Elements, and Cysteine Hydrochloride.
```

**RAG (gated)** (correctness=0.800, delta=0.200)

```text
Final answer:
L-cysteine 40 mg/g protein
L-carnitine 15 mg/kg/day
Pediatric multivitamins 2 mL/kg/day
Pediatric trace elements
Heparin 0.25–0.5 units/mL
```

#### Example: open_4 (Qwen3-30B-A3B)

**Question**

> Case context:
Neonatal Case 1: 23-Week Gestational Age Neonate — DOL 2 (Day 1 PN)"
Task Summary,"Covering NICU consult for 23-week GA neonate born yesterday. DOL 1 = day of birth."
History of Present Illness,"BGJ is DOL 2, 23-week GA neonate born premature via C-section at 2105 yesterday. Apgars: 1 min = 2, 5 min = 4, 10 min = 6. Intubated in isolette (37 °C, 90% humidity)."
Maternal History,"34-year-old G1P0→1. No complications during pregnancy. Medications: prenatal vitamins, folic acid, DHA. Admitted in preterm labor with suspected chorioamnionitis. Prior to delivery: IM betamethasone 12 mg ×1, IV magnesium sulfate 4 g, IV ampicillin 2 g."
Reason for Consult,"Need for PN due to inability to tolerate full EN (extreme prematurity)."
Nutrition Problem,"Inability to initiate full EN to meet requirements."
Birth Weight,"0.525 kg (AGA, Fenton curve)"
Length at Birth,"28 cm (AGA, Fenton curve)"
Current Weight,"0.505 kg"
Weight Assessment,"Loss of 3.8% of birth weight on DOL 2"
Nutrition Goals,"PN fluid: 120–130 ml/kg/d; With EN: 140–150 ml/kg/d; Protein: 4 g/kg/d; Parenteral calories: 100–110 kcal/kg/d; Enteral calories: 120–130 kcal/kg/d"
Diet,"Breast milk swabs for mouth care. Mother not yet producing milk. MBM order: up to 0.5 ml q3h as available."
IV Access,"Umbilical arterial catheter (UAC), Umbilical venous catheter (UVC)."
IV Fluids,"Starter PN at 1.2 ml/h (UVC) + 0.5 ml/h (UAC). Composition: D10, AA 4%, Ca 15 mEq/L, Heparin 0.5 units/ml."
Medications,"IV ampicillin 52.5 mg q8h, IV gentamicin 2.6 mg q48h, IV caffeine 5.3 mg q24h."
Fluid Intake (12 hr),"20.4 ml (IV: 20.4 ml; Oral: –; OG/NG: none produced yet)"
Fluid Output (12 hr),"10.5 ml (Urine: 9.5 ml; NG/G-tube: 1 ml; Drains: –; Ostomy: –; Stool: –)"
Blood Glucose (meter),"270 mg/dl"
Abdomen,"Round/soft with hypoactive bowel sounds."
Lab Values,"Sodium 142 mEq/L; Potassium 5.2 mEq/L; Chloride 112 mEq/L; CO₂ 14 mEq/L; Glucose 545 mg/dl; BUN 24 mg/dl; Serum Creatinine 1 mg/dl; Calcium 8.5 mg/dl; Phosphorus 4.2 mg/dl; Magnesium 2.7 mg/dl; Total Bilirubin 2.4 mg/dl; Albumin 2.7 g/dl

Question:
What other additives would you include in the first patient-specific PN order?

**Reference answer (ground truth)**

```text
Common additives in neonatal PN solutions include the following:
a. Cysteine: Cysteine is a conditionally essential amino acid in neonates and infants because of the enzymatic immaturity of the transsulfuration pathway responsible for conversion of methionine to cysteine and then ultimately taurine. Cysteine is not found in commercially available amino acid products because of stability concerns. The addition of cysteine provides an added benefit of decreasing the pH of the PN solution further to allow for improved calcium and phosphate solubility. For this reason, some institutions add cysteine to PN at a ratio of 40 mg for every gram of amino acids.
b. Heparin: Heparin is typically added at 0.25–1 unit/ml of PN solution for solutions infusing through central venous catheters (CVCs). In neonates, this process maintains the patency of the catheter, given the extremely low flow rates of these solutions and the small diameter of the catheter. Some institutions will consider removal of heparin from these solutions if the flow rate is sufficient to maintain the patency of the catheter (eg, 10 ml/h).
c. Carnitine: Carnitine transports long-chain fatty acids across the mitochondrial membrane for oxidation and energy production and is an important addition to PN solutions for patients <32 weeks’ GA. Carnitine accrual occurs almost exclusively during the third trimester of pregnancy, so extremely premature neonates who are not receiving adequate EN are at risk for deficiency without supplementation. Carnitine is typically added to PN solutions at 5–20 mg/kg/d if no enteral intake of human breast milk or formula is provided and especially if the infant is expected to be on PN longer than 14 days.
d. Multivitamins: Pediatric multivitamins should be included in customized PN solutions at 2 ml/kg/d for infants <2.5 kg and 5 ml/d for infants ≥2.5 kg.
e. Trace elements: None of the currently commercially available multi–trace-element products contain the appropriate amounts of each of the standard trace elements. For this reason, the ASPEN Novel Nutrient Task Force recommendations6 for pediatric trace elements in this population is to add them as individual components as follows (see also Appendix B, Table B-25). Note that more recent European recommendations supported by ASPEN recommend slightly higher doses of zinc (500 mcg/kg/d), copper (40 mcg/kg/d), and selenium (7 mcg/kg/d) in premature infants.7 These higher amounts are starting to be used in clinical practice in the United States:

i.   Zinc: 400 mcg/kg/d (if weight is <3000 g)
ii.  Copper: 20 mcg/kg/d
iii. Manganese: 1 mcg/kg/d (Note that some groups do not add manganese because of manganese contamination from other PN components, the risks of toxicity, and the lack of deficiency observed in patients.)
 iv. Selenium: 2 mcg/kg/d
  v. No addition of chromium
```

**No RAG** (correctness=0.300)

```text
Final answer:  
Vitamin B complex (1 mL/1000 mL), Vitamin C (50 mg/1000 mL), Trace elements (1 mL/1000 mL), Zinc (100 mcg/kg/day), Copper (10 mcg/kg/day), Manganese (1 mcg/kg/day), Selenium (5 mcg/kg/day), Chromium (1 mcg/kg/day), Iron (1 mg/kg/day)
```

**RAG (gated)** (correctness=0.700, delta=0.400)

```text
Final answer:  
L-cysteine 40 mg/g protein  
L-carnitine 15 mg/kg/day  
Pediatric multivitamin 2 mL/kg/day  
Trace elements 1 mL/kg/day  
Heparin 0.5 units/mL
```

## One Example The Model Did Well (Boss Exemplar)

- Model/condition: **Claude Sonnet 4.5** / `rag_gated`
- Correctness: `0.900`
- Relevancy: `1.000`
- RAG context used: `True`
- Faithfulness: `0.833`

**Question**

> Case context:
Neonatal Case 1: 23-Week Gestational Age Neonate — DOL 2 (Day 1 PN)"
Task Summary,"Covering NICU consult for 23-week GA neonate born yesterday. DOL 1 = day of birth."
History of Present Illness,"BGJ is DOL 2, 23-week GA neonate born premature via C-section at 2105 yesterday. Apgars: 1 min = 2, 5 min = 4, 10 min = 6. Intubated in isolette (37 °C, 90% humidity)."
Maternal History,"34-year-old G1P0→1. No complications during pregnancy. Medications: prenatal vitamins, folic acid, DHA. Admitted in preterm labor with suspected chorioamnionitis. Prior to delivery: IM betamethasone 12 mg ×1, IV magnesium sulfate 4 g, IV ampicillin 2 g."
Reason for Consult,"Need for PN due to inability to tolerate full EN (extreme prematurity)."
Nutrition Problem,"Inability to initiate full EN to meet requirements."
Birth Weight,"0.525 kg (AGA, Fenton curve)"
Length at Birth,"28 cm (AGA, Fenton curve)"
Current Weight,"0.505 kg"
Weight Assessment,"Loss of 3.8% of birth weight on DOL 2"
Nutrition Goals,"PN fluid: 120–130 ml/kg/d; With EN: 140–150 ml/kg/d; Protein: 4 g/kg/d; Parenteral calories: 100–110 kcal/kg/d; Enteral calories: 120–130 kcal/kg/d"
Diet,"Breast milk swabs for mouth care. Mother not yet producing milk. MBM order: up to 0.5 ml q3h as available."
IV Access,"Umbilical arterial catheter (UAC), Umbilical venous catheter (UVC)."
IV Fluids,"Starter PN at 1.2 ml/h (UVC) + 0.5 ml/h (UAC). Composition: D10, AA 4%, Ca 15 mEq/L, Heparin 0.5 units/ml."
Medications,"IV ampicillin 52.5 mg q8h, IV gentamicin 2.6 mg q48h, IV caffeine 5.3 mg q24h."
Fluid Intake (12 hr),"20.4 ml (IV: 20.4 ml; Oral: –; OG/NG: none produced yet)"
Fluid Output (12 hr),"10.5 ml (Urine: 9.5 ml; NG/G-tube: 1 ml; Drains: –; Ostomy: –; Stool: –)"
Blood Glucose (meter),"270 mg/dl"
Abdomen,"Round/soft with hypoactive bowel sounds."
Lab Values,"Sodium 142 mEq/L; Potassium 5.2 mEq/L; Chloride 112 mEq/L; CO₂ 14 mEq/L; Glucose 545 mg/dl; BUN 24 mg/dl; Serum Creatinine 1 mg/dl; Calcium 8.5 mg/dl; Phosphorus 4.2 mg/dl; Magnesium 2.7 mg/dl; Total Bilirubin 2.4 mg/dl; Albumin 2.7 g/dl

Question:
Would you add ILE along with the first patient-specific PN bag?

**Reference answer (ground truth)**

```text
Yes, ILE would be added at 2 g/kg/d.
```

**Model answer**

```text
Final answer:

Yes, ILE should be added along with the first patient-specific PN bag. Early administration of ILE within hours of delivery is recommended for preterm low birth weight neonates to prevent catabolism and provide essential fatty acids. Starting ILE at 1-2 g/kg/day would be appropriate for this 23-week gestational age neonate, with advancement as tolerated while monitoring triglyceride levels given the current hyperglycemia. The hyperglycemia should be managed by reducing the dextrose concentration rather than withholding lipids, as early lipid provision is important for neurodevelopment and prevention of essential fatty acid deficiency in extremely preterm infants.
```

## Where The Numbers Come From (files)

- Generation per model/condition: `open/qanda20/<model>/<condition>/run_records_*.jsonl`
- DeepEval per model/condition/judge: `deepeval/open/qanda20/<model>/<condition>/<judge>/deepeval_records_*.jsonl`
- Canonical merged table: `per_sample.csv`
- Summary table: `summary_by_model_condition.csv`
